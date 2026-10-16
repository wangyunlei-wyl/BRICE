import torch
import torch.nn as nn
from models.utilss import PointNet_SA_Module_KNN, MLP_Res, fps_subsample, Transformer, cross_transformer
from models.SPD import SPD
import torch.nn.functional as F
from extensions.chamfer_dist import ChamferDistanceL1

from models.crapcn import *

# VA from Point Transformer
class VectorAttention(nn.Module):
    def __init__(self, in_channel=128, dim=64, n_knn=16, attn_hidden_multiplier=4):
        super().__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )
        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, query, support):
        pq, fq = query
        ps, fs = support

        identity = fq
        query, key, value = self.conv_query(fq), self.conv_key(fs), self.conv_value(fs)

        B, D, N = query.shape

        pos_flipped_1 = ps.permute(0, 2, 1).contiguous()
        pos_flipped_2 = pq.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped_1, pos_flipped_2)

        key = grouping_operation(key, idx_knn)
        qk_rel = query.reshape((B, -1, N, 1)) - key

        pos_rel = pq.reshape((B, -1, N, 1)) - grouping_operation(ps, idx_knn)
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = grouping_operation(value, idx_knn) + pos_embedding
        agg = einsum('b c i j, b c i j -> b c i', attention, value)
        output = self.conv_end(agg) + identity

        return output


# project f from p1 onto p2
def three_inter(f, p1, p2):
    # print(f.shape, p1.shape, p2.shape)
    # p1_flipped = p1.permute(0, 2, 1).contiguous()
    # p2_flipped = p2.permute(0, 2, 1).contiguous()
    idx, dis = get_nearest_index(p2, p1, k=3, return_dis=True)
    dist_recip = 1.0 / (dis + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    proj_f = torch.sum(indexing_neighbor(f, idx) * weight.unsqueeze(1), dim=-1)
    return proj_f


def hierarchical_fps(pts, rates):
    pts_flipped = pts.permute(0, 2, 1).contiguous()
    B, _, N = pts.shape
    now = N
    fps_idxs = []
    for i in range(len(rates)):
        now = now // rates[i]
        if now == N:
            fps_idxs.append(None)
        else:
            fps_idxs.append(furthest_point_sample(pts_flipped, now))
    return fps_idxs


# Cross-Resolution Transformer
class CRT(nn.Module):
    def __init__(self, dim_in=128, is_inter=True, down_rates=[1, 4, 2], knns=[16, 12, 8]):
        super().__init__()
        self.down_rates = down_rates
        self.is_inter = is_inter
        self.num_scale = len(down_rates)

        self.attn_lists = nn.ModuleList()
        self.q_mlp_lists = nn.ModuleList()
        self.s_mlp_lists = nn.ModuleList()
        for i in range(self.num_scale):
            self.attn_lists.append(VectorAttention(in_channel=dim_in, dim=64, n_knn=knns[i]))

        for i in range(self.num_scale - 1):
            self.q_mlp_lists.append(MLP_Res(in_dim=128 * 2, hidden_dim=128, out_dim=128))
            self.s_mlp_lists.append(MLP_Res(in_dim=128 * 2, hidden_dim=128, out_dim=128))

    def forward(self, query, support, fps_idxs_q=None, fps_idxs_s=None):
        pq, fq = query
        ps, fs = support
        # prepare fps_idxs_q and fps_idxs_s
        if fps_idxs_q == None:
            fps_idxs_q = hierarchical_fps(pq, self.down_rates)

        if fps_idxs_s == None:
            if self.is_inter:
                fps_idxs_s = hierarchical_fps(ps, self.down_rates)  # inter-level
            else:
                fps_idxs_s = fps_idxs_q  # intra-level

        # top-down aggregation
        pre_f = None
        pre_pos = None

        for i in range(self.num_scale - 1, -1, -1):
            if fps_idxs_q[i] == None:
                _pos1 = pq
            else:
                _pos1 = gather_operation(pq, fps_idxs_q[i])

            if fps_idxs_s[i] == None:
                _pos2 = ps
            else:
                _pos2 = gather_operation(ps, fps_idxs_s[i])

            if i == self.num_scale - 1:
                if fps_idxs_q[i] == None:
                    _f1 = fq
                else:
                    _f1 = gather_operation(fq, fps_idxs_q[i])
                if fps_idxs_s[i] == None:
                    _f2 = fs
                else:
                    _f2 = gather_operation(fs, fps_idxs_s[i])

            else:
                proj_f1 = three_inter(pre_f, pre_pos, _pos1)
                proj_f2 = three_inter(pre_f, pre_pos, _pos2)
                if fps_idxs_q[i] == None:
                    _f1 = fq
                else:
                    _f1 = gather_operation(fq, fps_idxs_q[i])
                if fps_idxs_s[i] == None:
                    _f2 = fs
                else:
                    _f2 = gather_operation(fs, fps_idxs_s[i])

                _f1 = self.q_mlp_lists[i](torch.cat([_f1, proj_f1], dim=1))
                _f2 = self.s_mlp_lists[i](torch.cat([_f2, proj_f2], dim=1))

            f = self.attn_lists[i]([_pos1, _f1], [_pos2, _f2])

            pre_f = f
            pre_pos = _pos1

        agg_f = pre_f
        return agg_f, fps_idxs_q, fps_idxs_s

class Encoder_sa1(nn.Module):
    def __init__(self):
        super(Encoder_sa1, self).__init__()
        self.sa = PointNet_SA_Module_KNN(
            512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True
        )
        self.attn = Transformer(128, dim=64)

    def forward(self, point_cloud):
        l0_xyz = point_cloud
        l0_points = point_cloud
        l1_xyz, l1_points, _ = self.sa(l0_xyz, l0_points)
        l1_points = self.attn(l1_points, l1_xyz)
        return l1_xyz, l1_points.contiguous()


class Encoder_sa2(nn.Module):
    def __init__(self):
        super(Encoder_sa2, self).__init__()
        self.sa = PointNet_SA_Module_KNN(
            256, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True
        )
        self.attn = Transformer(256, dim=128)

    def forward(self, l1_xyz, l1_points):
        l2_xyz, l2_points, _ = self.sa(l1_xyz, l1_points)
        l2_points = self.attn(l2_points, l2_xyz)
        return l2_xyz, l2_points.contiguous()


class Encoder_sa3(nn.Module):
    def __init__(self):
        super(Encoder_sa3, self).__init__()
        self.sa = PointNet_SA_Module_KNN(
            128, 16, 256, [256, 512], group_all=False, if_bn=False, if_idx=True
        )

    def forward(self, l2_xyz, l2_points):
        l3_xyz, l3_points, _ = self.sa(l2_xyz, l2_points)
        return l3_xyz, l3_points

class Decoder_d2(nn.Module):
    def __init__(self, channel=64):
        super(Decoder_d2, self).__init__()
        self.channel = channel
        self.conv_out_2 = nn.Conv1d(channel * 2, 3, kernel_size=1)
        self.conv_out1_2 = nn.Conv1d(channel * 4, channel * 2, kernel_size=1)
        self.ps_2 = nn.ConvTranspose1d(channel * 8, channel, 128, bias=True)
        self.ps_refuse_2 = nn.Conv1d(channel, channel * 8, kernel_size=1)
        self.ps_adj_2 = nn.Conv1d(channel * 8, channel * 8, kernel_size=1)
        self.relu = nn.GELU()

    def forward(self, x, batch_size, N):
        x_g_2 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1)
        x = self.relu(self.ps_adj_2(x_g_2))
        x = self.relu(self.ps_2(x))
        x = self.relu(self.ps_refuse_2(x)).reshape(batch_size, self.channel * 4, N // 8)
        x = self.conv_out_2(self.relu(self.conv_out1_2(x)))
        return x

class Decoder_d1(nn.Module):
    def __init__(self, channel=64):
        super(Decoder_d1, self).__init__()
        self.channel = channel
        self.conv_out_1 = nn.Conv1d(channel, 3, kernel_size=1)
        self.conv_out1_1 = nn.Conv1d(channel * 2, channel, kernel_size=1)
        self.ps_1 = nn.ConvTranspose1d(channel * 4, channel, 256, bias=True)
        self.ps_refuse_1 = nn.Conv1d(channel, channel * 4, kernel_size=1)
        self.ps_adj_1 = nn.Conv1d(channel * 4, channel * 4, kernel_size=1)
        self.relu = nn.GELU()

    def forward(self, x, batch_size, N):
        x_g_1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1)
        x = self.relu(self.ps_adj_1(x_g_1))
        x = self.relu(self.ps_1(x))
        x = self.relu(self.ps_refuse_1(x)).reshape(batch_size, self.channel * 2, N // 4)
        x = self.conv_out_1(self.relu(self.conv_out1_1(x)))
        return x

class Decoder_d0(nn.Module):
    def __init__(self, channel=64):
        super(Decoder_d0, self).__init__()
        self.channel = channel
        self.conv_out_0 = nn.Conv1d(32, 3, kernel_size=1)
        self.conv_out1_0 = nn.Conv1d(channel // 2, 32, kernel_size=1)
        self.ps_0 = nn.ConvTranspose1d(channel * 2, channel, 512, bias=True)
        self.ps_refuse_0 = nn.Conv1d(channel, channel * 2, kernel_size=1)
        self.ps_adj_0 = nn.Conv1d(channel * 2, channel * 2, kernel_size=1)
        self.relu = nn.GELU()

    def forward(self, x, batch_size, N):
        x_g_0 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1)
        x = self.relu(self.ps_adj_0(x_g_0))
        x = self.relu(self.ps_0(x))
        x = self.relu(self.ps_refuse_0(x)).reshape(batch_size, self.channel // 2, N)
        x = self.conv_out_0(self.relu(self.conv_out1_0(x)))
        return x

class IC_Encoder(nn.Module):
    def __init__(self, timesteps=3, alpha=0.01, beta=0.8, lambdax=0.1, channel=64, pretrain=''):
        super().__init__()
        self.channel = channel
        self.timesteps = timesteps
        self.alpha = alpha
        self.beta = beta
        self.lambdax = lambdax
        print("alpha: ", self.alpha, "\nbeta: ", self.beta, "\nlambdax: ", self.lambdax)

        self.encoder_sa1 = Encoder_sa1()
        self.encoder_sa2 = Encoder_sa2()
        self.encoder_sa3 = Encoder_sa3()


        self.feedback_map_d1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1)
        )
        self.feedback_map_d2 = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )

        self.decoder_d2 = Decoder_d2(channel=self.channel)
        self.decoder_d1 = Decoder_d1(channel=self.channel)
        self.decoder_d0 = Decoder_d0(channel=self.channel)

        self.pretrain = pretrain
        self._load_pretrained_weights()
        self.loss_func = ChamferDistanceL1()

    def _load_pretrained_weights(self):
        pretrain = self.pretrain
        if pretrain:
            print(f'Use pretrained model {pretrain}')
            state_dict = torch.load(pretrain, map_location='cpu')
            weights_dict = {}
            model_param_names = set([name for name, _ in self.named_parameters()])
            model_buffer_names = set([name for name, _ in self.named_buffers()])
            model_all_names = model_param_names.union(model_buffer_names)
            skipped_names = []
            for k, v in state_dict.items():
                new_k = k
                if 'module.' in new_k:
                    new_k = new_k.replace('module.', '')
                if new_k.startswith('encoder.'):
                    new_k = new_k[len('encoder.'):]
                if 'sa_mlp_conv' in new_k:
                    new_k = new_k.replace('sa_mlp_conv', 'sa.mlp_conv')

                if new_k in model_all_names:
                    weights_dict[new_k] = v
                else:
                    skipped_names.append(new_k)
            load_result = self.load_state_dict(weights_dict)
        else:
            print('Not Use pretrained model.')

    def forward(self, pc):
        pc = pc.permute(0, 2, 1).contiguous()
        batch_size, _, N = pc.size()
        input = pc  # (B, 3, N)

        e1_feed_static, l1_feed_static = self.encoder_sa1(input)
        e1_curr, l1_curr = e1_feed_static, l1_feed_static
        e2_curr, l2_curr = self.encoder_sa2(e1_curr, l1_curr)
        e3_curr, l3_curr = self.encoder_sa3(e2_curr, l2_curr)

        if self.timesteps > 0:
            for t in range(self.timesteps):
                with torch.enable_grad():
                    l1_in = l1_curr.detach().requires_grad_(True)
                    l2_in = l2_curr.detach().requires_grad_(True)
                    l3_in = l3_curr.detach().requires_grad_(True)

                    d2_pred = self.decoder_d2(l3_in, batch_size, N)
                    d1_pred = self.decoder_d1(l2_in, batch_size, N)
                    d0_pred = self.decoder_d0(l1_in, batch_size, N)

                    d1_feedback = self.feedback_map_d1(d1_pred)
                    d2_feedback = self.feedback_map_d2(d2_pred)

                    loss_e2 = self.loss_func(d2_pred.transpose(1, 2), e2_curr.detach().transpose(1, 2))
                    loss_e1 = self.loss_func(d1_pred.transpose(1, 2), e1_curr.detach().transpose(1, 2))
                    loss_e0 = self.loss_func(d0_pred.transpose(1, 2), input.transpose(1, 2))

                    # loss_en += (loss_e2 + loss_e1 + loss_e0)

                    grad_l3 = torch.autograd.grad(loss_e2, l3_in, create_graph=True)[0]
                    grad_l2 = torch.autograd.grad(loss_e1, l2_in, create_graph=True)[0]
                    grad_l1 = torch.autograd.grad(loss_e0, l1_in, create_graph=True)[0]

                l1_next = (self.beta * l1_feed_static
                           - self.alpha * grad_l1
                           + self.lambdax * d1_feedback
                           + (1 - self.beta - self.lambdax) * l1_curr)
                e1_next = e1_feed_static

                e2_feed, l2_feed = self.encoder_sa2(e1_next, l1_next)
                l2_next = (self.beta * l2_feed
                           - self.alpha * grad_l2
                           + self.lambdax * d2_feedback
                           + (1 - self.beta - self.lambdax) * l2_curr)
                e2_next = e2_feed

                e3_feed, l3_feed = self.encoder_sa3(e2_next, l2_next)
                l3_next = (self.beta * l3_feed
                           - self.alpha * grad_l3
                           + (1 - self.beta) * l3_curr)
                e3_next = e3_feed

                e1_curr, l1_curr = e1_next, l1_next
                e2_curr, l2_curr = e2_next, l2_next
                e3_curr, l3_curr = e3_next, l3_next

        global_f = F.adaptive_max_pool1d(l3_curr, 1).view(batch_size, -1).unsqueeze(-1)
        return global_f


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion


class Decoder(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512,
                 radius=1, bounding=True, up_factors=None):
        super(Decoder, self).__init__()
        self.num_p0 = num_p0
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, bounding=bounding, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, feat, partial, return_P0=False):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        arr_pcd = []
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  # (B, num_pc, 3)
        arr_pcd.append(pcd)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0)
        if return_P0:
            arr_pcd.append(pcd)
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, feat, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd


class SnowflakeNet(nn.Module):
    def __init__(self, pretrain='', **kwargs):

        """
        Args:
            dim_feat: int, dimension of global feature
            num_pc: int
            num_p0: int
            radius: searching radius
            bounding： boolean, bound the displacements to [-1, 1] with tanh
            up_factors: list of int
        """

        dim_feat = kwargs.get('dim_feat', 512)
        num_pc = kwargs.get('num_pc', 256)
        num_p0 = kwargs.get('num_p0', 512)
        radius = kwargs.get('radius', 1)
        bounding = kwargs.get('bounding', True)
        up_factors = kwargs.get('up_factors', [1, 4])    # MVP
        # up_factors = kwargs.get('up_factors', [1, 4])   # PCN & PlantPCom

        super(SnowflakeNet, self).__init__()
        self.feat_extractor = IC_Encoder(pretrain=pretrain)
        self.decoder = Decoder(dim_feat=dim_feat, num_pc=num_pc, num_p0=num_p0,
                               radius=radius, bounding=bounding, up_factors=up_factors)

    def forward(self, point_cloud, return_P0=False):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        pcd_bnc = point_cloud
        # point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        feat = self.feat_extractor(point_cloud)
        out = self.decoder(feat, pcd_bnc, return_P0=return_P0)
        return out

if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_part = torch.randn(1, 2048, 3, requires_grad=True).to(device)
    model = SnowflakeNet(pretrain='').to(device)

    all_pc = model(x_part)
    for i in all_pc:
        print(i.shape)

