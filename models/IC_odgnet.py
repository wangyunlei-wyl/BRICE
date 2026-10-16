import torch.utils.data
from models.crapcn import *
from utils.loss_utils import chamfer_sqrt
from models.pointnet import UpLayer
import os

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn_point(k, xyz, new_xyz):
    """
    Input:
        x: point cloud, [B, N, 3]
    Return:
        group_xyz: [B, N, K]
    """
    dist = square_distance(new_xyz, xyz)
    _, group_idx = dist.topk(k, largest=False)
    return group_idx.int()


class Transformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """

        # 点云的self-attention计算的是一个领域内点与点之间的关系

        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()

        idx_knn = query_knn_point(self.n_knn, pos_flipped, pos_flipped)

        key = self.conv_key(x)

        value = self.conv_value(x)

        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn

        qk_rel = query.reshape((b, -1, n, 1)) - key

        # 获取点与
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn

        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = grouping_operation(value, idx_knn) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y + identity

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

        e1_curr, l1_curr = self.encoder_sa1(input)
        e2_curr, l2_curr = self.encoder_sa2(e1_curr, l1_curr)
        e3_curr, l3_curr = self.encoder_sa3(e2_curr, l2_curr)

        # loss_en = 0.0

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

                    loss_e2 = chamfer_sqrt(d2_pred.transpose(1, 2), e2_curr.detach().transpose(1, 2))
                    loss_e1 = chamfer_sqrt(d1_pred.transpose(1, 2), e1_curr.detach().transpose(1, 2))
                    loss_e0 = chamfer_sqrt(d0_pred.transpose(1, 2), input.transpose(1, 2))  # input 是固定的

                    grad_l3 = torch.autograd.grad(loss_e2, l3_in, create_graph=True)[0]
                    grad_l2 = torch.autograd.grad(loss_e1, l2_in, create_graph=True)[0]
                    grad_l1 = torch.autograd.grad(loss_e0, l1_in, create_graph=True)[0]

                # loss_en += (loss_e2 + loss_e1 + loss_e0)

                e1_feed, l1_feed = self.encoder_sa1(input)
                l1_next = (self.beta * l1_feed
                           - self.alpha * grad_l1
                           + self.lambdax * d1_feedback
                           + (1 - self.beta - self.lambdax) * l1_in)
                e1_next = e1_feed

                e2_feed, l2_feed = self.encoder_sa2(e1_next, l1_next)
                l2_next = (self.beta * l2_feed
                           - self.alpha * grad_l2
                           + self.lambdax * d2_feedback
                           + (1 - self.beta - self.lambdax) * l2_in)
                e2_next = e2_feed

                e3_feed, l3_feed = self.encoder_sa3(e2_next, l2_next)
                l3_next = (self.beta * l3_feed
                           - self.alpha * grad_l3
                           + (1 - self.beta) * l3_in)
                e3_next = e3_feed

                e1_curr, l1_curr = e1_next, l1_next
                e2_curr, l2_curr = e2_next, l2_next
                e3_curr, l3_curr = e3_next, l3_next

        return e3_curr, l3_curr


class PostProcess(nn.Module):
    def __init__(self, embed_dim=128, upscale=[1, 4, 4]):
        super(PostProcess, self).__init__()
        # self.mlp_1 = nn.Conv1d(64, 128, 1) #
        # self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[32, 128])
        ##Parametric Surface Constrained Upsampler(self,upscale=4, dim_feat=512)
        up_layers = []
        for i, factor in enumerate(upscale):
            up_layers.append(UpLayer(dim=embed_dim, seed_dim=embed_dim, up_factor=factor, i=i, n_knn=20, radius=1,
                                     interpolate='three', attn_channel=True))
        self.up_layers = nn.ModuleList(up_layers)
        # print(self.up_layers)

    def forward(self, seed, fea):
        """
        Args:
            global_shape_fea: Tensor, (b, dim_feat, 1)
            pcd: Tensor, (b, 3, n)
        """
        pred_pcds = [seed.permute(0, 2, 1).contiguous()]
        # Upsample layers
        K_prev = None
        pcd = seed  # (B, 3, 256)
        for layer in self.up_layers:
            pcd, K_prev = layer(pcd, seed, fea, K_prev)
            pred_pcds.append(pcd.permute(0, 2, 1).contiguous())

        return pred_pcds


class IC_odgnet_mvp(nn.Module):
    def __init__(self, pretrain=''):
        super().__init__()
        self.encoder = IC_Encoder(pretrain=pretrain)
        self.decoder = PostProcess(embed_dim=128, upscale=[1, 2])
        self.mlp = MLP_Res(in_dim=512, hidden_dim=512, out_dim=1024)

    def forward(self, partial_point_cloud):
        batch_size, N, _ = partial_point_cloud.size()
        xyz, fea = self.encoder(partial_point_cloud)
        fea = self.mlp(fea).permute(0, 2, 1).contiguous()
        xyz = fps_subsample(torch.cat([partial_point_cloud, xyz.permute(0, 2, 1).contiguous()], dim=1), 1024).permute(0, 2,
                                                                                       1).contiguous()  # b, 3, 512

        all_pc = self.decoder(xyz, fea)
        return all_pc


class IC_odgnet_pcn(nn.Module):
    def __init__(self, pretrain=''):
        super().__init__()
        self.encoder = IC_Encoder(pretrain=pretrain)
        self.decoder = PostProcess(embed_dim=128, upscale=[1, 4, 4])
        self.mlp = MLP_Res(in_dim=512, hidden_dim=512, out_dim=1024)

    def forward(self, partial_point_cloud):
        batch_size, N, _ = partial_point_cloud.size()
        xyz, fea = self.encoder(partial_point_cloud)
        fea = self.mlp(fea).permute(0, 2, 1).contiguous()
        xyz = fps_subsample(torch.cat([partial_point_cloud, xyz.permute(0, 2, 1).contiguous()], dim=1), 1024).permute(0, 2,
                                                                                       1).contiguous()  # b, 3, 512
        all_pc = self.decoder(xyz, fea)
        return all_pc


if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_part = torch.randn(1, 2048, 3, requires_grad=True).to(device)


    model = IC_odgnet_mvp().to(device)
    all_pc = model(x_part)
    for i in all_pc:
        print(i.shape)
