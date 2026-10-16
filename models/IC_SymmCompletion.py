import torch
import torch.nn as nn
from models.utilss import MLP_CONV, Transformer, PointNet_SA_Module_KNN, fps_subsample
import torch.nn.functional as F

from utils.loss_utils import chamfer_sqrt



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        _, NK, _ = y.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B, H, N, C
        k = self.k(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B, H, NK, C
        v = self.v(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B, H, NK, C

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, H, N, NK
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossFormer(nn.Module):

    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
                 drop_path=0.1):
        super().__init__()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.bn3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x, y):
        short_cut = x
        x = self.bn1(x)
        y = self.bn2(y)
        x = self.attn(query=x, key=y, value=y)[0]
        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.bn3(x)))
        return x


class Fusion(nn.Module):
    def __init__(self, in_channel=512):
        super(Fusion, self).__init__()

        self.corssformer_1 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None,
                                         attn_drop=0.0, proj_drop=0.0)
        self.corssformer_2 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None,
                                         attn_drop=0.0, proj_drop=0.0)

    def forward(self, feat_x, feat_y):
        # cross attention
        feat = self.corssformer_1(feat_x, feat_y)

        # self attention
        feat = self.corssformer_2(feat, feat)
        return feat


class SGFormer(nn.Module):
    def __init__(self, gf_dim=512, up_factor=2):
        super(SGFormer, self).__init__()
        self.up_factor = up_factor
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_gf = MLP_CONV(in_channel=gf_dim, layer_dims=[256, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        self.transformer = Transformer(in_channel=128, dim=64)

        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, 256])

        self.fusion_1 = Fusion(in_channel=256)
        self.fusion_2 = Fusion(in_channel=256)

        self.mlp_fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512)
        )
        self.fusion_3 = Fusion(in_channel=512)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3 * self.up_factor)
        )

    def forward(self, coarse, symmetry_feat, partial_feat):
        b, _, n = coarse.shape
        feat = self.mlp_1(coarse)
        feat_max = feat.max(dim=-1, keepdim=True)[0]
        feat = torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, coarse)

        feat = self.expand_dim_1(feat)
        partial_feat = self.expand_dim_2(partial_feat)
        symmetry_feat = self.expand_dim_3(symmetry_feat)

        feat = feat.transpose(2, 1).contiguous()
        partial_feat = partial_feat.transpose(2, 1).contiguous()
        symmetry_feat = symmetry_feat.transpose(2, 1).contiguous()

        # partial part awareness
        feat_p = self.fusion_1(feat, partial_feat)
        # symmetric part awareness
        feat_s = self.fusion_2(feat, symmetry_feat)
        # fusion feature
        feat = torch.cat([feat_p, feat_s], dim=-1)
        feat = self.mlp_fusion(feat)

        # self attention for upsampling
        feat = self.fusion_3(feat, feat)
        offset = self.fc(feat).view(b, -1, 3)  # B, N * up_ratio, 3
        pcd_up = coarse.transpose(2, 1).contiguous().unsqueeze(dim=2).repeat(1, 1, self.up_factor, 1).view(b, -1,
                                                                                                           3) + offset
        return pcd_up


class local_encoder(nn.Module):
    def __init__(self, out_channel=128):
        super(local_encoder, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2, layer_dims=[256, out_channel])
        self.transformer = Transformer(out_channel, dim=64)

    def forward(self, input):
        feat = self.mlp_1(input)
        feat = torch.cat([feat, torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, input)

        return feat


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

        # 编码器和解码器
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
        # pc = pc.permute(0, 2, 1).contiguous()
        batch_size, _, N = pc.size()
        input = pc  # (B, 3, N)
        e1_curr, l1_curr = self.encoder_sa1(input)
        e2_curr, l2_curr = self.encoder_sa2(e1_curr, l1_curr)
        e3_curr, l3_curr = self.encoder_sa3(e2_curr, l2_curr)

        loss_en = 0.0

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
                    loss_e0 = chamfer_sqrt(d0_pred.transpose(1, 2), input.transpose(1, 2))

                    grad_l3 = torch.autograd.grad(loss_e2, l3_in, create_graph=True)[0]
                    grad_l2 = torch.autograd.grad(loss_e1, l2_in, create_graph=True)[0]
                    grad_l1 = torch.autograd.grad(loss_e0, l1_in, create_graph=True)[0]

                loss_en += (loss_e2 + loss_e1 + loss_e0)

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

        return e2_curr, e3_curr, l3_curr


class SymmCompletion(nn.Module):
    # def __init__(self, config, **kwargs):
    def __init__(self, pretrain='', **kwargs):
        super().__init__()
        up_factors = "1, 2"
        # up_factors = "2, 8"
        self.up_factors = [int(i) for i in up_factors.split(',')]
        self.pc_encoder = IC_Encoder(pretrain=pretrain)
        self.local_encoder = local_encoder(out_channel=128)
        self.sgformer_1 = SGFormer(gf_dim=512, up_factor=self.up_factors[0])
        self.sgformer_2 = SGFormer(gf_dim=512, up_factor=self.up_factors[1])
        self.include_input = False

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        coarse, symmetry_points, keyfeatures = self.pc_encoder(point_cloud.transpose(2, 1).contiguous())
        # torch.Size([1, 3, 1024]) torch.Size([1, 3, 512]) torch.Size([1, 128, 512])
        # print(coarse.shape, symmetry_points.shape, keyfeatures.shape)
        # torch.Size([1, 3, 256]) torch.Size([1, 3, 128]) torch.Size([1, 512, 128])

        coarse = fps_subsample(torch.cat([coarse.permute(0, 2, 1).contiguous(), point_cloud], dim=1), 1024).permute(0, 2,
                                                                                                        1).contiguous()  # b, 3, 512

        feat_symmetry = self.local_encoder(symmetry_points)  # B,128,512
        feat_partial = keyfeatures.transpose(2, 1).contiguous()  # B,128,512
        fine1 = self.sgformer_1(coarse, feat_symmetry, feat_partial)
        fine2 = self.sgformer_2(fine1.transpose(2, 1).contiguous(), feat_symmetry, feat_partial)

        if self.include_input:
            fine2 = torch.cat([fine2, point_cloud], dim=1).contiguous()

        rets = [coarse.transpose(2, 1).contiguous(), fine1, fine2]
        return rets


if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_part = torch.randn(1, 2048, 3, requires_grad=True).to(device)

    model = SymmCompletion(pretrain='').to(device)

    all_pc = model(x_part)
    for a in all_pc:
        print(a.size())
