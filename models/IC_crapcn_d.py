import torch.utils.data

from models.crapcn import *
from models.utilss import Transformer, PointNet_SA_Module_KNN, cross_transformer
from extensions.chamfer_dist import ChamferDistanceL1

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
        feat = self.relu(self.ps_refuse_2(x)).reshape(batch_size, self.channel * 4, N // 8)
        x = self.conv_out_2(self.relu(self.conv_out1_2(feat)))
        return x, feat


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
        feat = self.relu(self.ps_refuse_1(x)).reshape(batch_size, self.channel * 2, N // 4)
        x = self.conv_out_1(self.relu(self.conv_out1_1(feat)))
        return x, feat


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
        feat = self.relu(self.ps_refuse_0(x)).reshape(batch_size, self.channel // 2, N)
        x = self.conv_out_0(self.relu(self.conv_out1_0(feat)))
        return x, feat

# BRICE main content V2
class IC_Encoder(nn.Module):
    def __init__(self, timesteps=3, alpha=0.01, beta=0.8, lambdax=0.01, channel=64, pretrain=''):
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

        # other methods
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
        # loss_en = 0.0

        if self.timesteps > 0:
            for t in range(self.timesteps):
                with torch.enable_grad():
                    l1_in = l1_curr.detach().requires_grad_(True)
                    l2_in = l2_curr.detach().requires_grad_(True)
                    l3_in = l3_curr.detach().requires_grad_(True)

                    d2_pred, d2_pred_feat = self.decoder_d2(l3_in, batch_size, N)
                    d1_pred, d1_pred_feat = self.decoder_d1(l2_in, batch_size, N)
                    d0_pred, d0_pred_feat = self.decoder_d0(l1_in, batch_size, N)

                    loss_e2 = self.loss_func(d2_pred.transpose(1, 2), e2_curr.detach().transpose(1, 2))
                    loss_e1 = self.loss_func(d1_pred.transpose(1, 2), e1_curr.detach().transpose(1, 2))
                    loss_e0 = self.loss_func(d0_pred.transpose(1, 2), input.transpose(1, 2))  # input 是固定的

                    grad_l3 = torch.autograd.grad(loss_e2, l3_in, create_graph=True)[0]
                    grad_l2 = torch.autograd.grad(loss_e1, l2_in, create_graph=True)[0]
                    grad_l1 = torch.autograd.grad(loss_e0, l1_in, create_graph=True)[0]

                    # grad_l3 = torch.clamp(grad_l3, -1.0, 1.0)
                    # grad_l2 = torch.clamp(grad_l2, -1.0, 1.0)
                    # grad_l1 = torch.clamp(grad_l1, -1.0, 1.0)

                # loss_en += (loss_e2 + loss_e1 + loss_e0)

                l1_next = (self.beta * l1_feed_static
                           - self.alpha * grad_l1
                           + self.lambdax * (d1_pred_feat - l1_curr)
                           + (1 - self.beta) * l1_curr)
                e1_next = e1_feed_static

                e2_feed, l2_feed = self.encoder_sa2(e1_next, l1_next)
                l2_next = (self.beta * l2_feed
                           - self.alpha * grad_l2
                           + self.lambdax * (d2_pred_feat - l2_curr)
                           + (1 - self.beta) * l2_curr)
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

        return e2_curr, l3_curr, global_f


class IC_crapcn_mvp(nn.Module):
    def __init__(self, pretrain=''):
        super().__init__()
        self.encoder = IC_Encoder(pretrain=pretrain)
        self.decoder = Decoder_mvp()
        self.mlp = MLP_Res(in_dim=512, hidden_dim=256, out_dim=256)

    def forward(self, xyz):
        # print(xyz.shape)
        p_sd, f_sd, global_f = self.encoder(xyz)  # torch.Size([1, 3, 256]) torch.Size([1, 512, 128]) torch.Size([1, 512, 1])
        # print(p_sd.shape, f_sd.shape, global_f.shape)

        f_sd = self.mlp(f_sd).permute(0, 2, 1).contiguous()

        p0 = fps_subsample(torch.cat([p_sd.permute(0, 2, 1).contiguous(), xyz], dim=1), 512).permute(0, 2,
                                                                                                        1).contiguous()  # b, 3, 512
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)

        return all_pc

# class IC_crapcn_mvp(nn.Module):
#     def __init__(self, pretrain=''):
#         super().__init__()
#         self.encoder = IC_Encoder(pretrain=pretrain)
#         self.seed_generator = SeedGenerator()
#         self.decoder = Decoder_mvp()
#
#     def forward(self, xyz):
#         pp, fp, global_f, loss_en = self.encoder(xyz)
#         p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
#         all_pc = self.decoder(global_f, p0, p_sd, f_sd)
#
#
#         if self.training:
#             return all_pc, loss_en
#         else:
#             return all_pc


class IC_crapcn_pcn(nn.Module):
    def __init__(self, pretrain=''):
        super().__init__()
        self.encoder = IC_Encoder(pretrain=pretrain)
        self.seed_generator = SeedGenerator()
        self.decoder = Decoder()


    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz)
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        return all_pc


if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_part = torch.randn(1, 2048, 3, requires_grad=True).to(device)
    model = IC_crapcn_mvp(pretrain='/media/ps/Data/WYL/CRA-PCN-main/pretrain/IC_post_d_pre/ckpt-best.pth').to(device)

    all_pc = model(x_part)
    for i in all_pc:
        print(i.shape)

    from thop import profile
    flops, params = profile(model, (x_part,))
    print('flops: ', flops, 'params: ', params)
    print('Gflops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))