import torch
import torch.nn as nn

from timm.layers import DropPath,trunc_normal_
# from utils.logger import *
import torch.nn.functional as F

from .EdgeConv import DGCNN_Grouper
from models.utilss import PointNet_SA_Module_KNN, Transformer, cross_transformer
from utils.loss_utils import chamfer_sqrt
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


# class Decoder_d2(nn.Module):
#     def __init__(self, channel=64):
#         super(Decoder_d2, self).__init__()
#         self.channel = channel
#         self.conv_out_2 = nn.Conv1d(channel * 2, 3, kernel_size=1)
#         self.conv_out1_2 = nn.Conv1d(channel * 4, channel * 2, kernel_size=1)
#         self.ps_2 = nn.ConvTranspose1d(channel * 8, channel, 128, bias=True)
#         self.ps_refuse_2 = nn.Conv1d(channel, channel * 8, kernel_size=1)
#         self.ps_adj_2 = nn.Conv1d(channel * 8, channel * 8, kernel_size=1)
#         self.sa0_d = cross_transformer(channel * 8, channel * 8)
#         self.relu = nn.GELU()
#
#     def forward(self, x, batch_size, N):
#         x_g_2 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1)
#         x = self.relu(self.ps_adj_2(x_g_2))
#         x = self.relu(self.ps_2(x))
#         x = self.relu(self.ps_refuse_2(x))
#         x0_d = self.sa0_d(x, x).reshape(batch_size, self.channel * 4, N // 8)
#         x = self.conv_out_2(self.relu(self.conv_out1_2(x0_d)))
#         return x, x0_d
#
#
# class Decoder_d1(nn.Module):
#     def __init__(self, channel=64):
#         super(Decoder_d1, self).__init__()
#         self.channel = channel
#         self.conv_out_1 = nn.Conv1d(channel, 3, kernel_size=1)
#         self.conv_out1_1 = nn.Conv1d(channel * 2, channel, kernel_size=1)
#         self.ps_1 = nn.ConvTranspose1d(channel * 4, channel, 256, bias=True)
#         self.ps_refuse_1 = nn.Conv1d(channel, channel * 4, kernel_size=1)
#         self.ps_adj_1 = nn.Conv1d(channel * 4, channel * 4, kernel_size=1)
#         self.sa0_d = cross_transformer(channel * 4, channel * 4)
#         self.relu = nn.GELU()
#
#     def forward(self, x, batch_size, N):
#         x_g_1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1)
#         x = self.relu(self.ps_adj_1(x_g_1))
#         x = self.relu(self.ps_1(x))
#         x = self.relu(self.ps_refuse_1(x))
#         x0_d = self.sa0_d(x, x).reshape(batch_size, self.channel * 2, N // 4)
#         x = self.conv_out_1(self.relu(self.conv_out1_1(x0_d)))
#         return x, x0_d
#
#
# class Decoder_d0(nn.Module):
#     def __init__(self, channel=64):
#         super(Decoder_d0, self).__init__()
#         self.channel = channel
#         self.conv_out_0 = nn.Conv1d(32, 3, kernel_size=1)
#         self.conv_out1_0 = nn.Conv1d(channel // 2, 32, kernel_size=1)
#         self.ps_0 = nn.ConvTranspose1d(channel * 2, channel, 512, bias=True)
#         self.ps_refuse_0 = nn.Conv1d(channel, channel * 2, kernel_size=1)
#         self.ps_adj_0 = nn.Conv1d(channel * 2, channel * 2, kernel_size=1)
#         self.sa0_d = cross_transformer(channel * 2, channel * 2)
#         self.relu = nn.GELU()
#
#     def forward(self, x, batch_size, N):
#         x_g_0 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1)
#         x = self.relu(self.ps_adj_0(x_g_0))
#         x = self.relu(self.ps_0(x))
#         x = self.relu(self.ps_refuse_0(x))
#         x0_d = self.sa0_d(x, x).reshape(batch_size, self.channel // 2, N)
#         x = self.conv_out_0(self.relu(self.conv_out1_0(x0_d)))
#         return x, x0_d


# class Decoder_d2(nn.Module):
#     def __init__(self, channel=64):
#         super(Decoder_d2, self).__init__()
#         self.channel = channel
#         self.conv_out_2 = nn.Conv1d(channel * 2, 3, kernel_size=1)
#         self.conv_out1_2 = nn.Conv1d(channel * 4, channel * 2, kernel_size=1)
#         self.ps_2 = nn.ConvTranspose1d(channel * 8, channel, 128, bias=True)
#         self.ps_refuse_2 = nn.Conv1d(channel, channel * 8, kernel_size=1)
#         self.ps_adj_2 = nn.Conv1d(channel * 8, channel * 8, kernel_size=1)
#         self.relu = nn.GELU()
#
#     def forward(self, x, batch_size, N):
#         x_g_2 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1)
#         x = self.relu(self.ps_adj_2(x_g_2))
#         x = self.relu(self.ps_2(x))
#         feat = self.relu(self.ps_refuse_2(x)).reshape(batch_size, self.channel * 4, N // 8)
#         x = self.conv_out_2(self.relu(self.conv_out1_2(feat)))
#         return x, feat
#
#
# class Decoder_d1(nn.Module):
#     def __init__(self, channel=64):
#         super(Decoder_d1, self).__init__()
#         self.channel = channel
#         self.conv_out_1 = nn.Conv1d(channel, 3, kernel_size=1)
#         self.conv_out1_1 = nn.Conv1d(channel * 2, channel, kernel_size=1)
#         self.ps_1 = nn.ConvTranspose1d(channel * 4, channel, 256, bias=True)
#         self.ps_refuse_1 = nn.Conv1d(channel, channel * 4, kernel_size=1)
#         self.ps_adj_1 = nn.Conv1d(channel * 4, channel * 4, kernel_size=1)
#         self.relu = nn.GELU()
#
#     def forward(self, x, batch_size, N):
#         x_g_1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1)
#         x = self.relu(self.ps_adj_1(x_g_1))
#         x = self.relu(self.ps_1(x))
#         feat = self.relu(self.ps_refuse_1(x)).reshape(batch_size, self.channel * 2, N // 4)
#         x = self.conv_out_1(self.relu(self.conv_out1_1(feat)))
#         return x, feat
#
#
# class Decoder_d0(nn.Module):
#     def __init__(self, channel=64):
#         super(Decoder_d0, self).__init__()
#         self.channel = channel
#         self.conv_out_0 = nn.Conv1d(32, 3, kernel_size=1)
#         self.conv_out1_0 = nn.Conv1d(channel // 2, 32, kernel_size=1)
#         self.ps_0 = nn.ConvTranspose1d(channel * 2, channel, 512, bias=True)
#         self.ps_refuse_0 = nn.Conv1d(channel, channel * 2, kernel_size=1)
#         self.ps_adj_0 = nn.Conv1d(channel * 2, channel * 2, kernel_size=1)
#         self.relu = nn.GELU()
#
#     def forward(self, x, batch_size, N):
#         x_g_0 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1)
#         x = self.relu(self.ps_adj_0(x_g_0))
#         x = self.relu(self.ps_0(x))
#         feat = self.relu(self.ps_refuse_0(x)).reshape(batch_size, self.channel // 2, N)
#         x = self.conv_out_0(self.relu(self.conv_out1_0(feat)))
#         return x, feat
#
# class Decoder_d1_Folding(nn.Module):
#     def __init__(self, channel=64):
#         super(Decoder_d1_Folding, self).__init__()
#         self.channel = channel
#         self.conv_out_1 = nn.Conv1d(channel, 3, kernel_size=1)
#         self.conv_out1_1 = nn.Conv1d(channel * 2, channel, kernel_size=1)
#         self.ps_adj_1 = nn.Conv1d(channel * 4, channel * 4, kernel_size=1)
#         # 【修改点：Folding 操作】
#         # 1. 生成 32 个粗糙点 (32 * 64 = 2048 维)
#         self.coarse_gen_1 = nn.Conv1d(channel * 4, 2048, kernel_size=1)
#         # 2. 结合 1D grid (8) 分裂为 256 个点，输入为特征(64) + grid(1) = 65
#         self.fine_gen_1 = nn.Conv1d(65, channel, kernel_size=1)
#         self.ps_refuse_1 = nn.Conv1d(channel, channel * 4, kernel_size=1)
#         self.relu = nn.GELU()
#
#     def forward(self, x, batch_size, N):
#         x_g_1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1)  # (B, 256, 1)
#         x = self.relu(self.ps_adj_1(x_g_1))
#         # --- Folding 生成过程 ---
#         # 1. 生成粗糙特征 (B, 2048, 1) -> view -> (B, 64, 32)
#         coarse = self.relu(self.coarse_gen_1(x)).view(batch_size, self.channel, 32)
#         # 2. 将每个点复制 8 次，准备分裂 (B, 64, 256)
#         coarse_repeated = coarse.repeat_interleave(8, dim=-1)
#         # 3. 创建 1D 网格，附加到特征上 (B, 1, 256)
#         grid = torch.linspace(-1, 1, 8, device=x.device).view(1, 1, 8).repeat(batch_size, 1, 32)
#         fine_input = torch.cat([coarse_repeated, grid], dim=1)  # (B, 65, 256)
#         # 4. 生成最终的 256 个特征点 (B, 64, 256)
#         x = self.relu(self.fine_gen_1(fine_input))
#         # 这里的输出 x 为 (B, 64, 256)
#         # ps_refuse_1 后变为 (B, 256, 256)，reshape 后为 (B, 128, N // 4) 即 (B, 128, 512)，与原版完全一致
#         feat = self.relu(self.ps_refuse_1(x)).reshape(batch_size, self.channel * 2, N // 4)
#         x = self.conv_out_1(self.relu(self.conv_out1_1(feat)))
#         return x, feat
#
#
# class Decoder_d0_Folding(nn.Module):
#     def __init__(self, channel=64):
#         super(Decoder_d0_Folding, self).__init__()
#         self.channel = channel
#         self.conv_out_0 = nn.Conv1d(32, 3, kernel_size=1)
#         self.conv_out1_0 = nn.Conv1d(channel // 2, 32, kernel_size=1)
#         self.ps_adj_0 = nn.Conv1d(channel * 2, channel * 2, kernel_size=1)
#         # 【修改点：Folding 操作】
#         # 1. 生成 64 个粗糙点 (64 * 64 = 4096 维)
#         self.coarse_gen_0 = nn.Conv1d(channel * 2, 4096, kernel_size=1)
#         # 2. 结合 1D grid (8) 分裂为 512 个点，输入为特征(64) + grid(1) = 65
#         self.fine_gen_0 = nn.Conv1d(65, channel, kernel_size=1)
#         self.ps_refuse_0 = nn.Conv1d(channel, channel * 2, kernel_size=1)
#         self.relu = nn.GELU()
#
#     def forward(self, x, batch_size, N):
#         x_g_0 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1)  # (B, 128, 1)
#         x = self.relu(self.ps_adj_0(x_g_0))
#         # --- Folding 生成过程 ---
#         # 1. 生成粗糙特征 (B, 4096, 1) -> view -> (B, 64, 64)
#         coarse = self.relu(self.coarse_gen_0(x)).view(batch_size, self.channel, 64)
#         # 2. 将每个点复制 8 次，准备分裂 (B, 64, 512)
#         coarse_repeated = coarse.repeat_interleave(8, dim=-1)
#         # 3. 创建 1D 网格，附加到特征上 (B, 1, 512)
#         grid = torch.linspace(-1, 1, 8, device=x.device).view(1, 1, 8).repeat(batch_size, 1, 64)
#         fine_input = torch.cat([coarse_repeated, grid], dim=1)  # (B, 65, 512)
#         # 4. 生成最终的 512 个特征点 (B, 64, 512)
#         x = self.relu(self.fine_gen_0(fine_input))
#         # 这里的输出 x 为 (B, 64, 512)
#         # ps_refuse_0 后变为 (B, 128, 512)，reshape 后为 (B, 32, N) 即 (B, 32, 2048)，与原版完全一致
#         feat = self.relu(self.ps_refuse_0(x)).reshape(batch_size, self.channel // 2, N)
#         x = self.conv_out_0(self.relu(self.conv_out1_0(feat)))
#         return x, feat
#
# class Decoder_d2_Folding(nn.Module):
#     def __init__(self, channel=64):
#         super(Decoder_d2_Folding, self).__init__()
#         self.channel = channel
#         self.conv_out_2 = nn.Conv1d(channel * 2, 3, kernel_size=1)
#         self.conv_out1_2 = nn.Conv1d(channel * 4, channel * 2, kernel_size=1)
#
#         self.ps_adj_2 = nn.Conv1d(channel * 8, channel * 8, kernel_size=1)
#
#         # 【核心修改】：用 MLP 替代 ConvTranspose1d
#         # 1. 生成 16 个粗糙点 (16 * 64 = 1024)
#         self.coarse_gen = nn.Conv1d(channel * 8, 1024, kernel_size=1)
#         # 2. 结合 1D grid (长度为8) 生成 128 个点，输入通道为 64(特征) + 1(网格) = 65
#         self.fine_gen = nn.Conv1d(65, channel, kernel_size=1)
#
#         self.ps_refuse_2 = nn.Conv1d(channel, channel * 8, kernel_size=1)
#         self.relu = nn.GELU()
#
#     def forward(self, x, batch_size, N):
#         x_g_2 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(-1)  # (B, 512, 1)
#         x = self.relu(self.ps_adj_2(x_g_2))
#
#         # --- Folding 生成过程 ---
#         # 1. 生成粗糙特征 (B, 1024, 1) -> view -> (B, 64, 16)
#         coarse = self.relu(self.coarse_gen(x)).view(batch_size, 64, 16)
#
#         # 2. 将每个点复制 8 次，准备分裂 (B, 64, 128)
#         coarse_repeated = coarse.repeat_interleave(8, dim=-1)
#
#         # 3. 创建 1D 网格，附加到特征上 (打破特征完全相同的局面)
#         grid = torch.linspace(-1, 1, 8, device=x.device).view(1, 1, 8).repeat(batch_size, 1, 16)
#         fine_input = torch.cat([coarse_repeated, grid], dim=1)  # (B, 65, 128)
#
#         # 4. 生成最终的 128 个特征点 (B, 64, 128)
#         x = self.relu(self.fine_gen(fine_input))
#         feat = self.relu(self.ps_refuse_2(x)).reshape(batch_size, self.channel * 4, N // 8)
#         x = self.conv_out_2(self.relu(self.conv_out1_2(feat)))
#         return x, feat
#
#
# class IC_Encoder(nn.Module):
#     def __init__(self, timesteps=3, alpha=0.01, beta=0.8, lambdax=0.1, channel=64, pretrain=''):
#         super().__init__()
#
#         self.channel = channel
#         # 时间步和更新参数
#         self.timesteps = timesteps  # 迭代时间步数
#         self.alpha = alpha  # 梯度校正权重
#         self.beta = beta  # 前馈权重
#         self.lambdax = lambdax  # 反馈权重
#
#         # 编码器和解码器
#         self.encoder_sa1 = Encoder_sa1()
#         self.encoder_sa2 = Encoder_sa2()
#         self.encoder_sa3 = Encoder_sa3()
#
#         # self.decoder_d2 = Decoder_d2(channel=self.channel)
#         # self.decoder_d1 = Decoder_d1(channel=self.channel)
#         # self.decoder_d0 = Decoder_d0(channel=self.channel)
#         self.decoder_d2 = Decoder_d2_Folding(channel=self.channel)
#         self.decoder_d1 = Decoder_d1_Folding(channel=self.channel)
#         self.decoder_d0 = Decoder_d0_Folding(channel=self.channel)
#
#         self.pretrain = pretrain
#         self._load_pretrained_weights()
#         self.loss_func = ChamferDistanceL1()
#
#     def _load_pretrained_weights(self):
#         pretrain = self.pretrain
#         if pretrain:
#             print(f'Use pretrained model {pretrain}')
#             # 1. 加载预训练权重
#             state_dict = torch.load(pretrain, map_location='cpu')
#             weights_dict = {}
#
#             # 2. 获取模型所有参数名 + buffer名（关键：包含BN层的running_mean等）
#             # 先获取可训练参数名
#             model_param_names = set([name for name, _ in self.named_parameters()])
#             # 再获取buffer名（BN层的running_mean/running_var/num_batches_tracked都在这里）
#             model_buffer_names = set([name for name, _ in self.named_buffers()])
#             # 合并：模型所有可加载的参数/缓冲区名
#             model_all_names = model_param_names.union(model_buffer_names)
#
#             # 3. 遍历预训练权重，处理前缀并匹配模型参数
#             skipped_names = []  # 存储真正不存在的参数（用于排查）
#             for k, v in state_dict.items():
#                 new_k = k
#                 # 第一步：移除 module. 前缀（多卡训练遗留）
#                 if 'module.' in new_k:
#                     new_k = new_k.replace('module.', '')
#                 # 第二步：移除 encoder. 前缀（核心修复点）
#                 if new_k.startswith('encoder.'):
#                     new_k = new_k[len('encoder.'):]
#                 # 第三步：替换 sa_mlp_conv 为 sa.mlp_conv（保留你的原有逻辑）
#                 if 'sa_mlp_conv' in new_k:
#                     new_k = new_k.replace('sa_mlp_conv', 'sa.mlp_conv')
#
#                 # # 关键：只跳过「模型真正不存在」的参数，不再误跳BN的buffer
#                 if new_k in model_all_names:
#                     weights_dict[new_k] = v
#                 else:
#                     skipped_names.append(new_k)
#
#             load_result = self.load_state_dict(weights_dict)
#             print(f"✅ 成功加载 {len(weights_dict)} 个匹配的参数")
#
#             # 打印真正跳过的参数（模型确实没有的）
#             if skipped_names:
#                 print(f"⚠️ 真正跳过的、模型不存在的参数：{len(skipped_names)} 个")
#                 # 只打印前10个，避免输出过长
#                 print("跳过参数示例：", skipped_names[:10])
#
#             # 打印加载结果（重点关注missing_keys）
#             if load_result.missing_keys:
#                 print(f"❌ 仍缺失的参数（模型有但权重里没有）：{len(load_result.missing_keys)} 个")
#                 print("缺失参数示例：", load_result.missing_keys[:10])
#             else:
#                 print("✅ 无缺失参数！所有模型参数都已匹配加载")
#
#             if load_result.unexpected_keys:
#                 print(f"❌ 意外的参数（权重里有但模型没有）：{len(load_result.unexpected_keys)} 个")
#                 print("意外参数示例：", load_result.unexpected_keys[:10])
#         else:
#             print('Not Use pretrained model.')
#
#     def forward(self, pc):
#         pc = pc.permute(0, 2, 1).contiguous()
#         batch_size, _, N = pc.size()
#         input = pc  # (B, 3, N)
#
#         e1_feed_static, l1_feed_static = self.encoder_sa1(input)
#         e1_curr, l1_curr = e1_feed_static, l1_feed_static
#         e2_curr, l2_curr = self.encoder_sa2(e1_curr, l1_curr)
#         e3_curr, l3_curr = self.encoder_sa3(e2_curr, l2_curr)
#         if self.timesteps > 0:
#             for t in range(self.timesteps):
#                 with torch.enable_grad():
#                     l1_in = l1_curr.detach().requires_grad_(True)
#                     l2_in = l2_curr.detach().requires_grad_(True)
#                     l3_in = l3_curr.detach().requires_grad_(True)
#
#                     d2_pred, d2_pred_feat = self.decoder_d2(l3_in, batch_size, N)
#                     d1_pred, d1_pred_feat = self.decoder_d1(l2_in, batch_size, N)
#                     d0_pred, d0_pred_feat = self.decoder_d0(l1_in, batch_size, N)
#
#                     loss_e2 = self.loss_func(d2_pred.transpose(1, 2), e2_curr.detach().transpose(1, 2))
#                     loss_e1 = self.loss_func(d1_pred.transpose(1, 2), e1_curr.detach().transpose(1, 2))
#                     loss_e0 = self.loss_func(d0_pred.transpose(1, 2), input.transpose(1, 2))  # input 是固定的
#
#                     grad_l3 = torch.autograd.grad(loss_e2, l3_in, create_graph=True)[0]
#                     grad_l2 = torch.autograd.grad(loss_e1, l2_in, create_graph=True)[0]
#                     grad_l1 = torch.autograd.grad(loss_e0, l1_in, create_graph=True)[0]
#
#                     # grad_l3 = torch.clamp(grad_l3, -1.0, 1.0)
#                     # grad_l2 = torch.clamp(grad_l2, -1.0, 1.0)
#                     # grad_l1 = torch.clamp(grad_l1, -1.0, 1.0)
#
#                 l1_next = (self.beta * l1_feed_static
#                            - self.alpha * grad_l1
#                            + self.lambdax * (d1_pred_feat - l1_curr)
#                            + (1 - self.beta) * l1_curr)
#                 e1_next = e1_feed_static
#
#                 e2_feed, l2_feed = self.encoder_sa2(e1_next, l1_next)
#                 l2_next = (self.beta * l2_feed
#                            - self.alpha * grad_l2
#                            + self.lambdax * (d2_pred_feat - l2_curr)
#                            + (1 - self.beta) * l2_curr)
#                 e2_next = e2_feed
#
#                 e3_feed, l3_feed = self.encoder_sa3(e2_next, l2_next)
#                 l3_next = (self.beta * l3_feed
#                            - self.alpha * grad_l3
#                            + (1 - self.beta) * l3_curr)
#                 e3_next = e3_feed
#
#                 e1_curr, l1_curr = e1_next, l1_next
#                 e2_curr, l2_curr = e2_next, l2_next
#                 e3_curr, l3_curr = e3_next, l3_next
#
#         # 保持返回值格式一致，不影响后续代码
#         return e3_curr, l3_curr


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

        e1_feed_static, l1_feed_static = self.encoder_sa1(input)  # torch.Size([1, 3, 512]) torch.Size([1, 128, 512])
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

                    d2_pred = self.decoder_d2(l3_in, batch_size, N)
                    d1_pred = self.decoder_d1(l2_in, batch_size, N)
                    d0_pred = self.decoder_d0(l1_in, batch_size, N)

                    d1_feedback = self.feedback_map_d1(d1_pred)
                    d2_feedback = self.feedback_map_d2(d2_pred)

                    loss_e2 = self.loss_func(d2_pred.transpose(1, 2), e2_curr.detach().transpose(1, 2))
                    loss_e1 = self.loss_func(d1_pred.transpose(1, 2), e1_curr.detach().transpose(1, 2))
                    loss_e0 = self.loss_func(d0_pred.transpose(1, 2), input.transpose(1, 2))

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

        return e3_curr, l3_curr


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GeoCrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=1, qkv_bias=False, qk_scale=1, attn_drop=0., proj_drop=0., aggregate_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Identity() # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Identity() # nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.x_map = nn.Identity() # nn.Linear(aggregate_dim, 1)
        
    def forward(self, q, k, v):
        B, N, _ = q.shape
        C = self.out_dim
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, NK, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, 3)

        x = self.x_map(x)

        return x

class SubFold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()
        self.in_channel = in_channel
        self.step = step
        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x, c):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = c.to(x.device) # b 3 n2
        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        return fd2

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_pred=16, num_point=128):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.norm_q = nn.Identity() # norm_layer(dim)
        self.norm_k = nn.Identity () # norm_layer(dim)
        self.attn = GeoCrossAttention(dim, dim, num_heads=1, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, aggregate_dim=16)
        
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.fold_step = int(pow(num_pred, 0.5) + 0.5)
        self.generate_anchor = SubFold(dim, step = self.fold_step, hidden_dim = dim // 2)
        
        self.num_pred = num_pred
        self.num_point = num_point
        self.generate_feature = nn.Sequential(
            nn.Conv1d(self.num_point, 256, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 16, 1)
        )
        
    def forward(self, x, coor):
        norm_x = self.norm1(x)
        x_1 = self.self_attn(norm_x)
        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x))) # B N dim
        
        global_x = torch.max(x, dim=1, keepdim=False)[0] # B dim
        diff_x = global_x.unsqueeze(1).repeat(1,self.num_point,1) - x
        x_2 = self.generate_feature(diff_x)
        
        norm_k = self.norm_k(x) # B N dim
        norm_q = self.norm_q(x_2) # B L dim
        coor_2 = self.attn(q=norm_q, k=norm_k, v=coor) 
        coor_2 = self.generate_anchor(global_x, coor_2.transpose(1,2)).transpose(1,2)
        
        x = torch.cat([x, x_2], dim=1)        
        coor = torch.cat([coor,coor_2], dim=1) # coor: B N 3 -> B N+L 3
        return x, coor

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, q, v):
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)

        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)

        q = q + self.drop_path(q_2)
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q

# --- Transfomer Architecture --- #
class AnchorTransformer(nn.Module):
    """ Edge Convolution +  Transformer Encoder +  Point Scattering + Transformer Decoder
    """
    def __init__(self, in_chans=3, embed_dim=768, depth=[6, 6], num_query = 224, num_heads=6, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_query = num_query
        self.depth = depth

         # feature extraction
        self.grouper = DGCNN_Grouper()  # B 3 N to B 3 N and B C N
        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, embed_dim, 1)
        )
        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        # encoder
        self.encoder = nn.ModuleList([
            EncoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, num_pred=16, num_point=128+i*16)
            for i in range(depth[0])])

        # increase dim and make coarse query
        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * (num_query-128))
        )
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1)
        )

        # decoder 
        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[1])])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, inpc):
        '''
            inpc : input incomplete point cloud with shape B N(2048) C(3)
        '''
        # build point feature
        bs = inpc.size(0)
        coor, f = self.grouper(inpc.transpose(1,2).contiguous())
        # print(coor.shape, f.shape)  # torch.Size([2, 3, 128]) torch.Size([2, 128, 128])
        pos =  self.pos_embed(coor).transpose(1,2)
        x = self.input_proj(f).transpose(1,2)
        coor = coor.transpose(1,2)
        
        # encoder
        x = x + pos
        for i, blk in enumerate(self.encoder): 
            # x, coor = blk(x+pos, coor)
            x, coor = blk(x, coor)

        # point scattering
        global_feature = self.increase_dim(x.transpose(1,2)) # B 1024 N 
        global_feature = torch.max(global_feature, dim=-1)[0] # B 1024
        coarse_relative = torch.cat([torch.zeros([bs, 128, 3]).to(x.device), self.coarse_pred(global_feature).reshape(bs, -1, 3)], dim=1)  # B M C(3)
        coarse_point = coarse_relative + coor
        
        query_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, coarse_point.size(1), -1), 
            coarse_point], dim=-1) # B M C+3 
        q = self.mlp_query(query_feature.transpose(1,2)).transpose(1,2) # B M C 
        
        # decoder
        for i, blk in enumerate(self.decoder): 
            q = blk(q, x)
            
        return q, coarse_point


# --- Transfomer Architecture --- #
class IC_AnchorTransformer(nn.Module):
    """ Edge Convolution +  Transformer Encoder +  Point Scattering + Transformer Decoder
    """

    def __init__(self, in_chans=3, embed_dim=768, depth=[6, 6], num_query=224, num_heads=6, mlp_ratio=2.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., pretrain=''):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_query = num_query
        self.depth = depth

        # feature extraction
        # self.grouper = DGCNN_Grouper()  # B 3 N to B 3 N and B C N
        self.grouper = IC_Encoder(pretrain=pretrain)
        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, embed_dim, 1)
        )
        self.input_proj = nn.Sequential(
            nn.Conv1d(512, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        # encoder
        self.encoder = nn.ModuleList([
            EncoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, num_pred=16, num_point=128 + i * 16)
            for i in range(depth[0])])

        # increase dim and make coarse query
        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * (num_query - 128))
        )
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1)
        )

        # decoder
        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[1])])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, inpc):
        '''
            inpc : input incomplete point cloud with shape B N(2048) C(3)
        '''
        # build point feature
        bs = inpc.size(0)
        coor, f = self.grouper(inpc)
        # print(coor.shape, f.shape)  # torch.Size([2, 3, 128]) torch.Size([2, 128, 128])
        pos = self.pos_embed(coor).transpose(1, 2)
        x = self.input_proj(f).transpose(1, 2)
        coor = coor.transpose(1, 2)

        # encoder
        x = x + pos
        for i, blk in enumerate(self.encoder):
            # x, coor = blk(x+pos, coor)
            x, coor = blk(x, coor)

        # point scattering
        global_feature = self.increase_dim(x.transpose(1, 2))  # B 1024 N
        global_feature = torch.max(global_feature, dim=-1)[0]  # B 1024
        coarse_relative = torch.cat(
            [torch.zeros([bs, 128, 3]).to(x.device), self.coarse_pred(global_feature).reshape(bs, -1, 3)],
            dim=1)  # B M C(3)
        coarse_point = coarse_relative + coor

        query_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, coarse_point.size(1), -1),
            coarse_point], dim=-1)  # B M C+3
        q = self.mlp_query(query_feature.transpose(1, 2)).transpose(1, 2)  # B M C

        # decoder
        for i, blk in enumerate(self.decoder):
            q = blk(q, x)

        return q, coarse_point
