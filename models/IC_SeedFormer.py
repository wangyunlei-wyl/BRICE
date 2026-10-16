import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from models.utilss import Transformer, PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, query_knn, \
    grouping_operation, get_nearest_index, indexing_neighbor, cross_transformer
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

        # 反馈映射层（将3维坐标映射到特征维度）
        self.feedback_map_d1 = nn.Sequential(  # 对应d1_t → 映射到128维（与l1匹配）
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1)
        )
        self.feedback_map_d2 = nn.Sequential(  # 对应d2_t → 映射到256维（与l2匹配）
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
                e2_next = e2_feed  # 更新 e2 供下一层使用

                e3_feed, l3_feed = self.encoder_sa3(e2_next, l2_next)
                l3_next = (self.beta * l3_feed
                           - self.alpha * grad_l3
                           + (1 - self.beta) * l3_curr)
                e3_next = e3_feed

                e1_curr, l1_curr = e1_next, l1_next
                e2_curr, l2_curr = e2_next, l2_next
                e3_curr, l3_curr = e3_next, l3_next

        global_f = F.adaptive_max_pool1d(l3_curr, 1).view(batch_size, -1).unsqueeze(-1)

        return global_f, e2_curr, l2_curr


class SeedGenerator(nn.Module):
    def __init__(self, feat_dim=512, seed_dim=128, n_knn=20, factor=2, attn_channel=True):
        super(SeedGenerator, self).__init__()
        self.uptrans = UpTransformer(256, 128, dim=64, n_knn=n_knn, use_upfeat=False, attn_channel=attn_channel,
                                     up_factor=factor, scale_layer=None)
        self.mlp_1 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=seed_dim)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat, patch_xyz, patch_feat):
        """
        Args:
            feat: Tensor (B, feat_dim, 1)
            patch_xyz: (B, 3, 128)
            patch_feat: (B, seed_dim, 128)
        """
        x1 = self.uptrans(patch_xyz, patch_feat, patch_feat, upfeat=None)  # (B, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (B, 128, 256)
        completion = self.mlp_4(x3)  # (B, 3, 256)
        return completion, x3


class UpTransformer(nn.Module):
    def __init__(self, in_channel, out_channel, dim, n_knn=20, up_factor=2, use_upfeat=True,
                 pos_hidden_dim=64, attn_hidden_multiplier=4, scale_layer=nn.Softmax, attn_channel=True):
        super(UpTransformer, self).__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        self.use_upfeat = use_upfeat
        attn_out_channel = dim if attn_channel else 1

        self.mlp_v = MLP_Res(in_dim=in_channel * 2, hidden_dim=in_channel, out_dim=in_channel)
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        if use_upfeat:
            self.conv_upfeat = nn.Conv1d(in_channel, dim, 1)

        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        # attention layers
        self.attn_mlp = [nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
                         nn.BatchNorm2d(dim * attn_hidden_multiplier),
                         nn.ReLU()]
        if up_factor:
            self.attn_mlp.append(
                nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor, 1), (up_factor, 1)))
        else:
            self.attn_mlp.append(nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        # upsample previous feature
        self.upsample1 = nn.Upsample(scale_factor=(up_factor, 1)) if up_factor else nn.Identity()
        self.upsample2 = nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()

        # residual connection
        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        if in_channel != out_channel:
            self.residual_layer = nn.Conv1d(in_channel, out_channel, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, pos, key, query, upfeat):
        """
        Inputs:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
        """

        value = self.mlp_v(torch.cat([key, query], 1))  # (B, dim, N)
        identity = value
        key = self.conv_key(key)  # (B, dim, N)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)

        key = grouping_operation(key, idx_knn)  # (B, dim, N, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)

        # upfeat embedding
        if self.use_upfeat:
            upfeat = self.conv_upfeat(upfeat)  # (B, dim, N)
            upfeat_rel = upfeat.reshape((b, -1, n, 1)) - grouping_operation(upfeat, idx_knn)  # (B, dim, N, k)
        else:
            upfeat_rel = torch.zeros_like(qk_rel)

        # attention
        attention = self.attn_mlp(qk_rel + pos_embedding + upfeat_rel)  # (B, dim, N*up_factor, k)

        # softmax function
        attention = self.scale(attention)

        # knn value is correct
        value = grouping_operation(value, idx_knn) + pos_embedding + upfeat_rel  # (B, dim, N, k)
        value = self.upsample1(value)  # (B, dim, N*up_factor, k)

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # (B, dim, N*up_factor)
        y = self.conv_end(agg)  # (B, out_dim, N*up_factor)

        # shortcut
        identity = self.residual_layer(identity)  # (B, out_dim, N)
        identity = self.upsample2(identity)  # (B, out_dim, N*up_factor)

        return y + identity


class UpLayer(nn.Module):
    """
    Upsample Layer with upsample transformers
    """

    def __init__(self, dim, seed_dim, up_factor=2, i=0, radius=1, n_knn=20, interpolate='three', attn_channel=True):
        super(UpLayer, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.n_knn = n_knn
        self.interpolate = interpolate

        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + seed_dim, layer_dims=[256, dim])

        self.uptrans1 = UpTransformer(dim, dim, dim=64, n_knn=self.n_knn, use_upfeat=True, up_factor=None)
        self.uptrans2 = UpTransformer(dim, dim, dim=64, n_knn=self.n_knn, use_upfeat=True, attn_channel=attn_channel,
                                      up_factor=self.up_factor)

        self.upsample = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=dim * 2, hidden_dim=dim, out_dim=dim)

        self.mlp_delta = MLP_CONV(in_channel=dim, layer_dims=[64, 3])

    def forward(self, pcd_prev, seed, seed_feat, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, feat_dim, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_new: Tensor, upsampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape

        # Collect seedfeature
        if self.interpolate == 'nearest':
            idx = get_nearest_index(pcd_prev, seed)
            feat_upsample = indexing_neighbor(seed_feat, idx).squeeze(3)  # (B, seed_dim, N_prev)
        elif self.interpolate == 'three':
            # three interpolate
            idx, dis = get_nearest_index(pcd_prev, seed, k=3, return_dis=True)  # (B, N_prev, 3), (B, N_prev, 3)
            dist_recip = 1.0 / (dis + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)  # (B, N_prev, 1)
            weight = dist_recip / norm  # (B, N_prev, 3)
            feat_upsample = torch.sum(indexing_neighbor(seed_feat, idx) * weight.unsqueeze(1),
                                      dim=-1)  # (B, seed_dim, N_prev)
        else:
            raise ValueError('Unknown Interpolation: {}'.format(self.interpolate))

        # Query mlps
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_upsample], 1)
        Q = self.mlp_2(feat_1)

        # Upsample Transformers
        H = self.uptrans1(pcd_prev, K_prev if K_prev is not None else Q, Q, upfeat=feat_upsample)  # (B, 128, N_prev)
        feat_child = self.uptrans2(pcd_prev, K_prev if K_prev is not None else H, H,
                                   upfeat=feat_upsample)  # (B, 128, N_prev*up_factor)

        # Get current features K
        H_up = self.upsample(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        # New point cloud
        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius ** self.i  # (B, 3, N_prev * up_factor)
        pcd_new = self.upsample(pcd_prev)
        pcd_new = pcd_new + delta

        return pcd_new, K_curr


class SeedFormer(nn.Module):
    """
    SeedFormer Point Cloud Completion with Patch Seeds and Upsample Transformer
    """

    def __init__(self, feat_dim=512, embed_dim=256, num_p0=512, n_knn=20, radius=1, up_factors=[1, 4], seed_factor=2,
                 interpolate='three', attn_channel=True, pretrain=''):
        """
        Args:
            feat_dim: dimension of global feature
            embed_dim: dimension of embedding feature
            num_p0: number of P0 coarse point cloud
            up_factors: upsampling factors
            seed_factor: seed generation factor
            interpolate: interpolate seed features (nearest/three)
            attn_channel: transformer self-attention dimension (channel/point)
        """
        super(SeedFormer, self).__init__()
        self.num_p0 = num_p0

        # Seed Generator
        self.feat_extractor = IC_Encoder(pretrain=pretrain)
        self.seed_generator = SeedGenerator(feat_dim=feat_dim, seed_dim=embed_dim, n_knn=n_knn, factor=seed_factor,
                                            attn_channel=attn_channel)

        # Upsample layers
        up_layers = []
        for i, factor in enumerate(up_factors):
            up_layers.append(
                UpLayer(dim=embed_dim, seed_dim=embed_dim, up_factor=factor, i=i, n_knn=n_knn, radius=radius,
                        interpolate=interpolate, attn_channel=attn_channel))
        self.up_layers = nn.ModuleList(up_layers)

    def forward(self, partial_cloud):
        """
        Args:
            partial_cloud: (B, N, 3)
        """
        # Encoder
        feat, patch_xyz, patch_feat = self.forward_encoder(partial_cloud)

        # print(feat.shape, patch_xyz.shape, patch_feat.shape)

        # Decoder
        pred_pcds = self.forward_decoder(feat, partial_cloud, patch_xyz, patch_feat)

        return pred_pcds

    def forward_encoder(self, partial_cloud):
        # feature extraction
        # partial_cloud = partial_cloud.permute(0, 2, 1).contiguous()
        feat, patch_xyz, patch_feat = self.feat_extractor(partial_cloud)  # (B, feat_dim, 1)

        return feat, patch_xyz, patch_feat

    def forward_decoder(self, feat, partial_cloud, patch_xyz, patch_feat):
        """
        Args:
            feat: Tensor, (B, feat_dim, 1)
            partial_cloud: Tensor, (B, N, 3)
            patch_xyz: (B, 3, 128)
            patch_feat: (B, seed_dim, 128)
        """
        pred_pcds = []

        # Generate Seeds
        seed, seed_feat = self.seed_generator(feat, patch_xyz, patch_feat)
        seed = seed.permute(0, 2, 1).contiguous()  # (B, num_pc, 3)
        pred_pcds.append(seed)

        # Upsample layers
        pcd = fps_subsample(torch.cat([seed, partial_cloud], 1), self.num_p0)  # (B, num_p0, 3)
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()  # (B, 3, num_p0)
        seed = seed.permute(0, 2, 1).contiguous()  # (B, 3, 256)
        for layer in self.up_layers:
            pcd, K_prev = layer(pcd, seed, seed_feat, K_prev)
            pred_pcds.append(pcd.permute(0, 2, 1).contiguous())

        return pred_pcds


###########################
# Recommended Architectures
###########################

def seedformer_mvp(**kwargs):
    model = SeedFormer(feat_dim=512, embed_dim=256, n_knn=20, up_factors=[1, 4], **kwargs)
    return model


def seedformer_pcn(**kwargs):
    model = SeedFormer(feat_dim=512, embed_dim=256, n_knn=20, up_factors=[1, 4, 8], **kwargs)
    return model


if __name__ == '__main__':

    model = seedformer_mvp().cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_part = torch.rand(1, 2048, 3).cuda()
    y = model(x_part)
    print([pc.size() for pc in y])
