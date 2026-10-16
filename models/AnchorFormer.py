# -*- encoding: utf-8 -*-
import torch
from torch import nn

from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.expansion_penalty.expansion_penalty_module import expansionPenaltyModule
from models.Transformer import AnchorTransformer
from models.Morphing import PointMorphing
from models.utilss import fps_subsample


def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc    

# 3D completion
class AnchorFormer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.trans_dim = 384
        self.num_pred = 16384
        self.num_query = 256
        self.num_encoder_blk = 8
        self.num_decoder_blk = 6
        self.up_ratio = self.num_pred//self.num_query
        # self.fold_step = int(pow(self.num_pred // self.num_query, 0.5) + 0.5)    # PCN & PlantPcom
        self.fold_step = 2   # MVP
        print(self.fold_step)
        
        self.base_model = AnchorTransformer(in_chans = 3, embed_dim = self.trans_dim, depth = [self.num_encoder_blk, self.num_decoder_blk], num_query = self.num_query)
        self.upsample_net = PointMorphing(self.trans_dim, step = self.fold_step, hidden_dim = 256)  # rebuild a cluster point 
        # self.upsample_net = ArbitraryPointMorphing(self.trans_dim, self.up_ratio, hidden_dim = 256)  # rebuild a cluster point 

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map_global = nn.Linear(1024, self.trans_dim)
        self.reduce_map_local = nn.Linear(self.trans_dim + 3, self.trans_dim)
        
        self.include_input = False

    def forward(self, xyz):
        q, coarse_point_cloud = self.base_model(xyz) # B M C and B M 3
        # print(q.shape, coarse_point_cloud.shape)  #torch.Size([2, 256, 384]) torch.Size([2, 256, 3])

        self.pred_coarse_point = coarse_point_cloud
    
        B, M ,C = q.shape

        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024
        
        rebuild_feature = torch.cat([global_feature.unsqueeze(-2).expand(-1, M, -1), q, coarse_point_cloud], dim=-1)  # B M 1027 + C
        global_feature = rebuild_feature[:,:,:1024].reshape(B*M,1024)
        local_feature = rebuild_feature[:,:,1024:].reshape(B*M,self.trans_dim+3)
        global_feature = self.reduce_map_global(global_feature)
        local_feature = self.reduce_map_local(local_feature)
        
        relative_xyz = self.upsample_net(global_feature, local_feature).reshape(B, M, 3, -1)    # B M 3 S
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3).reshape(B, -1, 3)  # B N 3

        self.pred_dense_point = rebuild_points

        inp_sparse = fps(xyz, self.num_query)
        coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        if self.include_input: rebuild_points = torch.cat([rebuild_points, xyz], dim=1).contiguous()

        median_points = fps_subsample(torch.cat([rebuild_points, xyz], dim=1), 2048)

        if self.fold_step == 2:
            return (coarse_point_cloud, rebuild_points, median_points)
        else:
            return (coarse_point_cloud, median_points, median_points, rebuild_points)


if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_part = torch.randn(1, 2048, 3, requires_grad=True).to(device)

    model = AnchorFormer().to(device)
    all_pc = model(x_part)
    for i in all_pc:
        print(i.shape)
