import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone

from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor

######
from opencood.models.sub_modules.LCRN import Repaired_Net, eval_noise_Loss
from opencood.models.sub_modules.V2VAM import V2V_AttFusion
from opencood.models.sub_modules.LC_noise  import regroup, data_dropout_uniform
import numpy as np


class PointPillarV2VAMLCRN(nn.Module):
    def __init__(self, args):
        super(PointPillarV2VAMLCRN, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])


        if args['backbone_fix']:
            self.backbone_fix()


        self.noise = args['lc_noise']
        self.key_max = args['lc_noise_max']

        if self.noise:

            print("lossy communication is considered under V2V")

    '''
    add the noise for feature map
    '''

    def operation_noise(self, x, record_len):

        split_x = regroup(x, record_len)
        out = []
        p = np.random.randint(0, 100)/100
        for xx in split_x:
            if xx.size()[0] > 1: # generating training noise without ego feature map
                xx = data_dropout_uniform(xx, p=p, key_max=self.key_max)
            out.append(xx)
        
        x = torch.cat(out, dim=0)
        return x


    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding =\
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)


        #add the LC noise
        if self.noise:
            spatial_features_2d_noise = self.operation_noise(spatial_features_2d.clone(), record_len)
        else:
            spatial_features_2d_noise = spatial_features_2d


        output_dict = {'spatial_features_2d_noise': spatial_features_2d_noise,
                        'spatial_features_2d': spatial_features_2d,
                        'record_len': record_len}
        

        return output_dict



class fusion_module(nn.Module):
    def __init__(self, args):
        super(fusion_module, self).__init__()

        ###### fusion module
        self.fusion_net = V2V_AttFusion(256)

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

    def forward(self, spatial_features_2d, output_dict):

        record_len = output_dict['record_len']

        fused_feature = self.fusion_net(spatial_features_2d, record_len)

        # log the smooth_l1_loss after the denoising 
        L1_noise_loss = eval_noise_Loss(spatial_features_2d, output_dict['spatial_features_2d'])

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        final_dict = {'psm': psm,
                       'rm': rm,
                       'L1_diff_loss': L1_noise_loss.item(),
                       'L1_diff_loss_computed': L1_noise_loss
        }

        return final_dict



class LCRN_module(nn.Module):
    def __init__(self):
        super(LCRN_module, self).__init__()

        self.LCRN = Repaired_Net(in_channel=256)###[64, 128, 256, 384]

    def forward(self, output_dict):
        record_len = output_dict['record_len']
        feature = output_dict['spatial_features_2d_noise']
        out = []
        split_x = regroup(feature, record_len)
        for xx in split_x:
            if xx[1:,].size()[0] > 0:
                b = xx[1:,].clone()
                b = self.LCRN(b, b)
                xx[1:,] = b.clone()
                b=b.cpu()
                out.append(xx)
            else:
                out.append(xx)
        features_2d_repair = torch.cat(out, dim=0)

        return features_2d_repair