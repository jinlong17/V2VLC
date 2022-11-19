import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.self_attn import AttFusion
from opencood.models.sub_modules.auto_encoder import AutoEncoder

######
from opencood.models.sub_modules.KPN import KPN, KPN_cov, LossFunc, eval_noise_Loss, LossBasic, KPN_MAX, KPN_with_core, LossBasic
from opencood.models.sub_modules.self_attn import V2V_AttFusion, regroup
from opencood.models.sub_modules.noise import data_dropout, data_dropout_uniform,transmission_with_noise

class AttBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False

        if 'compression' in model_cfg and model_cfg['compression'] > 0:
            self.compress = True
            self.compress_layer = model_cfg['compression']

        if 'layer_nums' in self.model_cfg:

            assert len(self.model_cfg['layer_nums']) == \
                   len(self.model_cfg['layer_strides']) == \
                   len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums']
            layer_strides = self.model_cfg['layer_strides']
            num_filters = self.model_cfg['num_filters']
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) \
                   == len(self.model_cfg['num_upsample_filter'])

            num_upsample_filters = self.model_cfg['num_upsample_filter']
            upsample_strides = self.model_cfg['upsample_strides']

        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.fuse_modules = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        if self.compress:
            self.compression_modules = nn.ModuleList()


        #TODO: ################KPN_Setting################################
        # self.KPN = False
        self.KPN=True
        if self.KPN:
            self.KPN_module = nn.ModuleList()
            self.KPN_layer_num = self.model_cfg['KPN_channels']
            self.KPN_layer = self.model_cfg['KPN']
            print("KPN module loading done!")
            self.KPN_module = KPN(in_channel=self.KPN_layer_num[0])###[64, 128, 256]
        self.KPN_loss=[]
        ################KPN_Setting################################
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]

            fuse_network = AttFusion(num_filters[idx])
            self.fuse_modules.append(fuse_network)
            if self.compress and self.compress_layer - idx > 0:
                self.compression_modules.append(AutoEncoder(num_filters[idx],
                                                            self.compress_layer-idx))

            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx],
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])

            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
                                       momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in



    '''
    add the noise for feature map
    '''
    def operation_noise(self, x, record_len):

        split_x = regroup(x, record_len)
        out = []
        p = np.random.randint(0, 100)/100
        # p=0.5
        key_max=30 #22.5
        for xx in split_x:
            if xx.size()[0] > 1: # generating training noise without ego feature map
                xx = data_dropout_uniform(xx, p=p, key_max=key_max)
            out.append(xx)
        x = torch.cat(out, dim=0)
        return x

    def feature_repair(self, xx):
        
        if xx[1:,].size()[0] > 0:
            b = xx[1:,].clone()
            # with torch.no_grad():
            b = self.KPN_module(b, b)
            xx[1:,] = b.clone()
            b=b.cpu()
        xx=xx
        return xx


    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        record_len = data_dict['record_len']

        ups = []
        ret_dict = {}
        x = spatial_features

        #########################
        KPN_losses=[]#### store the loss
        record_len = data_dict['record_len']
        #########################


        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if self.compress and i < len(self.compression_modules):
                x = self.compression_modules[i](x)
            x_fuse = self.fuse_modules[i](x, record_len)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x_fuse))
            else:
                ups.append(x_fuse)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        ####################
        xx = self.operation_noise( x.clone(), record_len) 
        KPN_loss= eval_noise_Loss(xx, x)
        KPN_losses.append(KPN_loss.item()) 
        self.KPN_loss.append(sum(KPN_losses)) # record KPN loss

        if self.KPN:
            xx =self.feature_repair(xx)        

        data_dict['spatial_features_2d'] = xx
        data_dict['L1_diff_loss'] = self.KPN_loss[-1]
        torch.cuda.empty_cache()
        ####################
        # data_dict['spatial_features_2d'] = x

        return data_dict
