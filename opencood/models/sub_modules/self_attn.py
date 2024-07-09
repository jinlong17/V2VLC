import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        batch_size = len(record_len)
        C, W, H = split_x[0].shape[1:]
        out = []
        for xx in split_x:
            cav_num = xx.shape[0]
            xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(xx, xx, xx)
            h = h.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...].unsqueeze(0)
            out.append(h)
        return torch.cat(out, dim=0)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

class Adding_fusion(nn.Module):
    def __init__(self, feature_dim):
        super(Adding_fusion, self).__init__()
        self.conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.conv1 = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        batch_size = len(record_len)
        C, W, H = split_x[0].shape[1:]
        out = []
        for xx in split_x: #xx=[num_car, C, W, H]
            ego_q = xx[0:1] 
            context = torch.zeros_like(ego_q)
            for i in range(len(xx[:,0,0,0])):
                context = context + self.conv(xx[i:i+1])
            out.append(context/len(xx[:,0,0,0]))
            torch.cuda.empty_cache()
        return  self.conv1(torch.cat(out, dim=0))

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

class V2V_AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(V2V_AttFusion, self).__init__()
        

        # self.cov_ego = torch.nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)
        # self.cov2_cav = torch.nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)

        # self.attention_1 = my_attention(feature_dim)
        self.attention_2 = CrissCrossAttention(feature_dim)
        # self.attention_3 =CAM_Module(feature_dim)
        self.cov_att = nn.Sequential(
                nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1),
                nn.BatchNorm2d(feature_dim,eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU()
            )

        

    def forward(self, x, record_len):


        #### the first attention
        # x = self.attention_2(x, x, x)


        split_x = self.regroup(x, record_len)  #x =[5, 64, 100, 352], record_len=[3,2]
        batch_size = len(record_len)
        C, W, H = split_x[0].shape[1:]

        out = []
        for xx in split_x:#split_x[0] [num_car, C, W, H]
            cav_num = xx.shape[0]

            ''' CCNet: Criss-Cross Attention Module: attention for ego vehicle feature + cav feature '''

            ego_q, ego_k, ego_v = xx[0:1], xx[0:1], xx[0:1] 
            context = torch.zeros_like(ego_q)
            for i in range(len(xx[:,0,0,0])):
                # pdb.set_trace()
                att_vehicle = self.attention_2(ego_q, xx[i:i+1], xx[i:i+1])
                context = att_vehicle + context

            # att = context + self.cov_ego(ego_v)######################################################
            att = self.cov_att(context)

            # pdb.set_trace()
            out.append(att) #[[1, 64, 100, 352], [1, 64, 100, 352]]
            torch.cuda.empty_cache()

        return torch.cat(out, dim=0) #[2, 64, 100, 352]
        # return self.attention_3(torch.cat(out, dim=0))

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        # pdb.set_trace()
        return split_x



class V2V_S_AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(V2V_S_AttFusion, self).__init__()
        
        # self.cov_ego = torch.nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)
        # self.cov_att = torch.nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)
        # self.cov_ego = nn.Sequential(
        #         nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1),
        #         nn.BatchNorm2d(feature_dim,eps=1e-5, momentum=0.01, affine=True),
        #         nn.ReLU()
        #     )
        # self.cov_att = nn.Sequential(
        #         nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1),
        #         nn.BatchNorm2d(feature_dim,eps=1e-5, momentum=0.01, affine=True),
        #         nn.ReLU()
        #     )
        self.cov_att = nn.Sequential(
                nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dim,eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU()
            )
        # self.cov_att = nn.Sequential(
        #         nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=5, padding=2),
        #         nn.BatchNorm2d(feature_dim,eps=1e-5, momentum=0.01, affine=True),
        #         nn.ReLU()
        #     )
        # self.cov_att = nn.Sequential(
        #         nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=7, padding=3),
        #         nn.BatchNorm2d(feature_dim,eps=1e-5, momentum=0.01, affine=True),
        #         nn.ReLU()
        #     )
        self.CCNet = CrissCrossAttention(feature_dim)
        # self.attention_3 =CAM_Module(feature_dim)

    def forward(self, x, record_len):

        split_x = self.regroup(x, record_len)  #x =[5, 64, 100, 352], record_len=[3,2]

        out = []
        att = []
        for xx in split_x:#split_x[0] [num_car, C, W, H]

            ''' CCNet: Criss-Cross Attention Module: attention for ego vehicle feature + cav feature '''

            ego_q, ego_k, ego_v = xx[0:1], xx[0:1], xx[0:1] 
            for i in range(len(xx[:,0,0,0])):
                att_vehicle = self.CCNet(ego_q, xx[i:i+1], xx[i:i+1])
                att.append(att_vehicle)

            pooling_max = torch.max(torch.cat(att, dim=0), dim=0, keepdim=True)[0]
            pooling_ave = torch.mean(torch.cat(att, dim=0), dim=0, keepdim=True)[0]

            fuse_fea = pooling_max + pooling_ave
            # fuse_fea = torch.cat([pooling_max, pooling_ave], dim=0)

            ####method 1
            # fuse_att = fuse_fea + ego_v
            ####method 2 note that works.
            fuse_att = fuse_fea



            fuse_att = self.cov_att(fuse_att)

            out.append(fuse_att) #[[1, 64, 100, 352], [1, 64, 100, 352]]
            torch.cuda.empty_cache()

        return torch.cat(out, dim=0) #[2, 64, 100, 352]
        # return self.attention_3(torch.cat(out, dim=0))

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())

        return split_x






class NO_Fusion_Att(nn.Module):
    def __init__(self, feature_dim):
        super(NO_Fusion_Att, self).__init__()
        self.CCNet = CrissCrossAttention(feature_dim)

    def forward(self, x, record_len):

        # split_x = self.regroup(x, record_len)  #x =[5, 64, 100, 352], record_len=[3,2]

        out = []
        att = []
        split_x = [x]
        for xx in split_x:#split_x[0] [num_car, C, W, H]

            ''' CCNet: Criss-Cross Attention Module: attention for ego vehicle feature + cav feature '''
            ego_q, ego_k, ego_v = xx[0:1], xx[0:1], xx[0:1] 
            ############# 1. no-fusion + only CCNet for Intra-Vehicle  #############
            # fuse_att = self.CCNet(ego_q, ego_k, ego_v)
            fuse_att = self.CCNet(xx, xx, xx)

            ############# 2. no-fusion + only CCNet for Intra-Vehicle  #############
            # if len(xx[:,0,0,0]) == 1:
            #     fuse_att = ego_v
            # else:
            #     for i in range(len(xx[:,0,0,0])):
            #         att_vehicle = self.CCNet(ego_q, xx[i:i+1], xx[i:i+1])
            #         att.append(att_vehicle)
            #     fuse_att = torch.sum(torch.cat(att[1:], dim=0), dim=0, keepdim=True)


            # pdb.set_trace()
            out.append(fuse_att) #[[1, 64, 100, 352], [1, 64, 100, 352]]

        return torch.cat(out, dim=0) #[2, 64, 100, 352]

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())

        return split_x


def regroup_1(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    # pdb.set_trace()
    # print("cum_sum_len: ", cum_sum_len, " split_x: ", len(split_x), " split_x[0]: ", split_x[0].size())
    return split_x


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()

        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)



        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)


        self.query_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
                nn.BatchNorm2d(in_dim,eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU()
            )
        self.key_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
                nn.BatchNorm2d(in_dim,eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU()
            )
        self.value_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
                nn.BatchNorm2d(in_dim,eps=1e-5, momentum=0.01, affine=True),
                nn.ReLU()
            )


        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, query, key, value):
        m_batchsize, _, height, width = query.size()

        # proj_query = query
        # proj_key = key
        # proj_value = value
        
        proj_query = self.query_conv(query)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        
        proj_key = self.key_conv(key)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        
        
        proj_value = self.value_conv(value)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        # pdb.set_trace()
        return self.gamma*(out_H + out_W) + value
        # return self.gamma*(out_H + out_W)





class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        # pdb.set_trace()
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out