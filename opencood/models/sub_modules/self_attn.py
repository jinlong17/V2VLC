import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

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



class V2V_AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(V2V_AttFusion, self).__init__()
        
        # self.attention1 = PAM_Module(feature_dim)
        # self.attention2 = CAM_Module(feature_dim)
        # self.attention3 = UFOAttention(feature_dim)
        # self.attention4 = TripletAttention()
        # self.attention5 = PSA(channel=feature_dim, reduction=8)
        # self.attention6 = OutlookAttention(feature_dim, num_heads=16)
        # self.attention7 = SpatialGroupEnhance(groups=8)
        # self.attention8 = DoubleAttention(in_channels=feature_dim ,c_m=256,c_n=256)
        # self.attention9 = ShuffleAttention(channel=feature_dim, G=8)


        # self.cov_ego = torch.nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)
        # self.cov2_cav = torch.nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)

        # self.attention_1 = my_attention(feature_dim)
        self.attention_2 = CrissCrossAttention(feature_dim)

        

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)  #x =[5, 64, 100, 352], record_len=[3,2]
        batch_size = len(record_len)
        C, W, H = split_x[0].shape[1:]

        out = []
        for xx in split_x:#split_x[0] [num_car, C, W, H]
          cav_num = xx.shape[0]


          # '''Co_method_1: original version: attention for ego vehicle feature + cav feature  '''

          # xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
          # att_ego, att_cav = self.attention_1(xx)
          # # pdb.set_trace()

          # # att = att.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...].unsqueeze(0)
          # att_ego = att_ego.permute(1,2,0).view(1, C, W, H)

          # if type(att_cav)==int:
          #       att = self.cov_ego(att_ego)

          # else:
            
          #   att_cav = att_cav.permute(1,2,0).view(1, C, W, H)
            
          
          #   att = self.cov_ego(att_ego)+ self.cov2_cav(att_cav)
            
          # out.append(att) #[[1, 64, 100, 352], [1, 64, 100, 352]]

          ''' CCNet: Criss-Cross Attention Module: attention for ego vehicle feature + cav feature '''

          ego_q, ego_k, ego_v = xx[0:1], xx[0:1], xx[0:1] 
          context = torch.zeros_like(ego_q)
          for i in range(len(xx[:,0,0,0])):
            # pdb.set_trace()
            att_vehicle = self.attention_2(ego_q, xx[i:i+1], xx[i:i+1])
            context = att_vehicle + context

          att = context

          # pdb.set_trace()
          out.append(att) #[[1, 64, 100, 352], [1, 64, 100, 352]]
          torch.cuda.empty_cache()

        #   ''' CCNet:contrast experiments '''



        #   xx = xx.view(1, -1, W, H)
        #   # feat_in_dim = xx.shape[3]
        #   # pdb.set_trace()
        #   att = self.attention_2(xx, xx, xx)
        #   # print("att shape:", att.size())
        #   # pdb.set_trace()
        #   # att_1 = att.permute(0,3,2,1)
        #   # cov = torch.nn.Conv2d(in_channels=feat_in_dim, out_channels=H, kernel_size=1).cuda()
        #   att_ego = att[:,0:C,:,:]
        #   # att_ego = att_ego.permute(0,3,2,1)

        #   out.append(att_ego)


            # '''Co_method_1: PCAM + PCAM '''

            # att1 = self.attention1(xx)
            # att2 = self.attention2(xx)
            # att1 = att1[0, ...].unsqueeze(0)
            # att2 = att2[0, ...].unsqueeze(0)
            # att_multi = (att1+att2)*0.5

            # att3 = self.attention1(xx[0].unsqueeze(0))
            # att4 = self.attention2(xx[0].unsqueeze(0))
            # att_ego = att3+att4

            # att = att_multi +att_ego
            # out.append(att)

            # '''Co_method_2: PCAM+Triplet_conv '''

            # att1 = self.attention1(xx)
            # att2 = self.attention2(xx)
            # att1 = att1[0, ...].unsqueeze(0)
            # att2 = att2[0, ...].unsqueeze(0)
            # att_multi = att1+att2

            # att3 = self.attention4(xx[0].unsqueeze(0))
            # att_ego = att3

            # att = self.cov1(att_multi) + self.cov2(att_ego)
            # out.append(att)

            # '''Co_method_3: PCAM+shuffle_conv '''

            # att1 = self.attention1(xx)
            # att2 = self.attention2(xx)
            # att1 = att1[0, ...].unsqueeze(0)
            # att2 = att2[0, ...].unsqueeze(0)
            # att_multi = att1+att2

            # att3 = self.attention9(xx[0].unsqueeze(0))
            # att_ego = att3

            # att = self.cov1(att_multi) + self.cov2(att_ego)
            # out.append(att)


            # '''Co_method_4: PCAM+A2_conv '''
            

            # att1 = self.attention1(xx)
            # att2 = self.attention2(xx)
            # att1 = att1[0, ...].unsqueeze(0)
            # att2 = att2[0, ...].unsqueeze(0)
            # att_multi = att1+att2

            # att3 = self.attention8(xx[0].unsqueeze(0))
            # att_ego = att3

            # att = self.cov1(att_multi) + self.cov2(att_ego)
            # out.append(att)

            # '''Method_1: PAM'''
            # att1 = self.attention1(xx)
            # att1 = att1[0, ...].unsqueeze(0)
            # att = att1
            # out.append(att)

            # '''Method_2: PAM+CAM'''
            # att1 = self.attention1(xx)
            # att2 = self.attention2(xx)
            # att1 = att1[0, ...].unsqueeze(0)
            # att2 = att2[0, ...].unsqueeze(0)
            # att = att1+att2
            # out.append(att)

            # '''Method_3: UFOAttention Nan!!!'''
            # xx = xx.view(cav_num, C, -1).permute(2, 0, 1) #[35200, 2, 64]
            # att1 = self.attention3(xx,xx,xx)
            # att1 = att1.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...].unsqueeze(0)
            # att = att1
            # out.append(att)

            # '''Method_4:TripletAttention'''
            # att1 = self.attention4(xx)
            # att1 = att1[0, ...].unsqueeze(0)
            # att = att1
            # out.append(att)

            # '''Method_5:PSA Attention'''
            # xx = xx.view(1, C, cav_num, H*W)
            # att1 = self.attention5(xx)
            # att1 = att1.permute(2, 1, 0, 3).view(cav_num, C, W, H)[0, ...].unsqueeze(0)
            # att = att1
            # out.append(att)

            # '''Method_6:OutlookAttention'''
            # xx = xx.view(1, H*W, cav_num, C)
            # att1 = self.attention6(xx)
            # att1 = att1.permute(2, 3, 0, 1).view(cav_num, C, W, H)[0, ...].unsqueeze(0)
            # att = att1
            # out.append(att)

            # '''Method_7:SpatialGroupEnhance'''
            # xx = xx.view(1, H*W, cav_num, C)
            # att1 = self.attention7(xx)
            # att1 = att1.permute(2, 3, 0, 1).view(cav_num, C, W, H)[0, ...].unsqueeze(0)
            # att = att1
            # out.append(att)


            # '''Method_8:A2Atttention'''
            # xx = xx.view(1, C, cav_num, H*W)
            # att1 = self.attention8(xx)
            # att1 = att1.permute(2, 1, 0, 3).view(cav_num, C, W, H)[0, ...].unsqueeze(0)
            # att = att1
            # out.append(att)

            # '''Method_9:ShuffleAttention'''
            # xx = xx.view(1, C, cav_num, H*W)
            # att1 = self.attention9(xx)
            # att1 = att1.permute(2, 1, 0, 3).view(cav_num, C, W, H)[0, ...].unsqueeze(0)
            # att = att1
            # out.append(att)


        return torch.cat(out, dim=0) #[2, 64, 100, 352]

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        # pdb.set_trace()
        return split_x

def regroup(x, record_len):
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
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, query, key, value):
        m_batchsize, _, height, width = query.size()

        proj_query = query
        proj_key = key
        proj_value = value
        
        # proj_query = self.query_conv(query)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        
        # proj_key = self.key_conv(key)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        
        
        # proj_value = self.value_conv(value)
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
