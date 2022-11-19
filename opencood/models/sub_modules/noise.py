'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-05-04 10:50:25
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2022-11-18 00:09:21
'''
import numpy as np
# from youtube_dl import main
import torch
import torch.nn as nn

import random
from opencood.models.sub_modules.self_attn import regroup

import pdb





def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True









def add_noise(x):

    setup_seed(100)

    N, C, W, H = x.shape[:]

    num = np.random.randint(C-4)

    # noise = torch.tensor.ones_like(x)

    noise = torch.randn(N-1,3,W,H)

    x[1:,num:num+3,:,:] = x[1:,num:num+3,:,:] + noise.cuda()

    return x




def sub_noise(x):

    setup_seed(100)

    noise = torch.ones_like(x)

    N, C, W, H = x.shape[:]

    num = np.random.randint(C-5)

    N = np.random.randint(0, 6)


    while N > 0:
        h = np.random.randint(0,H-4)
        w = np.random.randint(0,W-4)

        noise = torch.ones(N-1,1,W,H)

        # noise[N-1,1, w:w+3, h:h+3] = 0
        x[1:,num:num+3,w:w+3, h:h+3] = 0


        N= N-1

    # x[1:,num:num+1,:,:] = x[1:,num:num+1,:,:]*noise.cuda()

    return x






def data_dropout(x, p, train=True):

    if p <0. or p >=1:

        raise ValueError('p must be in interval[0, 1]')
        
    # pdb.set_trace()
    
    # size = 200
    one_prob = 1. - p


    if train:# all feature are included in the training  (ego and other cav)
        size = x.size()
        random_tensor = np.random.binomial(n=1, p = one_prob, size = size)

        random_tensor = torch.from_numpy(random_tensor)

        x = x*random_tensor.cuda()


    else:# only other cav is included in the testing

        size = x[1:,].size()
        random_tensor = np.random.binomial(n=1, p = one_prob, size = size)

        random_tensor = torch.from_numpy(random_tensor)

        x[1:,] = x[1:,]*random_tensor.cuda()

        # if x.size()[0] == 1:
            # print("noise:", x.size())
            # print(x[1:,].size())
            # pdb.set_trace()
    # x = sum(random_tensor)/size

    return  x

    
def data_dropout_uniform(x, p, key_max):

    if p <0. or p >=1:

        raise ValueError('p must be in interval[0, 1]')

    one_prob = 1. - p

    xx=x.clone()

    size = x[1:,].size()
    random_tensor = np.random.binomial(n=1, p = one_prob, size = size)

    random_tensor = torch.from_numpy(random_tensor)

    # print("random_tensor", random_tensor)
    adverse_random_tensor = random_tensor.clone()

    one_tensor =  torch.ones(size)

    adverse_random_tensor = one_tensor-adverse_random_tensor

    # adverse_random_tensor[adverse_random_tensor==0]=2
    # adverse_random_tensor[adverse_random_tensor==1]=0
    # adverse_random_tensor[adverse_random_tensor==2]=1

    # print("adverse_random_tensor", adverse_random_tensor)

    # noise=torch.rand(size)
    noise=torch.randint(0, int(key_max), size)



    # print("before_noise: ", noise)

    noise = noise*adverse_random_tensor

    # print("after_noise: ", noise)


    x[1:,] = x[1:,]*random_tensor.cuda()
    x[1:,] = x[1:,]+noise.cuda()


    # print("Noise percentage#########################: ", torch.all(torch.eq(xx,x)))
    return  x




def transmission_with_noise(feature, record_len, precentage, key_max):
    '''
    Input:
    feature: input feature map
    record_len:  divide the feature into several group according to scenario
    precentage: the percentage of noise in the feature map

    Output:
    spatial_features_2d: feature map with noise

    '''
    split_x = []
    split_xx = regroup(feature, record_len)
    for xx in split_xx:
        if xx.size()[0] > 1:# generating training noise except ego feature map
            x = data_dropout_uniform(xx.clone(), p=precentage, key_max=key_max)
            # print("Noise percentage#########################: ", torch.all(torch.eq(xx,x)))
            split_x.append(x)
        else:
            split_x.append(xx)
    features_2d = torch.cat(split_x, dim=0)
    # print("len_split_x: ",len(split_x),  "features_2d Noise percentage#########################: ", torch.all(torch.eq(features_2d,feature)))

    return features_2d



if __name__== '__main__':


    input = torch.ones(2,3,2,5).cuda()
    print(input)
    # output = data_dropout(input, p=0.5)
    output=data_dropout_uniform(input, p=0.5)
    print(output)

    






