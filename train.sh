#!/bin/sh
###
 # @Descripttion: 
 # @version: 
 # @Author: Jinlong Li CSU PhD
 # @Date: 2024-01-06 00:12:24
 # @LastEditors: Jinlong Li CSU PhD
 # @LastEditTime: 2024-02-01 14:20:08
### 




model='/home/jinlongli/1.Detection_Set/TIV_LCRN/NEW_log/4.lcrn/old/point_pillar_intermediate_V2VAM_nocompression_2024_01_06_01_10_11/net_epoch19.pth'

# hypes_yaml='/home/jinlongli/1.Detection_Set/TIV_LCRN/LC_V2V/opencood/hypes_yaml/point_pillar_intermediate_V2VAM.yaml'





# model='/home/jinlongli/1.Detection_Set/TIV_LCRN/NEW_log/1.NEW/V2VAM_nocompression_LCRN_perfect_1_2024_02_22_21_56_24/net_epoch90.pth'




hypes_yaml='/home/jinlongli/1.Detection_Set/TIV_LCRN/LC_V2V/opencood/hypes_yaml/point_pillar_intermediate_V2VAM_w_lcrn.yaml'




# run python script


######## using for oringinal point pillar training
# CUDA_VISIBLE_DEVICES=5 python3 opencood/tools/train.py  --hypes_yaml $hypes_yaml --model $model  #--model_dir $path #--model_dir $path #--model $model #--model_dir $path #--model $model   #--model 


# conda activate lcrn

###### using for LCRN and point pillar training together
CUDA_VISIBLE_DEVICES=7  python3 opencood/tools/train_w_lcrn.py  --hypes_yaml $hypes_yaml --model $model  #--LCRN_module $model  #--model_dir $path  #--model $model  




###### using for LCRN and point pillar training together
# CUDA_VISIBLE_DEVICES=1  python3 opencood/tools/train_lcrn.py  --hypes_yaml $hypes_yaml --model $model  #--LCRN_module $model  #--model_dir $path  #--model $model  







# exit the virtual environment
# conda deactivate