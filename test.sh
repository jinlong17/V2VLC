#!/bin/sh
###
 # @Descripttion: 
 # @version: 
 # @Author: Jinlong Li CSU PhD
 # @Date: 2022-07-03 18:01:32
 # @LastEditors: Jinlong Li CSU PhD
 # @LastEditTime: 2024-02-03 17:53:18
### 

# source /home/jinlong/anaconda3/etc/profile.d/conda.sh


# conda activate opencood



# model='/home/jinlong/1.Detection_Sets/TIV_LCRN_V2VAM_2023/Source/model_logs/point_pillar_intermediate_V2VAM_nocompressionV1_2023_05_02_22'
# model='/home/jinlong/jinlong_NAS/1.model_trained_saved/V2V_noise/NEW_log/point_pillar_intermediate_V2VAM_nocompression_w_lcrn_finetuned_modified_2024_01_22_10_57_44'

# model='/home/jinlong/Desktop/TO-DO-2023/DA_CPVIT_model23/no-DA/attfuse'
# model='/home/jinlong/Desktop/TO-DO-2023/DA_CPVIT_model23/no-DA/attfuse'

# model='/home/jinlong/jinlong_NAS/1.model_trained_saved/V2V_noise/NEW_log/point_pillar_intermediate_V2VAM_nocompression_LCRN_2024_01_07_17_23_56'
# model='/home/jinlong/jinlong_NAS/1.model_trained_saved/V2V_noise/NEW_log/point_pillar_intermediate_V2VAM_nocompression_LCRN_2024_01_08_16_55_51'
# model='/home/jinlong/jinlong_NAS/1.model_trained_saved/V2V_noise/NEW_log/point_pillar_intermediate_V2VAM_nocompression_w_lcrn_2024_01_10_23_59_36'


# model='/home/jinlong/jinlong_NAS/1.model_trained_saved/V2V_noise/NEW_log/point_pillar_intermediate_V2VAM_nocompression_w_lcrn_perfect_2024_01_16_12_04_43'

# model='/home/jinlong/jinlong_NAS/1.model_trained_saved/V2V_noise/NEW_log/point_pillar_intermediate_V2VAM_nocompression_w_lcrn_finetuned_2024_01_21_18_41_46'


# model='/home/jinlong/1.Detection_Sets/TIV_LCRN_V2VAM_2023/Source/model_logs/point_pillar_intermediate_V2VAM_nocompressionV2_2023_05_04_15'



# model='/home/jinlongli/1.Detection_Set/TIV_LCRN/NEW_log/point_pillar_intermediate_V2VAM_nocompression_2024_01_06_01_10_11'
# model='/home/jinlongli/1.Detection_Set/TIV_LCRN/NEW_log/1.NEW/V2VAM_nocompression_LCRN_perfect_2024_02_21_08_20_06'

# model='/home/jinlongli/1.Detection_Set/TIV_LCRN/NEW_log/1.NEW/V2VAM_compression_LCRN_perfect_1_2024_03_02_18_47_45'
# model='/home/jinlongli/1.Detection_Set/TIV_LCRN/NEW_log/1.NEW/V2VAM_nocompression_LCRN_perfect_1_2024_02_22_21_56_24'
# run python script

# model='/home/jinlongli/1.Detection_Set/TIV_LCRN/NEW_log/4.lcrn/1.V2VAM_nocompression_noise1_2024_03_24_13_26_48'
model='/home/jinlongli/1.Detection_Set/TIV_LCRN/NEW_log/4.lcrn/2.V2VAM_nocompression_sigmod_perfect_2024_04_07_15_05_13'


CUDA_VISIBLE_DEVICES=1 python3 opencood/tools/inference.py \
    --fusion_method intermediate \
    --model_dir $model 


# for LCRN 

# CUDA_VISIBLE_DEVICES=1 python3 opencood/tools/inference_lcrn.py \
#     --fusion_method intermediate \
#     --model_dir $model 


    
    

# cd ../




# exit the virtual environment
# conda deactivate