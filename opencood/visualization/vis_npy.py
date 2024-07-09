'''
Descripttion: 
version: 
Author: Jinlong Li CSU PhD
Date: 2022-11-03 23:53:26
LastEditors: Jinlong Li CSU PhD
LastEditTime: 2023-09-11 18:24:12
'''
import os

import numpy as np
import open3d as o3d

import sys
sys.path.append("/home/jinlong/4.3D_detection/Noise_V2V/v2vreal")
# import sys
# import os
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)
# sys.path.remove("/home/jinlong/1.Detection_Sets/V2V4Real")
print(sys.path)



from opencood.visualization.vis_utils import visualize_sequence_sample_output, \
    visualize_single_sample_output_gt , visualize_sequence_sample_output_1

if __name__ == "__main__":
    # save_npy_path = '../logs/' \
                    # 'point_pillar_late_fusion_low_res_2021_09_08_09_55_22/npy'
    
    # save_npy_path = '/home/jinlong/jinlong_NAS/1.model_trained_saved/V2V_noise/1.V2VAM_KPN_all__confident_V3_2023_08_07_17_35_36/npy'
    # save_npy_path='/home/jinlong/Desktop/LiDAR/vis_2023_08_09_15/npy'
    # save_npy_path='/home/jinlong/Desktop/LiDAR/1.attfuse_random_x_2023_08_09_09/npy'
    # save_npy_path='/home/jinlong/Desktop/LiDAR/1.attfuse_random_y1.118_2023_08_09_09/npy'
    # save_npy_path='/home/jinlong/Desktop/LiDAR/1.attfuse_random_y_2023_08_09_09/npy'
    # save_npy_path='/home/jinlong/Desktop/LiDAR/1.attfuse_random_z_2023_08_09_09/npy'

    # save_npy_path='/home/jinlong/Desktop/LiDAR/1.attfuse_random_x1.118_2023_08_09_09/npy'
    # save_npy_path=''
    # save_npy_path='/home/jinlong/Desktop/1.attfuse_sign_bp_result_withpose/1.attfuse_sign_bp_all_2023_08_15_16/npy'
    # save_npy_path= '/home/jinlong/Desktop/1.attfuse_sign_bp_result_withpose/1.attfuse_sign_bp_x_2023_08_15_15/npy'
    # save_npy_path= '/home/jinlong/Desktop/1.attfuse_sign_bp_result_withpose/1.attfuse_sign_bp_y_2023_08_15_15/npy'
    # save_npy_path= '/home/jinlong/Desktop/1.attfuse_sign_bp_result_withpose/1.attfuse_sign_bp_z_2023_08_15_16/npy'
    # save_npy_path= '/home/jinlong/Desktop/1.attfuse_sign_bp_result_withpose/1.attfuse_sign_bp_xy_2023_08_15_16/npy'
    # save_npy_path= '/home/jinlong/Desktop/1.attfuse_sign_bp_result_withpose/1.attfuse_sign_bp_xyz_2023_08_15_16/npy'
    # save_npy_path= '/home/jinlong/Desktop/1.attfuse_sign_bp_result_withpose/1.attfuse_sign_bp_ryp_2023_08_15_16/npy'

    # save_npy_path= ''
    # save_npy_path= '/home/jinlong/Desktop/Attack_vis/7.CoBEVT_AdvGPS_2023_09_11_08/npy'
    # save_npy_path= '/home/jinlong/Desktop/Attack_vis/7.CoBEVT_FSGM_2023_09_11_08/npy'
    # save_npy_path= '/home/jinlong/Desktop/Attack_vis/7.CoBEVT_IFSGM_2023_09_11_08/npy'
    save_npy_path= '/home/jinlong/Desktop/Attack_vis/7.CoBEVT_no_attack_2023_09_11_08/npy'
    # save_npy_path= '/home/jinlong/Desktop/Attack_vis/7.CoBEVT_PAA_2023_09_11_08/npy'
    # save_npy_path= '/home/jinlong/Desktop/Attack_vis/7.CoBEVT_PGD_2023_09_11_08/npy'
    # save_npy_path= '/home/jinlong/Desktop/Attack_vis/7.CoBEVT_random_2023_09_11_08/npy'


    pred_list = sorted([os.path.join(save_npy_path, x)
                        for x in os.listdir(save_npy_path) if 'pred' in x])
    gt_list = sorted([os.path.join(save_npy_path, x)
                      for x in os.listdir(save_npy_path) if 'gt' in x])
    
    pcd_list = sorted([os.path.join(save_npy_path, x)
                       for x in os.listdir(save_npy_path) if 'attacked' in x])
    # pcd_list = sorted([os.path.join(save_npy_path, x)
    #                    for x in os.listdir(save_npy_path) if 'pcd' in x])

    pred_tensor_list = [np.load(x) for x in pred_list]
    gt_tensor_list = [np.load(x) for x in gt_list]
    pcd_tensor_list = [np.load(x) for x in pcd_list]

    # visualize_sequence_sample_output(pred_tensor_list,
    #                                  gt_tensor_list,
    #                                  pcd_tensor_list)

    visualize_sequence_sample_output_1(pred_tensor_list,
                                     gt_tensor_list,
                                     pcd_tensor_list)