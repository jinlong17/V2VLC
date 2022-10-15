"""
Convert opv2v data to kitti dataset.
"""
import argparse
import os
import numpy as np

from torch.utils.data import DataLoader

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import common_utils
from opencood.data_utils.datasets.early_fusion_vis_dataset import \
    EarlyFusionVisDataset


def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    output_root = '/media/runshengxu/easystore/output_kitti'

    params = load_yaml(os.path.join(current_path,
                                    '../../hypes_yaml/visualization.yaml'))


    opencood_dataset = EarlyFusionVisDataset(params, visualize=True,
                                            train=False)
    data_loader = DataLoader(opencood_dataset, batch_size=1, num_workers=1,
                             collate_fn=opencood_dataset.collate_batch_train,
                             shuffle=False,
                             pin_memory=False)

    for i_batch, sample_batched in enumerate(data_loader):
        origin_lidar = sample_batched['ego']['origin_lidar'][0]
        if not isinstance(origin_lidar, np.ndarray):
            origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)

        scene_name = sample_batched['ego']['folder_name']
        file_index = sample_batched['ego']['file_index']
        output_folder = os.path.join(output_root, scene_name, 'lidar')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        origin_lidar = origin_lidar.reshape((-1, 4))
        np.ndarray.tofile(origin_lidar, os.path.join(output_folder,
                                                     '%06d.bin' % file_index))


if __name__ == '__main__':
    main()
