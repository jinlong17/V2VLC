"""
Remove the lidar points from the opposite direction in highway.
"""
import os
import numpy as np


folder_name = '/media/runshengxu/easystore/raw2kitti_new/Day19/testoutput_CAV_data_2022-03-15-10-09-50_2_tesla/lidar_bak'
output_folder = '/media/runshengxu/easystore/raw2kitti_new/Day19/testoutput_CAV_data_2022-03-15-10-09-50_2_tesla/lidar'
lidar_files = sorted([os.path.join(folder_name, x) for x in os.listdir(folder_name)])

for i,file in enumerate(lidar_files):
    # if i>=840 and i<=1141:
    #     continue
    bin_pcd = np.fromfile(file,
                          dtype=np.float32)
    # reshape
    points = bin_pcd.reshape((-1, 4))
    # filtered points
    index1 = points[:, 1] < 14
    # index2 = points[:, 0] < 50
    # index = np.logical_not(np.logical_and(index1, index2))
    index = index1
    points = points[index]

    basename = os.path.basename(file)
    print(basename)
    points.tofile(os.path.join(output_folder, basename))


