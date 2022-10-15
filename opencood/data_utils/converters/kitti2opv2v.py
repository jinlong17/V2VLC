"""
Convert processed kitti to opv2v
"""
import os
import numpy as np
import open3d as o3d

from shutil import copy


def write_to_pcd(output_path, agent_num, file_index, bin_file):
    """
    Write bin pcd to o3d file format.

    Parameters
    ----------
    output_path : str
        Output folder
    agent_num : int
        Agent id.
    file_index : str
        The index for this snashot.
    bin_file : str
        Pointcloud bin file..
    """
    bin_pcd = np.fromfile(bin_file,
                          dtype=np.float32)
    # reshape
    points = bin_pcd.reshape((-1, 4))

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points[:, :-1])

    point_intensity = np.zeros_like(points[:, :-1])
    point_intensity[:, 0] = points[:, -1] / 255.
    o3d_pcd.colors = o3d.utility.Vector3dVector(point_intensity)

    # write to pcd file
    pcd_name = file_index + '.pcd'
    o3d.io.write_point_cloud(os.path.join(output_path,
                                          pcd_name),
                             pointcloud=o3d_pcd,
                             write_ascii=True)


def main():
    data_path = '/media/runshengxu/easystore/kitti_intermediate/Day20'
    output_path = '/media/runshengxu/easystore/kitti2opv2v/Day20'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    scenario_list = sorted([os.path.join(data_path, x) for x in os.listdir(data_path)])

    for scenario in scenario_list:
        print(scenario)
        scene_name = os.path.basename(os.path.normpath(scenario))
        output_scene = os.path.join(output_path, scene_name)

        # tesla files
        tesla_lidar_path = os.path.join(scenario, 'tesla/lidar')
        tesla_label_path = os.path.join(scenario, 'tesla/label_opv2v')
        tesla_label_list = \
            sorted([os.path.join(tesla_label_path, x) for x in
                    os.listdir(tesla_label_path)])
        tesla_lidar_list = \
            sorted([os.path.join(tesla_lidar_path, x) for x in
                    os.listdir(tesla_lidar_path)])

        # astuff files
        astuff_lidar_path = os.path.join(scenario, 'astuff/lidar')
        astuff_label_path = os.path.join(scenario, 'astuff/label_opv2v')
        astuff_label_list = \
            sorted([os.path.join(astuff_label_path, x) for x in
                    os.listdir(astuff_label_path)])
        astuff_lidar_list = \
            sorted([os.path.join(astuff_lidar_path, x) for x in
                    os.listdir(astuff_lidar_path)])

        assert len(astuff_lidar_list) == len(tesla_lidar_list)
        assert len(astuff_label_list) == len(tesla_label_list)

        tesla_output_folder = os.path.join(output_path, output_scene,
                                           '0')
        astuff_output_folder = os.path.join(output_path, output_scene,
                                            '1')
        if not os.path.exists(tesla_output_folder):
            os.makedirs(tesla_output_folder)
            os.makedirs(astuff_output_folder)
        else:
            continue

        for i in range(len(tesla_label_list)):
            tesla_label_file = tesla_label_list[i]
            tesla_lidar_file = tesla_lidar_list[i]
            index_name = os.path.basename(tesla_lidar_file)[:-4]

            # copy label to the correct folder
            copy(tesla_label_file, tesla_output_folder)
            # write pcd file
            write_to_pcd(tesla_output_folder, 0, index_name, tesla_lidar_file)

            astuff_label_file = astuff_label_list[i]
            astuff_lidar_file = astuff_lidar_list[i]
            index_name = os.path.basename(astuff_lidar_file)[:-4]

            # copy label to the correct folder
            copy(astuff_label_file, astuff_output_folder)
            # write pcd file
            write_to_pcd(astuff_output_folder, 0, index_name, astuff_lidar_file)

if __name__ == '__main__':
    main()
