"""
Convert raw v2v real format to opv2v/kitti
"""
import math
import os
import re
import shutil
import statistics
from datetime import datetime

import numpy as np
import open3d as o3d

from opencood.hypes_yaml.yaml_utils import save_yaml

STEP = 1
KITTI = True
MIN_SCENE_LEN = 100
MAX_TIMESTAMP_GAP = 150
MAX_MIN_DIFF = 20
MAX_MATCHED_INDEX = 50


def load_timestamp(file_name):
    """
    Load timestamps to list.

    Parameters
    ----------
    file_name : str

    Returns
    -------
    list of timestamps in the current scenario.
    """
    with open(file_name) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    timestamps = []
    for line in lines:
        timestamps.append(datetime.strptime(line,
                                            "%Y-%m-%d %H:%M:%S.%f").timestamp()
                          * 1000)
    final_timestamps = []
    final_index = []

    tmp = []
    tmp_index = []
    for (i, timestamp) in enumerate(timestamps):
        if i == 0:
            tmp.append(timestamp)
            tmp_index.append(i)
            continue

        if timestamp - timestamps[i - 1] > MAX_TIMESTAMP_GAP:
            if len(tmp) > MIN_SCENE_LEN:
                final_timestamps.append(tmp)
                final_index.append(tmp_index)
            tmp = []
            tmp_index = []
        else:
            tmp.append(timestamp)
            tmp_index.append(i)

    return final_timestamps, final_index


def match_timestamp(astuff, tesla, pair_check=False):
    """
    Try to find the correct match up timestamp between the two vehicle.
    E.g., astuff 3 -> tesla 2.

    Parameters
    ----------
    astuff: list
        The list of timestamp in current scenario.
    tesla: list
        The list of timestamp in current scenario.
    pair_check: bool
        Whether to do pair check.

    Returns
    -------
    astuff_index: list
        The corrected index for astuff.
    tesla: list
        The corrected index for tesla.
    """
    min_diff_1 = 9999999
    min_diff_2 = 9999999
    astuff_index = 0
    tesla_index = 0

    for i in range(len(astuff)):
        diff = abs(tesla[0] - astuff[i])
        if diff < min_diff_1:
            min_diff_1 = diff
            astuff_index = i
        else:
            break

    for i in range(len(tesla)):
        diff = abs(astuff[0] - tesla[i])
        if diff < min_diff_2:
            min_diff_2 = diff
            tesla_index = i
        else:
            break

    if min_diff_1 > min_diff_2:
        return tesla_index, 'tesla', min(min_diff_1, min_diff_2)
    else:
        return astuff_index, 'astuff', min(min_diff_1, min_diff_2)


def match_timestamp_list(tesla_timestamps, astuff_timestamps):
    """
    Match two timestamp list  and return the pairs.
    """
    # list of (tesla list index, astuff list index)
    matched_list = []

    for (i, tesla_timestamp) in enumerate(tesla_timestamps):
        for (j, astuff_timestamp) in enumerate(astuff_timestamps):
            matched_index, _, min_diff = \
                match_timestamp(astuff_timestamp, tesla_timestamp, True)
            if min_diff < MAX_MIN_DIFF and matched_index < MAX_MATCHED_INDEX:
                matched_list.append((i, j))
                break
    return matched_list


def reindex_files(match_index,
                  match_cav,
                  astuff_timestamp,
                  tesla_timestamp,
                  astuff_index,
                  tesla_index,
                  astuff_tf_file_list,
                  astuff_lidar_file_list,
                  tesla_tf_file_list,
                  tesla_lidar_file_list):
    """
    Reindex the files to align with the timestamps.
    """
    if match_cav == 'astuff':
        start_index = astuff_index[match_index]
        end_index = astuff_index[-1]
        astuff_timestamp = astuff_timestamp[match_index:]

        astuff_lidar_file_list = astuff_lidar_file_list[
                                 start_index:end_index + 1]
        astuff_tf_file_list = astuff_tf_file_list[start_index:end_index + 1]

        tesla_lidar_file_list = tesla_lidar_file_list[tesla_index[0]:
                                                      tesla_index[-1] + 1]
        tesla_tf_file_list = tesla_tf_file_list[tesla_index[0]:
                                                tesla_index[-1] + 1]

    else:
        start_index = tesla_index[match_index]
        end_index = tesla_index[-1]
        tesla_timestamp = tesla_timestamp[match_index:]

        tesla_tf_file_list = tesla_tf_file_list[start_index:end_index + 1]
        tesla_lidar_file_list = tesla_lidar_file_list[
                                start_index:end_index + 1]

        astuff_lidar_file_list = astuff_lidar_file_list[astuff_index[0]:
                                                        astuff_index[-1] + 1]
        astuff_tf_file_list = astuff_tf_file_list[astuff_index[0]:
                                                  astuff_index[-1] + 1]

    min_len = min(len(astuff_tf_file_list), len(tesla_tf_file_list))
    astuff_tf_file_list = astuff_tf_file_list[:min_len]
    astuff_lidar_file_list = astuff_lidar_file_list[:min_len]
    tesla_tf_file_list = tesla_tf_file_list[:min_len]
    tesla_lidar_file_list = tesla_lidar_file_list[:min_len]

    astuff_timestamp = astuff_timestamp[:min_len]
    tesla_timestamp = tesla_timestamp[:min_len]
    tmp = []
    for i in range(len(astuff_timestamp)):
        time_diff = abs(astuff_timestamp[i] - tesla_timestamp[i])
        tmp.append(time_diff)
    if statistics.mean(tmp) > 50:
        return [], [], [], []
    return astuff_tf_file_list, astuff_lidar_file_list, \
           tesla_tf_file_list, tesla_lidar_file_list


def write_to_pcd(output_path, agent_num, file_index, bin_file):
    """
    Write bin pcd to o3d file format.

    Parameters
    ----------
    output_path : str
        Output folder
    agent_num : int
        Agent id.
    file_index : int
        The index for this snashot.
    bin_file : str
        Pointcloud bin file..
    """
    bin_pcd = np.fromfile(bin_file,
                          dtype=np.float32)
    # reshape
    points = bin_pcd.reshape((-1, 4))

    output_path_agent = os.path.join(output_path, str(agent_num))
    if not os.path.exists(output_path_agent):
        os.makedirs(output_path_agent)

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(points[:, :-1])

    point_intensity = np.zeros_like(points[:, :-1])
    point_intensity[:, 0] = points[:, -1] / 255.
    o3d_pcd.colors = o3d.utility.Vector3dVector(point_intensity)

    # write to pcd file
    pcd_name = '%06d' % (file_index // STEP) + '.pcd'
    o3d.io.write_point_cloud(os.path.join(output_path_agent,
                                          pcd_name),
                             pointcloud=o3d_pcd,
                             write_ascii=True)


def write_to_yaml(output_path, agent_num, file_index, tf_file, prev_x, prev_y):
    tf_matrix = np.loadtxt(tf_file, delimiter=',')
    # todo: hardcode to finetune
    # if agent_num == 1:
    #     tf_matrix[2, -1] -= 1.7
    yaml_dict = dict()

    output_path_agent = os.path.join(output_path, str(agent_num))
    if not os.path.exists(output_path_agent):
        os.makedirs(output_path_agent)

    yaml_dict['lidar_pose'] = tf_matrix
    yaml_dict['true_ego_pos'] = tf_matrix

    # calculate the speed
    if len(prev_x) == 0:
        ego_speed = 0
        prev_x.append(tf_matrix[0, -1])
        prev_y.append(tf_matrix[1, -1])
    else:
        x_offset = abs(tf_matrix[0, -1] - prev_x[-1])
        y_offset = abs(tf_matrix[1, -1] - prev_y[-1])
        prev_x.append(tf_matrix[0, -1])
        prev_y.append(tf_matrix[1, -1])

        offset = math.sqrt(x_offset ** 2 + y_offset ** 2)
        ego_speed = offset

    yaml_dict['ego_speed'] = ego_speed

    # mock the objects
    mocked_vehicle = {23: {
        'angle': [0, 0, 0],
        'center': [0, 0, 0.7188605070114136],
        'extent': [2.45, 1.06, 0.75],
        'location': [0, 0, 0]
    }}

    yaml_dict['vehicles'] = mocked_vehicle
    yaml_name = '%06d' % (file_index // STEP) + '.yaml'

    save_yaml(yaml_dict, os.path.join(output_path_agent, yaml_name))


def main():
    data_path = '/media/runshengxu/easystore/2nd_deployment/Day19_kitti'
    output_path = '/media/runshengxu/easystore/raw2kitti_new/Day19/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    scenario_list = [os.path.join(data_path, x) for x in os.listdir(data_path)]

    for scenario in scenario_list:
        print(scenario)
        scene_name = 'test'+os.path.basename(os.path.normpath(scenario))
        output_scene = os.path.join(output_path, scene_name)

        astuff_timstamps, astuff_indexes = \
            load_timestamp(os.path.join(scenario,
                                        'Astuff_GPS/timestamps.txt'))

        tesla_timstamps, tesla_indexes = \
            load_timestamp(os.path.join(scenario,
                                        'Tesla_GPS/timestamps.txt'))
        matched_list = match_timestamp_list(tesla_timstamps,
                                            astuff_timstamps)
        print('matched list length: %d' % len(matched_list))

        tesla_timstamps = [tesla_timstamps[x[0]] for x in matched_list]
        tesla_indexes = [tesla_indexes[x[0]] for x in matched_list]
        astuff_timstamps = [astuff_timstamps[x[1]] for x in matched_list]
        astuff_indexes = [astuff_indexes[x[1]] for x in matched_list]

        for k, (tesla_timstamp, tesla_index, astuff_timstamp, astuff_index) in \
                enumerate(zip(tesla_timstamps, tesla_indexes, astuff_timstamps,
                              astuff_indexes)):
            match_index, match_cav, _ = match_timestamp(astuff_timstamp,
                                                        tesla_timstamp)

            # load tf/lidar file list
            astuff_gps_path = os.path.join(scenario, 'Astuff_GPS/tf')
            astuff_lidar_path = os.path.join(scenario, 'Astuff_lidar/data')

            astuff_tf_file_list = \
                [os.path.join(astuff_gps_path, x) for x in
                 os.listdir(astuff_gps_path)]
            astuff_tf_file_list.sort(key=lambda f: int(re.sub('\D', '', f)))

            astuff_lidar_file_list = \
                [os.path.join(astuff_lidar_path, x) for x in
                 os.listdir(astuff_lidar_path)]
            astuff_lidar_file_list.sort(key=lambda f: int(re.sub('\D', '', f)))

            tesla_gps_path = os.path.join(scenario, 'Tesla_GPS/tf')
            tesla_lidar_path = os.path.join(scenario, 'Tesla_Lidar/data')
            tesla_tf_file_list = \
                [os.path.join(tesla_gps_path, x) for x in
                 os.listdir(tesla_gps_path)]
            tesla_lidar_file_list = \
                [os.path.join(tesla_lidar_path, x) for x in
                 os.listdir(tesla_lidar_path)]

            tesla_tf_file_list.sort(key=lambda f: int(re.sub('\D', '', f)))
            tesla_lidar_file_list.sort(key=lambda f: int(re.sub('\D', '', f)))

            # remove the indexes that are not aligned
            astuff_tf_file_list, astuff_lidar_file_list, \
            tesla_tf_file_list, tesla_lidar_file_list = reindex_files(
                match_index,
                match_cav,
                astuff_timstamp,
                tesla_timstamp,
                astuff_index,
                tesla_index,
                astuff_tf_file_list,
                astuff_lidar_file_list,
                tesla_tf_file_list,
                tesla_lidar_file_list)
            prev_x_tesla, prev_y_tesla, \
            prev_x_astuff, prev_y_astuff = [], [], [], []

            if len(astuff_tf_file_list) > 0:
                astuff_tf_tmp = astuff_tf_file_list[0]
                tesla_tf_tmp = tesla_tf_file_list[0]

                tf_matrix_0 = np.loadtxt(astuff_tf_tmp, delimiter=',')
                tf_matrix_1 = np.loadtxt(tesla_tf_tmp, delimiter=',')

                x_diff = abs(tf_matrix_0[0, -1] - tf_matrix_1[0, -1])
                y_diff = abs(tf_matrix_0[1, -1] - tf_matrix_1[1, -1])
                dist = math.sqrt(x_diff**2 + y_diff**2)
                if dist > 100:
                    print("too far away!")
                    continue


            if not KITTI:
                # load lidar file, write yaml file in a loop
                for i in range(0, len(tesla_tf_file_list), STEP):
                    # write to pcd
                    write_to_pcd(output_scene, 0, i, tesla_lidar_file_list[i])
                    write_to_pcd(output_scene, 1, i, astuff_lidar_file_list[i])

                    # write to yaml file
                    write_to_yaml(output_scene, 0, i, tesla_tf_file_list[i],
                                  prev_x_tesla, prev_y_tesla)
                    write_to_yaml(output_scene, 1, i, astuff_tf_file_list[i],
                                  prev_x_astuff, prev_y_astuff)
            else:
                for i in range(0, len(tesla_tf_file_list), STEP):
                    output_folder = os.path.join(output_path,
                                                 scene_name + '_%d_tesla' % k
                                                 )
                    lidar_folder = os.path.join(output_folder, 'lidar')
                    tf_folder = os.path.join(output_folder, 'tf')
                    label_folder = os.path.join(output_folder, 'label')

                    if not os.path.exists(lidar_folder):
                        os.makedirs(lidar_folder)
                        os.makedirs(tf_folder)
                    if not os.path.exists(label_folder):
                        os.makedirs(label_folder)

                    shutil.copyfile(tesla_lidar_file_list[i],
                                    os.path.join(lidar_folder,
                                                 '%06d.bin' % (i // STEP)))
                    shutil.copyfile(tesla_tf_file_list[i],
                                    os.path.join(tf_folder,
                                                 '%06d.txt' % (i // STEP)))

                for i in range(0, len(astuff_tf_file_list), STEP):
                    output_folder = os.path.join(output_path,
                                                 scene_name + '_%d_astuff' % k
                                                 )
                    lidar_folder = os.path.join(output_folder, 'lidar')
                    tf_folder = os.path.join(output_folder, 'tf')
                    label_folder = os.path.join(output_folder, 'label')

                    if not os.path.exists(lidar_folder):
                        os.makedirs(lidar_folder)
                        os.makedirs(tf_folder)
                        os.makedirs(label_folder)

                    shutil.copy2(astuff_lidar_file_list[i],
                                 os.path.join(lidar_folder,
                                              '%06d.bin' % (i // STEP)))
                    shutil.copy2(astuff_tf_file_list[i],
                                 os.path.join(tf_folder,
                                              '%06d.txt' % (i // STEP)))


if __name__ == '__main__':
    main()
