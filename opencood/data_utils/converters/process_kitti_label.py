"""
Merge the object ids for Tesla and Astuff from the kitti label format
"""
import os
import math
import json

import statistics
import numpy as np

from easydict import EasyDict as edict
from opencood.utils.box_utils import boxes_to_corners_3d, \
    project_points_by_matrix_torch
from opencood.utils.common_utils import convert_format
from opencood.utils.transformation_utils import x1_to_x2
from opencood.hypes_yaml.yaml_utils import save_yaml


def json2dict(json_file, tf_file):
    """
    Load object json labels and covert it to dictionary.

    Parameters
    ----------
    json_file : str
    tf_file : str

    Returns
    -------
    object_dict: dictionary for the object.
    """
    f = open(json_file)
    object_dict = dict()
    object_list = json.load(f)
    tf_matrix = np.loadtxt(tf_file, delimiter=',')

    for object_meta in object_list:
        opv2v_info = edict()
        processed_info = edict()
        object_meta = edict(object_meta)
        object_meta.ass_id = -1

        # raw information
        obj_id = object_meta.obj_id
        obj_type = object_meta.obj_type
        location = object_meta.psr.position
        angle = object_meta.psr.rotation
        extent = object_meta.psr.scale

        # process the information to the opv2v format
        opv2v_info.location = [location.x, location.y, location.z]
        opv2v_info.angle = [math.degrees(angle.x),
                            math.degrees(angle.z),
                            math.degrees(angle.y)]
        opv2v_info.extent = [extent.x / 2, extent.y / 2, extent.z / 2]
        opv2v_info.center = [0, 0, 0]
        opv2v_info.obj_type = obj_type
        opv2v_info.ass_id = -1

        # convert to center format to get the corner format
        center = np.array([opv2v_info.location +
                           [extent.x, extent.y, extent.z] +
                           [angle.z]])
        # convert center to corner format
        corner3d = boxes_to_corners_3d(center, 'lwh')
        # find the 2d corner as well
        corner2d = corner3d[:, :4, :]
        processed_info.center = center
        processed_info.corner3d = corner3d
        processed_info.corner2d = corner2d
        processed_info.tf = tf_matrix

        object_dict.update({obj_id:
                                {'raw_info': object_meta,
                                 'opv2v_info': opv2v_info,
                                 'processed_info': processed_info,
                                 'overlap': False}})

    return object_dict


def merge_id(tesla, astuff, height_diff):
    """
    Give the same id to the objects that are in different vehicles' coordinate

    Parameters
    ----------
    height_diff : list
    tesla : dict
    astuff : dict
    """

    for obj_id_tesla, obj_content_tesla in tesla.items():
        box2d = obj_content_tesla['processed_info'].corner2d
        box3d = obj_content_tesla['processed_info'].corner3d
        box2d_poly = convert_format(box2d)[0]
        tf_tesla = obj_content_tesla['processed_info'].tf

        for obj_id_ast, obj_content_ast in astuff.items():
            if obj_content_ast['overlap']:
                continue

            box3d_ast = obj_content_ast['processed_info'].corner3d
            tf_ast = obj_content_ast['processed_info'].tf
            # astuff to tesla transformation matrix
            ast2tesla = x1_to_x2(tf_ast, tf_tesla)

            # transform to tesla coordinate
            box3d_ast = project_points_by_matrix_torch(
                box3d_ast.reshape(-1, 3),
                ast2tesla).reshape(1, 8, 3)
            box2d_ast = box3d_ast[:, :4, :]
            box2d_ast_poly = convert_format(box2d_ast)[0]

            obj_content_ast['processed_info'].corner3d_t = box3d_ast
            obj_content_ast['processed_info'].corner2d_t = box2d_ast

            iou = box2d_poly.intersection(box2d_ast_poly).area / \
                  box2d_poly.union(box2d_ast_poly).area

            if iou > 0.1:
                obj_content_ast['raw_info'].ass_id = obj_id_tesla
                obj_content_ast['processed_info'].obj_id = obj_id_tesla
                obj_content_ast['opv2v_info'].ass_id = int(obj_id_tesla)

                z_ast = np.mean(box3d_ast, axis=1)[0, -1]
                z_tes = np.mean(box3d, axis=1)[0, -1]
                height_diff.append(z_tes - z_ast)

                obj_content_ast['overlap'] = True
                obj_content_tesla['overlap'] = True
                break
    return height_diff


def write_kitti_json(output_file_name, obj_dict):
    """
    Write the processed kitti (id merged version) to disk.

    Parameters
    ----------
    output_file_name : str
    obj_dict : dict
    """

    output_list = []

    for _, obj_content in obj_dict.items():
        output_list.append(obj_content['raw_info'])

    with open(output_file_name, 'w') as fout:
        json.dump(output_list, fout, indent=2)


def write_opv2v_yaml(output_file_name, tf_file, obj_dict, height_diff=None):
    """
    Write label data to opv2v format.

    Parameters
    ----------
    height_diff : list
    tf_file : str
    output_file_name : str
    obj_dict : dict
    """
    tf_matrix = np.loadtxt(tf_file, delimiter=',')

    # cav information
    yaml_dict = dict()
    if height_diff is not None and len(height_diff) > 0:
        diff_mean = statistics.mean(height_diff)
        tf_matrix[2, -1] += diff_mean
    yaml_dict['lidar_pose'] = tf_matrix
    yaml_dict['true_ego_pos'] = tf_matrix
    yaml_dict['ego_speed'] = 0
    yaml_dict['vehicles'] = {}

    for obj_id, obj_content in obj_dict.items():
        veh_dict = dict(obj_content['opv2v_info'])
        yaml_dict['vehicles'].update({int(obj_id): veh_dict})

    save_yaml(yaml_dict, output_file_name)


if __name__ == '__main__':
    data_path = '/media/runshengxu/easystore/kitti_intermediate/Day20'
    scenario_list = [os.path.join(data_path, x) for x in os.listdir(data_path)]

    for scenario in scenario_list:
        print(scenario)
        tesla_folder = os.path.join(scenario, 'tesla')
        astuff_folder = os.path.join(scenario, 'astuff')

        tesla_label_folder = os.path.join(tesla_folder, 'label')
        tesla_lidar_folder = os.path.join(tesla_folder, 'lidar')
        tesla_tf_folder = os.path.join(tesla_folder, 'tf')
        tesla_label_processed_folder = os.path.join(tesla_folder,
                                                    'label_kitti')
        tesla_opv2v_folder = os.path.join(tesla_folder,
                                          'label_opv2v')

        astuff_label_folder = os.path.join(astuff_folder, 'label')
        astuff_lidar_folder = os.path.join(astuff_folder, 'lidar')
        astuff_tf_folder = os.path.join(astuff_folder, 'tf')
        astuff_label_processed_folder = os.path.join(astuff_folder,
                                                     'label_kitti')
        astuff_opv2v_folder = os.path.join(astuff_folder,
                                           'label_opv2v')

        if not os.path.exists(tesla_label_processed_folder):
            os.makedirs(tesla_label_processed_folder)
            os.makedirs(tesla_opv2v_folder)
            os.makedirs(astuff_label_processed_folder)
            os.makedirs(astuff_opv2v_folder)
        else:
            continue

        tesla_json_list = sorted([os.path.join(tesla_label_folder, x)
                            for x in os.listdir(tesla_label_folder)])
        tesla_tf_list = sorted([os.path.join(tesla_tf_folder, x)
                            for x in os.listdir(tesla_tf_folder)])
        tesla_lidar_list = sorted([os.path.join(tesla_lidar_folder, x)
                            for x in os.listdir(tesla_lidar_folder)])

        astuff_json_list = sorted([os.path.join(astuff_label_folder, x)
                            for x in os.listdir(astuff_label_folder)])
        astuff_tf_list = sorted([os.path.join(astuff_tf_folder, x)
                            for x in os.listdir(astuff_tf_folder)])
        astuff_lidar_list = sorted([os.path.join(astuff_lidar_folder, x)
                            for x in os.listdir(astuff_lidar_folder)])


        height_diff = []

        for i in range(len(tesla_json_list)):
            json_file = tesla_json_list[i]
            index_name = os.path.basename(json_file)[:-5]
            tf_file_tesla = tesla_tf_list[i]
            obj_dict_tesla = json2dict(json_file, tf_file_tesla)

            json_file = astuff_json_list[i]
            tf_file_astuff = astuff_tf_list[i]
            obj_dict_astuff = json2dict(json_file, tf_file_astuff)

            height_diff = merge_id(obj_dict_tesla, obj_dict_astuff,
                                   height_diff)

            # write to folder
            json_file = os.path.join(tesla_label_processed_folder,
                                     index_name+'.json')
            write_kitti_json(json_file, obj_dict_tesla)

            json_file = os.path.join(astuff_label_processed_folder,
                                     index_name+'.json')
            write_kitti_json(json_file, obj_dict_astuff)

            # write opv2v yaml
            yaml_file = os.path.join(tesla_opv2v_folder, index_name+'.yaml')
            write_opv2v_yaml(yaml_file, tf_file_tesla, obj_dict_tesla)

            yaml_file = os.path.join(astuff_opv2v_folder, index_name+'.yaml')
            write_opv2v_yaml(yaml_file, tf_file_astuff, obj_dict_astuff,
                             height_diff)

    # json_file = '/media/runshengxu/easystore/raw2kitti_labeled/day20/output_CAV_data_2022-03-17-10-43-13/Tesla/label/000001.json'
    # tf_file_tesla = '/media/runshengxu/easystore/raw2kitti_labeled/day20/output_CAV_data_2022-03-17-10-43-13/Tesla/tf/000001.txt'
    # obj_dict_tesla = json2dict(json_file, tf_file_tesla)
    #
    # json_file = '/media/runshengxu/easystore/raw2kitti_labeled/day20/output_CAV_data_2022-03-17-10-43-13/Astuff/label/000001.json'
    # tf_file_astuff = '/media/runshengxu/easystore/raw2kitti_labeled/day20/output_CAV_data_2022-03-17-10-43-13/Astuff/tf/000001.txt'
    # obj_dict_astuff = json2dict(json_file, tf_file_astuff)
    #
    # height_diff = merge_id(obj_dict_tesla, obj_dict_astuff, height_diff)
    # print('test')
    #
    # # write processed kitti label
    # json_file = '/media/runshengxu/easystore/raw2kitti_labeled/day20/output_CAV_data_2022-03-17-10-43-13/Astuff/label_kitti/000001.json'
    # write_kitti_json(json_file, obj_dict_astuff)
    # json_file = '/media/runshengxu/easystore/raw2kitti_labeled/day20/output_CAV_data_2022-03-17-10-43-13/Tesla/label_kitti/000001.json'
    # write_kitti_json(json_file, obj_dict_tesla)
    #
    # # write opv2v format label
    # yaml_file = '/media/runshengxu/easystore/raw2kitti_labeled/day20/output_CAV_data_2022-03-17-10-43-13/Tesla/label_opv2v/000001.yaml'
    # write_opv2v_yaml(yaml_file, tf_file_tesla, obj_dict_tesla)
    # yaml_file = '/media/runshengxu/easystore/raw2kitti_labeled/day20/output_CAV_data_2022-03-17-10-43-13/Astuff/label_opv2v/000001.yaml'
    # write_opv2v_yaml(yaml_file, tf_file_astuff, obj_dict_astuff, height_diff)
