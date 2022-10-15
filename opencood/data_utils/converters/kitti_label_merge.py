# merge two labeled folders into one
import os
from shutil import copy, copytree

if __name__ == '__main__':
    data_path = '/media/runshengxu/easystore/raw2kitt_label_rawformat/Day20'
    output_path = '/media/runshengxu/easystore/kitti_intermediate/Day20'
    scenario_list = [os.path.join(data_path, x) for x in os.listdir(data_path)]

    for scenario in scenario_list:
        if scenario[-1] == 'f':
            base_name = scenario[:-7].split('/')[-1]
            car = "astuff"
        else:
            base_name = scenario[:-6].split('/')[-1]
            car = "tesla"

        if not os.path.exists(os.path.join(output_path, base_name, car)):
            os.makedirs(os.path.join(output_path, base_name, car))

        folders = [os.path.join(scenario, x) for x in os.listdir(scenario)]
        for folder in folders:
            folder_name = folder.split('/')[-1]
            copytree(folder, os.path.join(output_path, base_name, car, folder_name))

        print(scenario)
