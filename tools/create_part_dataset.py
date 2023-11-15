'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-06-25 19:41:35
Email: haimingzhang@link.cuhk.edu.cn
Description: Create the part of the dataset samples to accelerate the
training process.
'''

import pickle
from collections import defaultdict
from tqdm import tqdm
import os.path as osp
import numpy as np
import mmcv


def load_mmdetection3d_infos(nuscenes_info_pickle):
    with open(nuscenes_info_pickle, "rb") as fp:
        nuscenes_infos = pickle.load(fp)

    print(type(nuscenes_infos))
    print(nuscenes_infos.keys())

    return nuscenes_infos

def get_scene_sequence_data(data_infos):
    scene_name_list = []
    total_scene_seq = []
    curr_seq = []
    for idx, data in enumerate(data_infos):
        scene_token = data['scene_token']
        next_idx = min(idx + 1, len(data_infos) - 1)
        next_scene_token = data_infos[next_idx]['scene_token']

        curr_seq.append(data)

        if next_scene_token != scene_token:
            total_scene_seq.append(curr_seq)
            scene_name_list.append(scene_token)
            curr_seq = []

    total_scene_seq.append(curr_seq)
    scene_name_list.append(scene_token)
    return scene_name_list, total_scene_seq


def create_part_pickle(pickle_path, ratio=0.5):
    nuscenes_infos = load_mmdetection3d_infos(pickle_path)

    metadata = nuscenes_infos['metadata']
    infos = nuscenes_infos['infos']

    infos = list(sorted(infos, key=lambda e: e['timestamp']))


    scene_name_list, total_scene_seq = get_scene_sequence_data(infos)

    new_infos = []
    for idx, (scene_name, scene_seq) in tqdm(enumerate(zip(scene_name_list, total_scene_seq)),
                                             total=len(scene_name_list)):
        keep_length = int(len(scene_seq) * ratio)
        new_infos.extend(scene_seq[:keep_length])

    print(len(new_infos))
    data = dict(infos=new_infos, metadata=metadata)

    filename, ext = osp.splitext(osp.basename(pickle_path))
    root_path = "./data/nuscenes"
    new_filename = f"{filename}-quarter{ext}"
    info_path = osp.join(root_path, new_filename)
    print(f"The results would be saved into {info_path}")
    mmcv.dump(data, info_path)
    

if __name__ == "__main__":
    nuscenes_info_pickle = "data/nuscenes/bevdetv2-nuscenes_infos_train.pkl"
    create_part_pickle(nuscenes_info_pickle, ratio=0.25)