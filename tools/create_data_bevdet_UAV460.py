# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter as nuscenes_converter
from tools.data_converter import coUAV_converter as coUAV_converter

# map_name_from_general_to_detection = {
#     'human.pedestrian.adult': 'pedestrian',
#     'human.pedestrian.child': 'pedestrian',
#     'human.pedestrian.wheelchair': 'ignore',
#     'human.pedestrian.stroller': 'ignore',
#     'human.pedestrian.personal_mobility': 'ignore',
#     'human.pedestrian.police_officer': 'pedestrian',
#     'human.pedestrian.construction_worker': 'pedestrian',
#     'animal': 'ignore',
#     'vehicle.car': 'car',
#     'vehicle.motorcycle': 'motorcycle',
#     'vehicle.bicycle': 'bicycle',
#     'vehicle.bus.bendy': 'bus',
#     'vehicle.bus.rigid': 'bus',
#     'vehicle.truck': 'truck',
#     'vehicle.construction': 'construction_vehicle',
#     'vehicle.emergency.ambulance': 'ignore',
#     'vehicle.emergency.police': 'ignore',
#     'vehicle.trailer': 'trailer',
#     'movable_object.barrier': 'barrier',
#     'movable_object.trafficcone': 'traffic_cone',
#     'movable_object.pushable_pullable': 'ignore',
#     'movable_object.debris': 'ignore',
#     'static_object.bicycle_rack': 'ignore',
# }
# classes = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]
map_name_from_general_to_detection = {
    'vehicle.jeep.wrangler_rubicon': 'car',
    'vehicle.audi.etron': 'car',
    'vehicle.bmw.grandtourer': 'car',
    'vehicle.chargercop2020.chargercop2020': 'car',
    'vehicle.citroen.c3': 'car',
    'vehicle.toyota.prius': 'car',
    'vehicle.lincoln.mkz2017': 'car',
    'vehicle.lincoln2020.mkz2020': 'car',
    'vehicle.nissan.patrol': 'car',
    'vehicle.seat.leon': 'car',
    'vehicle.mercedes-benz.coupe': 'car',
    'vehicle.nissan.micra': 'car',
    'vehicle.tesla.model3': 'car',
    'vehicle.audi.tt': 'car',
    'vehicle.audi.a2': 'car',
    'vehicle.mustang.mustang': 'car',
    'vehicle.mini.cooperst': 'car',
    'vehicle.dodge_charger.police': 'car',
    'vehicle.mercedesccc.mercedesccc': 'car',
    'vehicle.charger2020.charger2020': 'car',
    'vehicle.chevrolet.impala': 'car',
    'static vehicle': 'car',
}

classes = [
    'car'
]
def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_BACK_id_2']['ego2global_rotation']  # %%% 本来是CAM_FRONT
    ego2global_translation = info['cams']['CAM_BACK_id_2'][  # %%%
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        # if (map_name_from_general_to_detection[ann_info['category_name']]
        #         not in classes
        #         or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):  #我觉得全注释掉吧都生成info
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

def coUAV_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    coUAV_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

def add_ann_adj_info(extra_tag):
    nuscenes_version = 'v60-trainval'  # %%%
    dataroot = '/workspace/data/coperception_uav_town4'  # %%%
    nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in ['train', 'val']:
        dataset = pickle.load(
            open('/workspace/data/coperception_uav_town4/%s_infos_%s.pkl' % (extra_tag, set), 'rb'))  # %%%
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']

            scene = nuscenes.get('scene', sample['scene_token'])
            dataset['infos'][id]['occ_path'] = \
                '/workspace/data/coperception_uav_town4/gts/%s/%s'%(scene['name'], info['token'])  # %%%
        with open('/workspace/data/coperception_uav_town4/%s_infos_%s.pkl' % (extra_tag, set),  # %%%
                  'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == '__main__':
    dataset = 'coperception_uav_town4'
    version = 'v60'
    train_version = f'{version}-trainval'  # %%%
    root_path = '/workspace/data/coperception_uav_town4'  # %%%
    extra_tag = 'bevdetv2-UAV460'  # %%%
    coUAV_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=train_version,
        max_sweeps=0)

    print('add_ann_infos')
    add_ann_adj_info(extra_tag)
