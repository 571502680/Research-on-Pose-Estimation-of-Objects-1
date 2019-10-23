import os
import torch
import json
from torchvision import transforms

opj = os.path.join
ROOT = '/home/liqi/PycharmProjects/kp6d/detect'


def prepare_weight(ckpt):
    """
    Args
    - ckpt: (str) Relative path to weight file
    """
    weightfile = opj(ROOT, 'darknet/backup', ckpt)
    return weightfile


def prepare_cfg(name):
    """
    Prepare configuration file path

    Args
    - name: (str) Dataset name

    Return
    - cfgfile: (str) Configuration file path
    """
    if name == 'linemod-single':
        return opj(ROOT, 'darknet/cfg/linemod-single.cfg')
    elif name == 'linemod-occ':
        return opj(ROOT, 'darknet/cfg/linemod-occ.cfg')
    elif name == 'ycb':
        return opj(ROOT, 'darknet/cfg/ycb.cfg')
    else:
        raise NotImplementedError(name)


def class_name(dataset, idx):
    """
    Args
    - dataset: (str) Dataset name
    - idx: (int or string) Class index
    """
    LINEMOD = ('ape', 'bvise', 'bowl', 'camera', 'can', 'cat', 'cup', 'driller',
               'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone')
    YCB = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can'
           '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box',
           '010_potted_meat_can', '011_banana', '019_pitcher_base', '021_bleach_cleanser',
           '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors',
           '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
    if 'linemod' in dataset:
        return LINEMOD[int(idx) - 1]
    elif 'ycb' in dataset:
        return YCB[int(idx) - 1]
    else:
        raise NotImplementedError(dataset)


def class_idx(dataset, name):
    LINEMOD = ('ape', 'bvise', 'bowl', 'camera', 'can', 'cat', 'cup', 'driller',
               'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone')
    OCCLINEMOD = ('ape', 'can', 'cat', 'driller', 'duck',
                  'eggbox', 'glue', 'holepuncher')
    YCB = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can'
           '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box',
           '010_potted_meat_can', '011_banana', '019_pitcher_base', '021_bleach_cleanser',
           '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors',
           '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
    if dataset == 'linemod-single':
        return LINEMOD.index(name)
    elif dataset == 'linemod-occ':
        return OCCLINEMOD.index(name)
    elif dataset == 'ycb':
        return YCB.index(name)
    else:
        raise NotImplementedError(dataset)
