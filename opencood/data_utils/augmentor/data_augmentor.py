# -*- coding: utf-8 -*-
"""
Class for data augmentation
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from functools import partial

import numpy as np

from opencood.data_utils.augmentor import augment_utils
from opencood.data_utils.augmentor import database_sampler
import random

from opencood.utils.box_utils import get_projection_length_for_vector_projection, boxes_to_corners_3d

class DataAugmentor(object):
    """
    Data Augmentor.

    Parameters
    ----------
    augment_config : list
        A list of augmentation configuration.

    Attributes
    ----------
    data_augmentor_queue : list
        The list of data augmented functions.
    """

    def __init__(self, augment_config, train=True, data_dir=None, class_names=None):
        self.data_augmentor_queue = []
        self.train = train
        self.data_dir = data_dir
        self.class_names = class_names
        self.random_seed = None

        if augment_config:
            for cur_cfg in augment_config:
                cur_augmentor = getattr(self, cur_cfg['NAME'])(config=cur_cfg)
                self.data_augmentor_queue.append(cur_augmentor)
                print(cur_cfg['NAME'])

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.data_dir,
            sampler_cfg=config,
            class_names=self.class_names
        )
        return db_sampler

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                    data_dict['object_bbx_mask'], \
                                    data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes_valid, points = getattr(augment_utils,
                                             'random_flip_along_%s' % cur_axis)(
                gt_boxes_valid, points,
            )

        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)

        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                    data_dict['object_bbx_mask'], \
                                    data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]
        gt_boxes_valid, points = augment_utils.global_rotation(
            gt_boxes_valid, points, rot_range=rot_range
        )
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)

        gt_boxes, gt_mask, points = data_dict['object_bbx_center'], \
                                    data_dict['object_bbx_mask'], \
                                    data_dict['lidar_np']
        gt_boxes_valid = gt_boxes[gt_mask == 1]

        gt_boxes_valid, points = augment_utils.global_scaling(
            gt_boxes_valid, points, config['WORLD_SCALE_RANGE']
        )
        gt_boxes[:gt_boxes_valid.shape[0], :] = gt_boxes_valid

        data_dict['object_bbx_center'] = gt_boxes
        data_dict['object_bbx_mask'] = gt_mask
        data_dict['lidar_np'] = points

        return data_dict              



    def forward(self, data_dict, choice=None):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...
            Choice:
                choose specific augmentation
        Returns:
        """
        if self.train:
            if choice == None:
                choice = 0
            for cur_augmentor in  self.data_augmentor_queue[choice:]:
                #for apply same augmentation
                if self.random_seed:
                    np.random.seed(self.random_seed)
                data_dict = cur_augmentor(data_dict=data_dict)
            self.random_seed = None
        return data_dict

