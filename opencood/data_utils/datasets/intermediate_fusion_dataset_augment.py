# intermediate fusion dataset
import random
import math
from collections import OrderedDict
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.heter_utils import AgentSelector
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.common_utils import read_json

from opencood.utils.box_utils import boxes_to_corners_3d, get_points_mask_in_rotated_box_3d

def getIntermediateaugmentFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class IntermediateaugmentFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            # intermediate and supervise single
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            self.proj_first = False if 'proj_first' not in params['fusion']['args']\
                                         else params['fusion']['args']['proj_first']

            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.kd_flag = params.get('kd_flag', False)

        def gt_masking(self, gt_boxes, gt_mask, points):
            gt_boxes_valid = gt_boxes[gt_mask == 1]
            mask_ratio = 0.5 #config['MASK_RATIO']
            # mask part of the points inside the box
            
            masked_boxes_index = np.random.random(gt_boxes_valid.shape[0]) > mask_ratio
            if masked_boxes_index.sum() == 0:
                return points
            masked_boxes = gt_boxes_valid[masked_boxes_index]
            #keep_boxes = gt_boxes_valid[~masked_boxes_index]
            masked_boxes = boxes_to_corners_3d(masked_boxes, 'hwl')
            p_masked = np.zeros(points.shape[0])
            for masked_box in masked_boxes:
                mask = get_points_mask_in_rotated_box_3d(points[:,:3], masked_box)
                p_masked = np.logical_or(mask, p_masked)
            masked_points = points[~p_masked]
            
            #gt_boxes = np.zeros_like(gt_boxes)
            #gt_boxes[:keep_boxes.shape[0], :] = keep_boxes
            return masked_points

        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_pose : list, length 6
                The ego vehicle lidar pose under world coordinate.
            ego_pose_clean : list, length 6
                only used for gt box generation

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            # calculate the transformation matrix
            transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_pose) # T_ego_cav
            transformation_matrix_clean = x1_to_x2(selected_cav_base['params']['lidar_pose_clean'], ego_pose_clean)
            
            # note the reference pose ego
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base], ego_pose_clean)

            # lidar
            if self.load_lidar_file or self.visualize:
                # process lidar
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)
                # project the lidar to ego space
                # x,y,z in ego space
                projected_lidar = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                if self.proj_first:
                    lidar_np[:, :3] = projected_lidar

                # data augmentation for student
                if selected_cav_base['ego']:
                    #random_seed = np.random.randint(100000)
                    #object_bbx_center_o = copy.deepcopy(object_bbx_center)
                    #object_bbx_mask_o = copy.deepcopy(object_bbx_mask)
                    #student_sample_group = copy.deepcopy(self.data_augmentor.data_augmentor_queue[0].sample_groups) #keep the sample group of the student
                    lidar_np, object_bbx_center, object_bbx_mask = self.augment(lidar_np, object_bbx_center, object_bbx_mask)
                    projected_lidar = copy.deepcopy(lidar_np)[:,:3]
                    # self.data_augmentor.data_augmentor_queue[0].sample_groups = student_sample_group #useing the same sample group for teacher
                    # projected_lidar, _, _ = self.augment(projected_lidar, object_bbx_center_o, object_bbx_mask_o, random_seed)

                if self.visualize:
                    # filter lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                if self.kd_flag:
                    lidar_proj_np = copy.deepcopy(lidar_np)
                    lidar_proj_np[:,:3] = projected_lidar

                    selected_cav_processed.update({'projected_lidar': lidar_proj_np})

                processed_lidar = self.pre_processor.preprocess(lidar_np)
                selected_cav_processed.update({'processed_features': processed_lidar})

            # anchor box
            selected_cav_processed.update({"anchor_box": self.anchor_box})
            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                    'transformation_matrix': transformation_matrix,
                    'transformation_matrix_clean': transformation_matrix_clean
                }
            )


            return selected_cav_processed

        def get_item_all_agent(self, selected_cav_base, selected_inf_base, ego_cav_base):
            """
            Process all CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
            selected_inf_base : dict
                The dictionary contains a single CAV's raw information.
            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            selected_inf_processed : dict
                The dictionary contains the inf's processed information.
            """
            selected_cav_processed = {}
            selected_inf_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            # calculate the transformation matrix for cav
            transformation_matrix_cav = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_pose)
            transformation_matrix_clean_cav = x1_to_x2(selected_cav_base['params']['lidar_pose_clean'], ego_pose_clean)
            cav_bbx_center, cav_bbx_mask, cav_ids = self.generate_object_center([selected_cav_base], ego_pose_clean)
            # calculate the transformation matrix for inf
            transformation_matrix_inf = x1_to_x2(selected_inf_base['params']['lidar_pose'], ego_pose) 
            transformation_matrix_clean_inf = x1_to_x2(selected_inf_base['params']['lidar_pose_clean'], ego_pose_clean)
            inf_bbx_center, inf_bbx_mask, inf_ids = self.generate_object_center([selected_inf_base], ego_pose_clean)
            # lidar
            if self.load_lidar_file or self.visualize:
                # process lidar for cav
                lidar_np_cav = selected_cav_base['lidar_np']
                lidar_np_cav = shuffle_points(lidar_np_cav)
                lidar_np_cav = mask_ego_points(lidar_np_cav)# remove points that hit itself
                #projected_lidar_cav = box_utils.project_points_by_matrix_torch(lidar_np_cav[:, :3], transformation_matrix_cav)# project the lidar to ego space
                # process lidar for inf
                lidar_np_inf = selected_inf_base['lidar_np']
                lidar_np_inf = shuffle_points(lidar_np_inf)
                lidar_np_inf = mask_ego_points(lidar_np_inf)
                # data augmentation for both cav and inf
                random_seed = np.random.randint(100000) #for same random augmentation
                if len(self.data_augmentor.data_augmentor_queue)!=0:  
                    cav_sample_group = copy.deepcopy(self.data_augmentor.data_augmentor_queue[0].sample_groups) #keep the sample group of the cav
                    lidar_np_cav, cav_bbx_center, cav_bbx_mask = self.augment(lidar_np_cav, cav_bbx_center, cav_bbx_mask, random_seed) #choice=1 skip gt_sampling
                    self.data_augmentor.data_augmentor_queue[0].sample_groups = cav_sample_group #useing the same sample group for teacher
                    lidar_np_inf, inf_bbx_center, inf_bbx_mask = self.augment(lidar_np_inf, inf_bbx_center, inf_bbx_mask, random_seed) #choice=1 skip gt_sampling
                    
                projected_lidar_cav = box_utils.project_points_by_matrix_torch(lidar_np_cav[:, :3], transformation_matrix_cav)# project the lidar to ego space
                projected_lidar_inf = box_utils.project_points_by_matrix_torch(lidar_np_inf[:, :3], transformation_matrix_inf)# project the lidar to ego space

                if self.visualize:
                    selected_cav_processed.update({'projected_lidar': projected_lidar_cav})
                    selected_inf_processed.update({'projected_lidar': projected_lidar_inf})

                if self.kd_flag:
                    lidar_proj_np_cav = copy.deepcopy(lidar_np_cav)
                    lidar_proj_np_cav[:,:3] = projected_lidar_cav
                    selected_cav_processed.update({'projected_lidar': lidar_proj_np_cav})

                    lidar_proj_np_inf = copy.deepcopy(lidar_np_inf)
                    lidar_proj_np_inf[:,:3] = projected_lidar_inf
                    selected_inf_processed.update({'projected_lidar': lidar_proj_np_inf})

                # if self.mae and self.train:
                #     lidar_np_cav_masked = self.gt_masking(cav_bbx_center, cav_bbx_mask, lidar_np_cav)
                #     lidar_np_cav = lidar_np_cav_masked
                    # lidar_np_cav_masked = self.gt_masking(cav_bbx_center, cav_bbx_mask, lidar_np_cav)
                    # lidar_np_cav = lidar_np_cav_masked
                    # processed_lidar_cav_masked = self.pre_processor.preprocess(lidar_np_cav_masked)
                    # selected_cav_processed.update({'processed_features_masked': processed_lidar_cav_masked})
                processed_lidar_cav = self.pre_processor.preprocess(lidar_np_cav)
                processed_lidar_inf = self.pre_processor.preprocess(lidar_np_inf)
                selected_cav_processed.update({'processed_features': processed_lidar_cav})
                selected_inf_processed.update({'processed_features': processed_lidar_inf})

            selected_cav_processed.update(
                {
                    "object_bbx_center": cav_bbx_center[cav_bbx_mask == 1],
                    "object_bbx_mask": cav_bbx_mask,
                    "object_ids": cav_ids,
                    'transformation_matrix': transformation_matrix_cav,
                    'transformation_matrix_clean': transformation_matrix_clean_cav,
                    "anchor_box": self.anchor_box
                }
            )
            selected_inf_processed.update(
                {
                    "object_bbx_center": inf_bbx_center[inf_bbx_mask == 1],
                    "object_bbx_mask": inf_bbx_mask,
                    "object_ids": inf_ids,
                    'transformation_matrix': transformation_matrix_inf,
                    'transformation_matrix_clean': transformation_matrix_clean_inf,
                    "anchor_box": self.anchor_box
                }
            )

            return selected_cav_processed, selected_inf_processed


        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break
                
            assert cav_id == list(base_data_dict.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            agents_image_inputs = []
            processed_features = []
            object_stack = []
            object_id_stack = []
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            too_far = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []
            projected_lidar_clean_list = [] # disconet

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = math.sqrt((selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 + 
                              (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2)

                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    too_far.append(cav_id)
                    continue

                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                cav_id_list.append(cav_id)   

            for cav_id in too_far:
                base_data_dict.pop(cav_id)


            pairwise_t_matrix = get_pairwise_transformation(base_data_dict, self.max_cav, self.proj_first)

            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
            
            # merge preprocessed features from different cavs into the same dict
            cav_num = len(cav_id_list)

            if cav_num == 1:
                selected_cav_base = base_data_dict[0]
                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_cav_base)
                    
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                if self.load_lidar_file:
                    processed_features.append(selected_cav_processed['processed_features'])

                if self.visualize or self.kd_flag:
                    projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
            else:
                selected_cav_processed, selected_inf_processed = self.get_item_all_agent(base_data_dict[0], base_data_dict[1], ego_cav_base)           
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_stack.append(selected_inf_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                object_id_stack += selected_inf_processed['object_ids']
                if self.load_lidar_file:
                    processed_features.append(selected_cav_processed['processed_features'])
                    processed_features.append(selected_inf_processed['processed_features'])

                if self.visualize or self.kd_flag:
                    projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
                    projected_lidar_stack.append(selected_inf_processed['projected_lidar'])


            if self.kd_flag:
                stack_lidar_np_origin = np.vstack(projected_lidar_stack)
                stack_lidar_np = copy.deepcopy(stack_lidar_np_origin)
                stack_lidar_np = mask_points_by_range(stack_lidar_np, self.params['preprocess']['cav_lidar_range'])
                stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                processed_data_dict['ego'].update({'teacher_processed_lidar': stack_feature_processed})

            
            # exclude all repetitive objects    
            #unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            #object_stack = object_stack[unique_indices]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            
            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_features)
                processed_data_dict['ego'].update({'processed_lidar': merged_feature_dict})
            # generate targets label
            label_dict = self.post_processor.generate_label(gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=mask)

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [i for i in range(len(object_stack))] ,#[object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_poses_clean': lidar_poses_clean,
                'lidar_poses': lidar_poses})


            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar': np.vstack(projected_lidar_stack)})


            processed_data_dict['ego'].update({'sample_idx': idx, 'cav_id_list': cav_id_list})

            return processed_data_dict


        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            image_inputs_list = []
            # used to record different scenario
            record_len = []
            label_dict_list = []
            lidar_pose_list = []
            origin_lidar = []
            lidar_pose_clean_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            # disconet
            teacher_processed_lidar_list = []
            teacher_processed_global_lidar_list = []
            
            ### 2022.10.10 single gt ####
            if self.supervise_single:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []
                object_bbx_center_single = []
                object_bbx_mask_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
                lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
                
                record_len.append(ego_dict['cav_num'])
                label_dict_list.append(ego_dict['label_dict'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                if self.kd_flag:
                    teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])
                ### 2022.10.10 single gt ####
                if self.supervise_single:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])
                    object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
                    object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])


            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)
                processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(merged_feature_dict)
                output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

                output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
            
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                     'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # add pairwise_t_matrix to label dict
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict['record_len'] = record_len
            

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'record_len': record_len,
                                    'label_dict': label_torch_dict,
                                    'object_ids': object_ids[0],
                                    'pairwise_t_matrix': pairwise_t_matrix,
                                    'lidar_pose_clean': lidar_pose_clean,
                                    'lidar_pose': lidar_pose,
                                    'anchor_box': self.anchor_box_torch})


            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.kd_flag:
                teacher_processed_lidar_torch_dict = self.pre_processor.collate_batch(teacher_processed_lidar_list)
                output_dict['ego'].update({'teacher_processed_lidar':teacher_processed_lidar_torch_dict})

            if self.supervise_single:
                output_dict['ego'].update({
                    "label_dict_single":{
                            "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                            "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                            "targets": torch.cat(targets_single, dim=0),
                            # for centerpoint
                            "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                            "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                        },
                    "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                })


            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            output_dict = self.collate_batch_train(batch)
            if output_dict is None:
                return None

            # check if anchor box in the batch
            if batch[0]['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box':
                    self.anchor_box_torch})

            # save the transformation matrix (4, 4) to ego vehicle
            # transformation is only used in post process (no use.)
            # we all predict boxes in ego coord.
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict['ego'].update({'transformation_matrix':
                                        transformation_matrix_torch,
                                        'transformation_matrix_clean':
                                        transformation_matrix_clean_torch,})

            output_dict['ego'].update({
                "sample_idx": batch[0]['ego']['sample_idx'],
                "cav_id_list": batch[0]['ego']['cav_id_list']
            })

            return output_dict

        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict, output_dict)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor


    return IntermediateaugmentFusionDataset


