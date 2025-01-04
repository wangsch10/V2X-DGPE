import os
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.camera_utils import load_camera_data, load_intrinsic_DAIR_V2X
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose, rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor

import argparse
import opencood.hypes_yaml.yaml_utils as yaml_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from opencood.utils.transformation_utils import x1_to_x2, x_to_world
from opencood.utils.box_utils import corner_to_center

def project_world_objects_dairv2x(object_list, lidar_pose, order='lwh'):
    """
    Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_list : list
        The list contains all objects surrounding a certain cav.

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    lidar_pose : list
        (6, ), lidar pose under world coordinate, [x, y, z, roll, yaw, pitch].

    order : str
        'lwh' or 'hwl'
    """
    boxes_lidar = []
    corners_lidar_list = []
    for object_content in object_list: 
        lidar_to_world = x_to_world(lidar_pose) # T_world_lidar
        world_to_lidar = np.linalg.inv(lidar_to_world)

        corners_world = np.array(object_content['world_8_points']) # [8,3]
        corners_world_homo = np.pad(corners_world, ((0,0), (0,1)), constant_values=1) # [8, 4]
        corners_lidar = (world_to_lidar @ corners_world_homo.T).T 
        corners_lidar_list.append(corners_lidar)

        bbx_lidar = corners_lidar
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0) # [1, 8, 3]
        bbx_lidar = corner_to_center(bbx_lidar, order=order)
        boxes_lidar.append(bbx_lidar[0])
    return boxes_lidar, corners_lidar_list

class DAIRV2XBaseDataset(Dataset):
    print('###########Dairv2x_Basedataset Running#############')
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        #self.normolize_intensity = params["preprocess"]["normolize_intensity"]
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        if 'global_preprocess' in params:
            self.global_preprocessor = build_preprocessor(params["global_preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.post_processor.generate_gt_bbx = self.post_processor.generate_gt_bbx_by_iou
        self.data_augmentor = DataAugmentor(params['data_augment'], train, params['data_dir'], params['class_names'])

        if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        assert self.load_depth_file is False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                                    else self.generate_object_center_camera

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        if self.train:
            split_dir = params['root_dir']#/data/dairv2x/cooperative-vehicle-infrastructure/train.json
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']# /data/dairv2x/cooperative-vehicle-infrastructure

        print('communication range: ', self.params['comm_range'])

        self.split_info = read_json(split_dir)#训练集的序号，如000354,001235
        co_datainfo = read_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            #print(veh_frame_id)
            self.co_data[veh_frame_id] = frame_info

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False
    
    def reinitialize(self):
        pass

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.
        NOTICE!
        It is different from Intermediate Fusion and Early Fusion
        Label is not cooperative and loaded for both veh side and inf side.
        Parameters
        ----------
        idx : int
            Index given by dataloader.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        veh_frame_id = self.split_info[idx]#idx 为0,1,2,3,4,5,
        frame_info = self.co_data[veh_frame_id] #veh_frame_id为序号，如000354
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()

        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[1] = OrderedDict()
        data[1]['ego'] = False

        data[0]['params'] = OrderedDict()
        data[1]['params'] = OrderedDict()
        
        # pose of agent 
        lidar_to_novatel = read_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world = read_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))
        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel, novatel_to_world)
        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        virtuallidar_to_world = read_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        transformation_matrix = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world, system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        data[0]['params']['vehicles_front'] = read_json(os.path.join(self.root_dir,frame_info['cooperative_label_path'].replace("label_world", "label_world_backup"))) 
        data[0]['params']['vehicles_all'] = read_json(os.path.join(self.root_dir,frame_info['cooperative_label_path'])) 

        data[1]['params']['vehicles_front'] = [] # we only load cooperative label in vehicle side
        data[1]['params']['vehicles_all'] = [] # we only load cooperative label in vehicle side

        if self.load_camera_file:
            data[0]['camera_data'] = load_camera_data([os.path.join(self.root_dir, frame_info["vehicle_image_path"])])
            data[0]['params']['camera0'] = OrderedDict()
            data[0]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix( \
                                            read_json(os.path.join(self.root_dir, 'vehicle-side/calib/lidar_to_camera/'+str(veh_frame_id)+'.json')))
            data[0]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X( \
                                            read_json(os.path.join(self.root_dir, 'vehicle-side/calib/camera_intrinsic/'+str(veh_frame_id)+'.json')))
            
            data[1]['camera_data']= load_camera_data([os.path.join(self.root_dir,frame_info["infrastructure_image_path"])])
            data[1]['params']['camera0'] = OrderedDict()
            data[1]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix( \
                                            read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/virtuallidar_to_camera/'+str(inf_frame_id)+'.json')))
            data[1]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X( \
                                            read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/camera_intrinsic/'+str(inf_frame_id)+'.json')))


        if self.load_lidar_file or self.visualize:
            data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
            data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["infrastructure_pointcloud_path"]))


        # Label for single side
        data[0]['params']['vehicles_single_front'] = read_json(os.path.join(self.root_dir, \
                                'vehicle-side/label/lidar_backup/{}.json'.format(veh_frame_id)))
        data[0]['params']['vehicles_single_all'] = read_json(os.path.join(self.root_dir, \
                                'vehicle-side/label/lidar/{}.json'.format(veh_frame_id)))
        data[1]['params']['vehicles_single_front'] = read_json(os.path.join(self.root_dir, \
                                'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id)))
        data[1]['params']['vehicles_single_all'] = read_json(os.path.join(self.root_dir, \
                                'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id)))


        return data,veh_frame_id


    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx):
        pass


    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_all']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_front']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)
                                                        
    ### Add new func for single side
    def generate_object_center_single(self,
                               cav_contents,
                               reference_lidar_pose,
                               **kwargs):
        """
        veh or inf 's coordinate
        """
        suffix = "_single"
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                    cav_content['params']['vehicles_single_front'] if self.label_type == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        return self.post_processor.generate_object_center_dairv2x_single(cav_contents, suffix)

    def get_ext_int(self, params, camera_id):
        lidar_to_camera = params["camera%d" % camera_id]['extrinsic'].astype(np.float32) # R_cw
        camera_to_lidar = np.linalg.inv(lidar_to_camera) # R_wc
        camera_intrinsic = params["camera%d" % camera_id]['intrinsic'].astype(np.float32
        )
        return camera_to_lidar, camera_intrinsic

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask, random_seed=None, choice=None):
        """
        Given the raw point cloud, augment by flipping and rotation.
        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape
        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw
        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        if random_seed:
            self.data_augmentor.random_seed = random_seed
        if choice:
            tmp_dict = self.data_augmentor.forward(tmp_dict, choice)
        else:
            tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask
    

    # lixiang 2023.3 add create gt sampling database
    def create_groundtruth_database(self, used_classes=None, split='train', sensor='vehicle'):
        import torch
        from pathlib import Path
        import pickle
        import tqdm
        from opencood.utils.transformation_utils import x1_to_x2
        from opencood.utils import box_utils

        database_save_path = Path(self.root_dir) / ('gt_database_%s' % sensor)
        db_info_save_path = Path(self.root_dir) / ('dairv2x_dbinfos_%s.pkl' % sensor)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        print('generating {} gt-databse for augmentation'.format(sensor))

        if sensor == 'mix':
            sensor_list = ['vehicle', 'fusion', 'inf']
        else:
            sensor_list = None

        for k in tqdm.tqdm(range(len(self.split_info))):
            base_data_dict = self.retrieve_base_data(k)
            sample_idx = self.split_info[k]
            print('*****************create_groundtruth_database')
            print(k,sample_idx)
            if sensor_list is not None:
                sensor = random.choice(sensor_list)

            if sensor == 'vehicle':
                points = base_data_dict[0]['lidar_np']
                annos = base_data_dict[0]['params']['vehicles_single_all'] #ego coord
                boxes_lidar = []
                for anno in annos:
                    x = anno['3d_location']['x']
                    y = anno['3d_location']['y']
                    z = anno['3d_location']['z']
                    l = anno['3d_dimensions']['l']
                    h = anno['3d_dimensions']['h']
                    w = anno['3d_dimensions']['w']
                    rotation = anno['rotation']
                    box_lidar = [x,y,z,l,w,h,rotation] 
                    boxes_lidar.append(box_lidar)
            elif sensor == 'fusion':
                ego_lidar_pose = base_data_dict[0]['params']['lidar_pose']
                projected_lidar_stack = []
                for cav_id, selected_cav_base in base_data_dict.items():
                    # transformation
                    transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_lidar_pose)
                    lidar_np = selected_cav_base['lidar_np']
                    # project the lidar to ego space
                    lidar_np[:, :3] = \
                        box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                    # all these lidar and object coordinates are projected to ego already.
                    projected_lidar_stack.append(lidar_np)
                projected_lidar_stack = np.vstack(projected_lidar_stack)
                points = projected_lidar_stack
                annos = base_data_dict[0]['params']['vehicles_all']
                boxes_lidar, _ = project_world_objects_dairv2x(annos, ego_lidar_pose)
            
            elif sensor == 'inf':
                ego_lidar_pose = base_data_dict[0]['params']['lidar_pose']
                if len(base_data_dict) == 1:
                    continue
                for cav_id, selected_cav_base in base_data_dict.items():
                # transformation
                    if cav_id == 0:
                        continue
                    transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_lidar_pose)
                    lidar_np = selected_cav_base['lidar_np']
                    # project the lidar to ego space
                    lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                    # all these lidar and object coordinates are projected to ego already.
                    points = lidar_np
                    annos = base_data_dict[0]['params']['vehicles_all']
                    boxes_lidar, _ = project_world_objects_dairv2x(annos, ego_lidar_pose)

            elif sensor == 'sensor_mix':
                ego_lidar_pose = base_data_dict[0]['params']['lidar_pose']
                projected_lidar_stack = []
                for cav_id, selected_cav_base in base_data_dict.items():
                    # transformation
                    transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_lidar_pose)
                    lidar_np = selected_cav_base['lidar_np']
                    # project the lidar to ego space
                    lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                    # all these lidar and object coordinates are projected to ego already.
                    projected_lidar_stack.append(lidar_np)
                
                #instance-level mixup augmentation
                if len(projected_lidar_stack) == 2:
                    ego_point_num, inf_point_num = projected_lidar_stack[0].shape[0], projected_lidar_stack[1].shape[0]
                    mix_ratio = np.random.random()
                    ego_choose_num = int(ego_point_num * mix_ratio)
                    inf_choose_num = int(ego_point_num * (1-mix_ratio))
                    ego_choose_mask = np.random.choice(ego_point_num, ego_choose_num)
                    inf_choose_mask = np.random.choice(inf_point_num, inf_choose_num)
                    projected_lidar_stack[0] = projected_lidar_stack[0][ego_choose_mask]
                    projected_lidar_stack[1] = projected_lidar_stack[1][inf_choose_mask]


                projected_lidar_stack = np.vstack(projected_lidar_stack)
                points = projected_lidar_stack
                annos = base_data_dict[0]['params']['vehicles_all']
                boxes_lidar, _ = project_world_objects_dairv2x(annos, ego_lidar_pose)

            else:
                raise NameError("gt sampling method: {} is not implementated".format(sensor))
            
            if len(base_data_dict)  == 1 and sensor == 'inf':
                continue
            else:
                names = np.array([anno['type'] for anno in annos])
                gt_boxes = np.array(boxes_lidar)

                num_obj = gt_boxes.shape[0]
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints)

                #save early-fusion points
                # filename_pts = '%s_early_fusion_pts.bin' % (sample_idx)
                # filepath = database_save_path / filename_pts
                # with open(filepath, 'w') as f:
                #     points.tofile(f)
                #save gt_boxes
                # filename_gt = '%s_early_fusion_gt' % (sample_idx)
                # filepath = database_save_path / filename_gt
                # np.save(filepath, np.array(corners_lidar_list))

                for i in range(num_obj):
                    filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                    filepath = database_save_path / filename
                    gt_points = points[point_indices[i] > 0]

                    gt_points[:, :3] -= gt_boxes[i, :3]
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    if (used_classes is None) or names[i] in used_classes:
                        db_path = str(filepath.relative_to(self.root_dir))  # gt_database/xxxxx.bin
                        db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                        if names[i] in all_db_infos:
                            all_db_infos[names[i]].append(db_info)
                        else:
                            all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, default='/home/wangsichao/myv2x/opencood/hypes_yaml/dairv2x/lidar_only/pointpillar_single.yaml',
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    
    return opt



if __name__ == '__main__':
    opt = train_parser()
    # hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    # print('###########Dairv2x_Basedataset Running')
    # print('Dataset Building')
    # dairv2x_base_dataset = DAIRV2XBaseDataset(hypes, visualize=False, train=True) #[vehicle, inf, cooperative]
    # dairv2x_base_dataset.create_groundtruth_database(sensor='fusion') #create gt base for [vehicle, fuison, inf, mix, sensor_mix] data

