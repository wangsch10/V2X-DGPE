"""
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
"""

import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
import tqdm
from matplotlib import pyplot as plt
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

from pathlib import Path

torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, default='/home/lixiang/CoAlign/opencood/logs/dairv2x_point_pillar_lidar_early_e90_2023_04_05_10_57_55',
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='early',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=100,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="mae", type=str, help="early-fusion, intermediate-fusion")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)
    hypes['validate_dir'] = hypes['test_dir']

    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)

    
    infer_info = opt.fusion_method + opt.note

    vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
    if not os.path.exists(vis_save_path_root):
        os.makedirs(vis_save_path_root)

    database_save_path = '/home/lixiang/CoAlign/visualize/'
    if not os.path.exists(database_save_path):
        os.makedirs(database_save_path)

    for i, batch_data in enumerate(tqdm.tqdm(data_loader)):
        if i % opt.save_vis_interval != 0:
            continue
        with torch.no_grad():
            #batch_data = train_utils.to_device(batch_data, device)
            # vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
            # for early_fusion
            ego_lidar = batch_data['ego']['ego_lidar']
            inf_lidar = batch_data['ego']['inf_lidar'] if 'inf_lidar' in batch_data['ego'] else None
            # for intermediate fusion
            intermediate_fusion_lidar = batch_data['ego']['origin_lidar'][0]
            

            # save point clouds to .bin file
            save_bin_file = 1
            if save_bin_file:
                num_vpoint, num_apoint = batch_data['ego']['ego_lidar'].shape[0], batch_data['ego']['origin_lidar'][0].shape[0]
                if inf_lidar is not None:
                    early_fusion_lidar = np.concatenate([np.array(ego_lidar), np.array(inf_lidar)], 0)
                else:
                    continue
                #intermediate_fusion_lidar = np.array(intermediate_fusion_lidar)
                filepath = os.path.join(database_save_path , '{}_{}_{}.bin'.format(i, num_vpoint, num_apoint))
                with open(filepath, 'w') as f:
                    early_fusion_lidar.tofile(f)

            # visualize(intermediate_fusion_lidar, hypes['postprocess']['gt_range'],
            #                     vis_save_path,
            #                     method='bev',
            #                     left_hand=left_hand)
            
        torch.cuda.empty_cache()




def visualize(pcd, pc_range, save_path, method='3d', left_hand=False, inf_pcd=None):
        """
        Visualize the prediction, ground truth with point cloud together.
        They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

        Parameters
        ----------
        pcd : torch.Tensor
            PointCloud, (N, 4).

        pc_range : list
            [xmin, ymin, zmin, xmax, ymax, zmax]

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        method: str, 'bev' or '3d'

        """
        plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40])
        pc_range = [int(i) for i in pc_range]

        pcd_np = pcd.numpy()
        

        if method == 'bev':
            canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=left_hand) 

            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
            canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(255,255,255)) # Only draw ego points

            # if inf_pcd is not None:
            #     inf_pcd_np = inf_pcd.numpy()
            #     canvas_xy, valid_mask = canvas.get_canvas_coords(inf_pcd_np)
            #     canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(255,0,0)) # Only draw lidar points



        plt.axis("off")

        plt.imshow(canvas.canvas)
        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=500)
        plt.clf()
        plt.close()




if __name__ == '__main__':
    main()
