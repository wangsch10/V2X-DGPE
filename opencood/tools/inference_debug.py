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
torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, default='/home/yc37929/lixiang/CoAlign_1130/opencood/logs/dairv2x_pointpillar_DIV2X_naivecomm_kd_2024_01_11_13_05_17',
                        help='Continued training path')
    parser.add_argument('--model2_dir', type=str, default='',
                        help='Continued training path for inf model')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single', 'late_2model', 'no_inf'] 

    hypes = yaml_utils.load_yaml(None, opt)
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if opt.fusion_method == 'late_2model':
        model_inf = train_utils.create_model(hypes)
        saved_path = opt.model2_dir
        resume_epoch, model_inf = train_utils.load_saved_model(saved_path, model_inf)
        print(f"inf model resume from {resume_epoch} epoch.")
        if torch.cuda.is_available():
            model_inf.cuda()
        model_inf.eval()

    # setting noise
    np.random.seed(303)
    
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
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    comm_rates = []

    
    infer_info = opt.fusion_method + opt.note


    for i, batch_data in enumerate(data_loader):
        print(f"{infer_info}_{i}")
        if (i % opt.save_vis_interval != 0):
            continue
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            elif opt.fusion_method == 'late_2model':
                infer_result = inference_utils.inference_late_fusion_2model(batch_data,
                                                        model, model_inf,
                                                        opencood_dataset)
            elif opt.fusion_method == 'no_inf':
                if len(batch_data) == 1:
                    continue
                infer_result = inference_utils.inference_no_fusion_inf(batch_data,
                                                                model,
                                                                opencood_dataset)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            if 'comm_rate' in infer_result:
                comm_rates.append(infer_result['comm_rate'])

            
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, lidar_agent_record = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "lidar_agent_record": lidar_agent_record})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                """
                If you want 3D visualization, uncomment lines below
                """
                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)

            vis_feats = 0
            if vis_feats and  (i % opt.save_vis_interval == 0):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_feats_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)
                att_save_path_root = os.path.join(opt.model_dir, f'vis_att_{infer_info}')
                if not os.path.exists(att_save_path_root):
                    os.makedirs(att_save_path_root)
                tsne_save_path_root = os.path.join(opt.model_dir, f'vis_tsne_{infer_info}')
                if not os.path.exists(tsne_save_path_root):
                    os.makedirs(tsne_save_path_root)
                #vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                # my_vis.vis_features(infer_result, vis_save_path_root, i)
                # my_vis.vis_attention(infer_result, att_save_path_root, i)
                # my_vis.vis_tsne(infer_result, batch_data['ego']['origin_lidar'],
                #                     hypes['postprocess']['gt_range'], tsne_save_path_root, i)

        torch.cuda.empty_cache()

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)
    if len(comm_rates) != 0:
        comm_rates = sum(comm_rates)/len(data_loader)
    print('communication rates: {}'.format(comm_rates))

if __name__ == '__main__':
    main()
