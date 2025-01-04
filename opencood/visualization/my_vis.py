from operator import gt
import numpy as np
import pickle
from pyquaternion import Quaternion
from matplotlib import pyplot as plt
from icecream import ic
import cv2
import os
import torch

def feature2heatmap2(features):
    '''
    features: (C H W)
    '''
    min_value = features.min(axis=(1, 2))
    max_value = features.max(axis=(1, 2))
    heatmap_pred = np.zeros_like(features)
    for c in range(heatmap_pred.shape[0]):
        heatmap_pred[c] = (features[c] - min_value[c]) / (max_value[c] - min_value[c])
    heatmap_pred[np.isnan(heatmap_pred)] = 0

    heatmap_pred = heatmap_pred.max(axis=0)
    heatmap_pred = np.power(heatmap_pred, 4) #280

    heatmap_pred[heatmap_pred < 0.2] = 0

    heatmap_pred = np.uint8(255 * heatmap_pred)[::-1, :]
    heatmap_pred = cv2.applyColorMap(cv2.resize(heatmap_pred, (1008, 400)), cv2.COLORMAP_JET)
    #heatmap_pred = heatmap_pred * 0.4 + points_bev * 0.6
    return heatmap_pred

def feature2heatmap(features):
    '''
    features: (C H W)
    '''
    min_value = features.min(axis=(1, 2))
    max_value = features.max(axis=(1, 2))
    heatmap_pred = np.zeros_like(features)
    for c in range(heatmap_pred.shape[0]):
        heatmap_pred[c] = (features[c] - min_value[c]) / (max_value[c] - min_value[c])
    heatmap_pred[np.isnan(heatmap_pred)] = 0

    heatmap_pred = heatmap_pred.max(axis=0)
    heatmap_pred = np.power(heatmap_pred, 4) 

    heatmap_pred[heatmap_pred < 0.2] = 0

    heatmap_pred = np.uint8(255 * heatmap_pred)[::-1, :]
    heatmap_pred = cv2.applyColorMap(cv2.resize(heatmap_pred, (1008, 400)), cv2.COLORMAP_JET)
    #heatmap_pred = heatmap_pred * 0.4 + points_bev * 0.6
    return heatmap_pred

def vis_features(results, vis_save_path_root, index):
    '''
    results: 
    '''
    fused_feat, single_feats = results['feature'].cpu().detach().numpy(), results['single_features'].cpu().detach().numpy()

    if single_feats.shape[0] == 2:
        heatmap_pred = feature2heatmap2(fused_feat[0])
    else:
        heatmap_pred = feature2heatmap(fused_feat[0])
    vis_save_path = os.path.join(vis_save_path_root, 'bev_{}_fusion.png'.format(index))

    cv2.imwrite(vis_save_path, heatmap_pred)
    heatmap_pred = feature2heatmap(single_feats[0])
    vis_save_path = os.path.join(vis_save_path_root, 'bev_{}_veh.png'.format(index))
    cv2.imwrite(vis_save_path, heatmap_pred)

    if single_feats.shape[0] == 2:
        heatmap_pred = feature2heatmap(single_feats[1])
        vis_save_path = os.path.join(vis_save_path_root, 'bev_{}_inf.png'.format(index))
        cv2.imwrite(vis_save_path, heatmap_pred)   

def vis_attention(results, vis_save_path_root, index):
    '''
    '''
    domain_att, spatial_att = results['domain_att'].cpu().detach().numpy(), results['spatial_att'].cpu().detach().numpy()

    if single_feats.shape[0] == 2:
        heatmap_pred = feature2heatmap(fused_feat[0])
        vis_save_path = os.path.join(vis_save_path_root, 'domain_att_{}.png'.format(index))
        cv2.imwrite(vis_save_path, heatmap_pred)

        heatmap_pred = feature2heatmap(single_feats[0])
        vis_save_path = os.path.join(vis_save_path_root, 'spatial_att_{}.png'.format(index))
        cv2.imwrite(vis_save_path, heatmap_pred)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.nn import functional as F


def downsample_lidar(pcd_np, num):
    """
    Downsample the lidar points to a certain number.

    Parameters
    ----------
    pcd_np : np.ndarray
        The lidar points, (n, 4).

    num : int
        The downsample target number.

    Returns
    -------
    pcd_np : np.ndarray
        The downsampled lidar points.
    """
    assert pcd_np.shape[0] >= num

    selected_index = np.random.choice((pcd_np.shape[0]),
                                      num,
                                      replace=False)
    pcd_np = pcd_np[selected_index]

    return pcd_np

def vis_tsne(results, pcd, pc_range, vis_save_path_root, index):
    # Assuming you have a high-dimensional dataset X
    # X.shape = (n_samples, n_features)
    fused_feat, single_feats = results['feature'], results['single_features']#.cpu().detach().numpy()
    N, C, H, W = single_feats.shape

    if N == 1:
        return

    mask = (pcd[:, :, 0] > pc_range[0]) & (pcd[:, :, 0] < pc_range[3]) & (pcd[:, :, 1] > pc_range[1]) & (pcd[:, :, 1] < pc_range[4])
    pcd = pcd[mask]
    num_point = pcd.shape[0]
    half_num = int(num_point/2)

    veh_pcd = downsample_lidar(pcd[:half_num, :], 10000)
    inf_pcd = downsample_lidar(pcd[half_num:, :], 10000)
    pcd = torch.cat((veh_pcd.unsqueeze(0), inf_pcd.unsqueeze(0)), axis=0)

    bev_coords = pcd[:, :, :2]
    bev_coords[:, :, 0] = (bev_coords[:, :, 0] - pc_range[0]) / (0.4*2)
    bev_coords[:, :, 1] = (bev_coords[:, :, 1] - pc_range[1]) / (0.4*2)        
    bev_coords[:, :, 0] = (bev_coords[:, :, 0] - W / 2) / (W / 2)
    bev_coords[:, :, 1] = (bev_coords[:, :, 1] - H / 2) / (H / 2)
    bev_coords[:, :, [0,1]] = bev_coords[:, :, [1,0]]
    feature_lidar_sample = F.grid_sample(single_feats, bev_coords.unsqueeze(1))
    feature_lidar_sample = feature_lidar_sample[:,:,0,:].permute(0,2,1).cpu().detach().numpy()

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    # Perform t-SNE embedding
    X_embedded = tsne.fit_transform(feature_lidar_sample.reshape(-1, C))
    # Visualize the embedded points
    # bev_coords = bev_coords.cpu().detach().numpy()
    plt.cla()
    # y1 = (bev_coords[0,:,0]*bev_coords[0,:,0] + bev_coords[0,:,1]*bev_coords[0,:,1])
    # y2 = (bev_coords[1,:,0]*bev_coords[1,:,0] + bev_coords[1,:,1]*bev_coords[1,:,1])
    plt.scatter(X_embedded[:10000, 0], X_embedded[:10000, 1], s=1, c=plt.cm.Dark2(0), marker='o')
    plt.scatter(X_embedded[10000:, 0], X_embedded[10000:, 1], s=1, c=plt.cm.Dark2(1), marker='o')
    #plt.show()
    vis_save_path = os.path.join(vis_save_path_root, 'tsne_{}.png'.format(index))
    plt.savefig(vis_save_path)


from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler


def plot_embedding(X, title):
    _, ax = plt.subplots()
    X = MinMaxScaler().fit_transform(X)

    for digit in digits.target_names:
        ax.scatter(
            *X[y == digit].T,
            marker=f"${digit}$",
            s=60,
            color=plt.cm.Dark2(digit),
            alpha=0.425,
            zorder=2,
        )

    ax.set_title(title)
    ax.axis("off")


def visualize(pred_box_tensor, gt_tensor, pcd, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """

        pcd_np = pcd.cpu().numpy()
        pred_box_np = pred_box_tensor.cpu().numpy()
        gt_box_np = gt_tensor.cpu().numpy()

        plt.figure(dpi=400)
        # draw point cloud. It's in lidar coordinate
        plt.scatter(pcd_np[:,0], pcd_np[:,1], s=0.5)

        N = gt_tensor.shape[0]
        for i in range(N):
            plt.plot(gt_box_np[i,:,0], gt_box_np[i,:,1], c= "r", marker='.', linewidth=1, markersize=1.5)

        N = pred_box_tensor.shape[0]
        for i in range(N):
            plt.plot(pred_box_np[i,:,0], pred_box_np[i,:,1], c= "g", marker='.', linewidth=1, markersize=1.5)
        

        plt.savefig(save_path)
        plt.clf()