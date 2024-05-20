import os
os.chdir('/home.dokt/vacekpa2/let-it-flow')
import torch
import numpy as np
from pytorch3d.ops.knn import knn_points

from ops.visibility2D import *

import sys
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data.PATHS import EXP_PATH



from pytorch3d.ops.knn import knn_points
from models.NP import NeuralPriorNetwork

from loss.flow import SC2_KNN, GeneralLoss


from configs.evalute_datasets import cfg as cfg_df

def generate_polar_KNN(pc, K=9, max_depth_diff=2.5):
    '''
    :param pc: point cloud 1 x N x 3
    :param max_depth_diff: maximum depth difference
    :return: nearest neighbors 1 x N x K
    '''
    yaw, pitch, depth = calculate_polar_coords(pc[0].detach().cpu().numpy())


    polar_points = np.stack((yaw, pitch, depth), axis=-1)
    polar_points = torch.tensor(polar_points, device=pc.device).unsqueeze(0)
    dist, nn_ind, _ = knn_points(polar_points[:,:,:2], polar_points[:,:,:2], K=K)

    depth_nn = polar_points[0, :, 2][nn_ind[0]].unsqueeze(0)
    # subtract
    depth_nn = (depth_nn - polar_points[0, :, 2].unsqueeze(1)).abs()

    tmp_idx = nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, nn_ind.shape[-1]).to(nn_ind.device)

    nn_ind[depth_nn > max_depth_diff] = tmp_idx[depth_nn > max_depth_diff]

    return nn_ind

def generate_temporal_knn(pc, K=9, st_ratio=0.2, max_dist=0.4, mode='geometric'):

    assert pc.shape[-1] == 4 # x, y, z, t

    temp_pc = pc[..., :4].clone()
    temp_pc[..., 3] *= st_ratio

    if mode == 'geometric':
        d, nn_ind, _ = knn_points(temp_pc, temp_pc, K=K)
    
    else:
        raise NotImplementedError

    tmp_idx = nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, nn_ind.shape[-1]).to(nn_ind.device)
    valid_idx = (d < max_dist)
    nn_ind[d > max_dist] = tmp_idx[d > max_dist]

    return nn_ind, valid_idx

# def majority_knn_voting():
# for i in range(len(mask) - 1):
    #     indexed_ids = id_mask[i, nn_ind_f[i]]
    #     maxed_id, _ = torch.mode(indexed_ids, dim=-1)

    #     counts = (indexed_ids == maxed_id.unsqueeze(-1)).sum(-1) # indexed_ids == maxed_id 
    #     majority_neighborhoods = (counts >= nbr_same_ids)

    #     # max_probs = torch.gather(m, 2, id_mask[..., None])  # B x N x 1
    #     majority_KNN_loss = - torch.log(max_probs[i, majority_neighborhoods]).mean()

    #     loss += majority_KNN_loss
    
    
def fit_polar_rigidity_prior(pc1, deformed_pc1, pc2, K=16, beta=1, max_iters=400, lr=0.004):
    # Assumes already poses from kiss-icp in deformed_pc1
    
    # Init model for each sequence frame
    
    model = NeuralPriorNetwork().to(pc1.device)    
    net_inv = NeuralPriorNetwork().to(pc1.device)

    params = [{'params': model.parameters(), 'lr': lr, 'weight_decay': 0},
              {'params': net_inv.parameters(), 'lr': lr, 'weight_decay': 0}]

    optimizer = torch.optim.Adam(params, lr=lr)
    
    
    # LossModule = GeneralLoss(pc1=pc1, pc2=pc2, dist_mode='DT', K=0, max_radius=2,
    #                             smooth_weight=0,
    #                             forward_weight=0, sm_normals_K=0, pc2_smooth=False)

    # LossModule2 = GeneralLoss(pc1=pc2, pc2=pc1, dist_mode='DT', K=0, max_radius=2,
    #                             smooth_weight=0,
    #                             forward_weight=0, sm_normals_K=0, pc2_smooth=False)

    nn_ind = generate_polar_KNN(pc1, K=K)    
    SC2_Loss = SC2_KNN(pc1=pc1, K=K, use_normals=False)
    SC2_Loss.kNN = nn_ind

    for flow_e in range(max_iters):
        pc1 = pc1.contiguous()

        pred_flow = model(deformed_pc1)

        pc1_deformed_forward = deformed_pc1 + pred_flow
        
        # Backward flow and cycle consistency in one step
        flow_pred_1_prime = net_inv(pc1_deformed_forward)
        pc1_prime_deformed = pc1_deformed_forward - flow_pred_1_prime
        

        # loss1 = LossModule(deformed_pc1, pred_flow, pc2)
        
        loss1, _, _ = knn_points(pc1_deformed_forward, pc2)
        loss1_prime, _, _ = knn_points(pc1_prime_deformed, pc1)
        loss = loss1[loss1 < 2].mean() + loss1_prime[loss1_prime < 2].mean()
# 
        # loss1 = LossModule(deformed_pc1, pred_flow, pc2)
        # loss1_prime = LossModule2(pc1_deformed_forward, - flow_pred_1_prime, pc1)
        
        
        # loss = loss1.mean() + loss1_prime.mean()
        
        loss += beta * SC2_Loss(pred_flow)

        loss.mean().backward()

        optimizer.step()
        optimizer.zero_grad()

    ego_flow = deformed_pc1 - pc1

    final_flow = pred_flow + ego_flow
    
    return final_flow