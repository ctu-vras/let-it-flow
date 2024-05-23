# Imports and data
import numpy as np
import os
import sys
import glob
import torch
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

from pytorch3d.ops.knn import knn_points

from models.chodosh import infer_chodosh, fit_NeuralPrior, fit_MB_NeuralPrior
from models import let_it_flow
from data.argoverse2 import sample_argoverse2


with open('config.yaml') as file:
    config_file = yaml.load(file, Loader=yaml.FullLoader)

cfg = config_file['cfg']
use_gpu = 0

if torch.cuda.is_available():
    device = torch.device(cfg['device'])
else:
    device = torch.device('cpu')
    
available_gpus = cfg['nbr_of_devices']

Argoverse2_seqs = 150
seq_arrays = np.array_split(range(Argoverse2_seqs), available_gpus)


model = cfg['model']
##### Implemented models
# model = 'MBNSFP'
# model = 'chodosh'
# model = 'NP'

# Hyperparams
frame = cfg['frame']
dist_w = cfg['dist_w']
TEMPORAL_RANGE = cfg['TEMPORAL_RANGE']
sc_w = cfg['sc_w']
trunc_dist = cfg['trunc_dist']
passing_ids = cfg['passing_ids']
K = cfg['K']
d_thre = cfg['d_thre']
eps = cfg['eps']
min_samples = cfg['min_samples']
lr = cfg['lr']

# changes for tests
# import socket
# if socket.gethostname().capitalize() == 'Patrik':
folder_path = cfg['folder_path']
store_path = cfg['store_path']

# else:
#     folder_path = '/mnt/personal/vacekpa2/data/argoverse2/'
#     store_path = '/mnt/personal/vacekpa2/data/argoverse2_results/'
os.makedirs(store_path, exist_ok=True)

for i, seq_id in tqdm(enumerate(seq_arrays[use_gpu]), total=len(seq_arrays[use_gpu])):
    # time includes the loading of the data
    global_list, poses, gt_flow, compensated_gt_flow_list, dynamic_list, category_indices_list, seq_names = \
    sample_argoverse2(folder_path, seq_id, cfg=cfg)

    pcs = np.concatenate(global_list[:TEMPORAL_RANGE], axis=0)

    p1 = pcs[pcs[:,3] == frame][None, :,:3]
    p2 = pcs[pcs[:,3] == frame + 1][None, :,:3]


    if model == 'let_it_flow':

        p1, p2, c1, c2, f1 = let_it_flow.initial_clustering(global_list, frame, TEMPORAL_RANGE, device, eps=eps, min_samples=min_samples, z_scale=0.5)

        optimizer = torch.optim.Adam([f1], lr=lr)

        RigidLoss = let_it_flow.SC2_KNN_cluster_aware(p1, K=K, d_thre=d_thre)

        for it in range(cfg['iters']): 
            loss = 0

            dist, nn, _ = knn_points(p1 + f1, p2, lengths1=None, lengths2=None, K=1, return_nn=True)   
            dist_b, nn_b, _ = knn_points(p2, p1 + f1, lengths1=None, lengths2=None, K=1, return_nn=True)    
            loss += dist_w * (dist[dist < trunc_dist].mean() + dist_b[dist_b < trunc_dist].mean())

            sc_loss = RigidLoss(f1, c1)
            
            if sc_w > 0:
                loss += sc_w * sc_loss
                loss += sc_w * let_it_flow.center_rigidity_loss(p1, f1, c1[None] + 1)    # + 1 for noise, works!

            loss += f1[..., 2].norm().mean()
            

            if it % 10 == 0 and passing_ids:
                c1 = let_it_flow.pass_id_clusters(c1, c2, nn)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        
        
    elif model == 'chodosh':
        f1 = infer_chodosh(p1[0], p1[0], p2[0], max_iters=cfg['iters']).unsqueeze(0)
    elif model == 'NP':
        f1 = fit_NeuralPrior(p1, p2, max_iters=cfg['iters'])
    elif model == 'MBNSFP':
        f1 = fit_MB_NeuralPrior(p1, p2, max_iters=cfg['iters'])
        
    # store flow
    store_dict = {'p1' : p1.detach().cpu().numpy(), 'p2' : p2.detach().cpu().numpy(),
                  'c1' : c1.detach().cpu().numpy(), 'f1' : f1.detach().cpu().numpy(),
                  'gt_flow' : gt_flow[frame], 'compensated_gt_flow' : compensated_gt_flow_list[frame], 'dynamic' : dynamic_list[frame],
                  'category_indices' : category_indices_list[frame], "model" : model}
    
    np.savez(store_path + f'/{seq_names[seq_id]}.npz', **store_dict)
    
    if i == 3: break
