import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from pytorch3d.ops.knn import knn_points
from tqdm import tqdm

from loss.flow import MBSC

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

class NeuralPriorNetwork(torch.nn.Module):
    
    def __init__(self, lr=0.008, early_stop=30, loss_diff=0.001, dim_x=3, filter_size=128, act_fn='relu', layer_size=8, initialize=True,
                 verbose=False, **kwargs):
        super().__init__()
        self.layer_size = layer_size
        bias = True
        self.nn_layers = torch.nn.ModuleList([])

        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size, bias=bias)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size - 1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size, bias=bias)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x, bias=bias))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x, bias=bias)))

        if initialize:
            self.apply(init_weights)

        self.lr = lr
        self.early_stop = early_stop
        self.loss_diff = loss_diff
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.verbose = verbose

    def update(self, pc1=None, pc2=None):
        pass
    
    def forward(self, x):

        for layer in self.nn_layers:
            x = layer(x)

        return x


def fit_rigid(A, B):
    """
    Fit Rigid transformation A @ R.T + t = B
    """
    assert A.shape == B.shape
    num_rows, num_cols = A.shape

    # find mean column wise
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # ensure centroids are 1x3
    centroid_A = centroid_A.reshape(1, -1)
    centroid_B = centroid_B.reshape(1, -1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -centroid_A @ R.T + centroid_B

    return R, t


def ransac_fit_rigid(pts1, pts2, inlier_thresh, ntrials):
    best_R = np.eye(3)
    best_t = np.zeros((3, ))
    best_inliers = (np.linalg.norm(pts1 - pts2, axis=-1) < inlier_thresh)
    best_inliers_sum = best_inliers.sum()
    for i in range(ntrials):
        choice = np.random.choice(len(pts1), 3)
        R, t = fit_rigid(pts1[choice], pts2[choice])
        inliers = (np.linalg.norm(pts1 @ R.T + t - pts2, axis=-1) <
                   inlier_thresh)
        if inliers.sum() > best_inliers_sum:
            best_R = R
            best_t = t
            best_inliers = inliers
            best_inliers_sum = best_inliers.sum()
            if best_inliers_sum / len(pts1) > 0.5:
                break
    best_R, best_t = fit_rigid(pts1[best_inliers], pts2[best_inliers])
    return best_R, best_t

def fit_NeuralPrior(pc1, pc2, max_iters=500):
    """
    Fit Neural Prior to pointclouds after ego-motion
    """
    
    model = NeuralPriorNetwork().to(pc1.device)                
    net_inv = NeuralPriorNetwork().to(pc1.device)

    params = [{'params': model.parameters(), 'lr': 0.004, 'weight_decay': 0},
              {'params': net_inv.parameters(), 'lr': 0.004, 'weight_decay': 0}]

    optimizer = torch.optim.Adam(params, lr=0.004)
    
            # Parametrized sceneflow loss module
    # LossModule = GeneralLoss(pc1=pc1, pc2=pc2, dist_mode='DT', K=0, max_radius=cfg['max_radius'],
    #                             smooth_weight=0,
    #                             forward_weight=0, sm_normals_K=0, pc2_smooth=False)

    for flow_e in tqdm(range(max_iters)):
        pc1 = pc1.contiguous()

        pred_flow = model(pc1)

        pc1_deformed = pc1 + pred_flow
        
        # Backward flow and cycle consistency in one step
        flow_pred_1_prime = net_inv(pc1_deformed)
        pc1_prime_deformed = pc1_deformed - flow_pred_1_prime
        
        loss1, _, _ = knn_points(pc1 + pred_flow, pc2)
        loss1_prime, _, _ = knn_points(pc1_prime_deformed, pc1)

        loss = loss1[loss1 < 2].mean() + loss1_prime[loss1_prime < 2].mean()
        
        loss.backward()
        # loss = LossModule(pc1, pred_flow, pc2)

        optimizer.step()
        optimizer.zero_grad()


    return pred_flow


def fit_MB_NeuralPrior(pc1, pc2, max_iters=500):
    """
    Fit Neural Prior to pointclouds after ego-motion
    """
    
    model = NeuralPriorNetwork().to(pc1.device)                
    net_inv = NeuralPriorNetwork().to(pc1.device)

    params = [{'params': model.parameters(), 'lr': 0.004, 'weight_decay': 0},
              {'params': net_inv.parameters(), 'lr': 0.004, 'weight_decay': 0}]

    optimizer = torch.optim.Adam(params, lr=0.004)
    
            # Parametrized sceneflow loss module
    # LossModule = GeneralLoss(pc1=pc1, pc2=pc2, dist_mode='DT', K=0, max_radius=cfg['max_radius'],
    #                             smooth_weight=0,
    #                             forward_weight=0, sm_normals_K=0, pc2_smooth=False)

    MBLoss = MBSC(pc1)

    for flow_e in tqdm(range(max_iters)):
        pc1 = pc1.contiguous()

        pred_flow = model(pc1)

        pc1_deformed = pc1 + pred_flow
        
        # Backward flow and cycle consistency in one step
        flow_pred_1_prime = net_inv(pc1_deformed)
        pc1_prime_deformed = pc1_deformed - flow_pred_1_prime
        
        loss1, _, _ = knn_points(pc1 + pred_flow, pc2)
        loss1_prime, _, _ = knn_points(pc1_prime_deformed, pc1)

        loss = loss1[loss1 < 2].mean() + loss1_prime[loss1_prime < 2].mean()
        loss += MBLoss(pred_flow)
        
        loss.backward()
        # loss = LossModule(pc1, pred_flow, pc2)

        optimizer.step()
        optimizer.zero_grad()


    return pred_flow


def refine_flow(pc0,
                flow_pred,
                eps=0.4,
                min_points=10,
                motion_threshold=0.05,
                inlier_thresh=0.2,
                ntrials=250):
    labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(pc0)
    max_label = labels.max()
    refined_flow = np.zeros_like(flow_pred)
    for l in range(max_label + 1):
        label_mask = labels == l
        cluster_pts = pc0[label_mask]
        cluster_flows = flow_pred[label_mask]
        R, t = ransac_fit_rigid(cluster_pts, cluster_pts + cluster_flows,
                                inlier_thresh, ntrials)
        if np.linalg.norm(t) < motion_threshold:
            R = np.eye(3)
            t = np.zeros((3, ))
        refined_flow[label_mask] = (cluster_pts @ R.T + t) - cluster_pts
        
    refined_flow[labels == -1] = flow_pred[labels == -1]
    
    return refined_flow

def infer_chodosh(pc1, deformed_pc1, pc2, max_iters=1500):

    pred_flow = fit_NeuralPrior(deformed_pc1.to(torch.float).unsqueeze(0), pc2.unsqueeze(0), max_iters=max_iters).squeeze(0)

    chodosh_pc1 = deformed_pc1.detach().cpu().numpy()
    chodosh_pred_flow = pred_flow.detach().cpu().numpy()
    chodosh_refined_flow = refine_flow(chodosh_pc1, chodosh_pred_flow)

    ego_flow = deformed_pc1 - pc1
    final_flow = ego_flow + torch.from_numpy(chodosh_refined_flow).to(pc1.device)

    return final_flow

# final_flow = infer_chodosh(pc1, deformed_pc1, pc2)