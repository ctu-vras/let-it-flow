import torch
import sys
import argparse
import importlib
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from loss import sc_utils
from sklearn.cluster import DBSCAN

try:
    import FastGeodis
except:
    pass
    # print("FastGeodis not found, will not use it. This is not error, just future work")
try:
    import cupy as cp
    from cucim.core.operations import morphology
except:
    pass
    # print("Cupy works only on GPU or is not found, will not use it. This is not error, just future work")


def masked_smoothness_loss(mask, padded_mask_N, nn_ind, valid_nn_ind):
    ''' Correct verions!'''
    values = mask[padded_mask_N]

    smoothness_loss = (mask[padded_mask_N].unsqueeze(0).unsqueeze(-2) - values[nn_ind]) * valid_nn_ind.unsqueeze(-1)
    smoothness_loss = smoothness_loss.norm(dim=2)

    return smoothness_loss.mean()

def mask_NN_by_dist(dist, nn_ind, max_radius):
    # todo refactor to loss utils
    # Deprecated
    tmp_idx = nn_ind[:, :, 0].unsqueeze(2).repeat(1, 1, nn_ind.shape[-1]).to(nn_ind.device)
    nn_ind[dist > max_radius] = tmp_idx[dist > max_radius]
    print('old masking of smoothness loss in loss.flow.py')
    return nn_ind

def chamfer_distance_loss(x, y, x_lengths=None, y_lengths=None, both_ways=False, normals_K=0, loss_norm=1):
    '''
    Unique Nearest Neightboors?
    :param x:
    :param y:
    :param x_lengths:
    :param y_lengths:
    :param reduction:
    :return:
    '''
    if normals_K >= 3:
        normals1 = estimate_pointcloud_normals(x, neighborhood_size=normals_K)
        normals2 = estimate_pointcloud_normals(y, neighborhood_size=normals_K)

        x = torch.cat([x, normals1], dim=-1)
        y = torch.cat([y, normals2], dim=-1)


    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1, norm=loss_norm)
    cham_x = x_nn.dists[..., 0]  # (N, P1)
    x_nearest_to_y = x_nn[1]

    if both_ways:
        y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1, norm=loss_norm)
        cham_y = y_nn.dists[..., 0]  # (N, P2)
        # y_nearest_to_x = y_nn[1]

        nn_loss = (cham_x.mean() + cham_y.mean() ) / 2 # different shapes

    else:
        nn_loss = cham_x.mean()

    return nn_loss, cham_x, x_nearest_to_y


class FastNN(torch.nn.Module):
    '''
    Fast NN module with accelerated NN through Distance transform by with perservation of indices
    '''

    def __init__(self, pc1, pc2, cell_size=0.075):
        super().__init__()
        # ------------------ Beginning of Init ----------------------- #
        # Hyperparams
        # st = time.time()
        max_range = cp.array([40., 40., 3.5])
        min_range = cp.array([-40., -40., -1.])
        self.min_range = min_range
        cell_size = cp.array(cell_size)
        self.cell_size = cell_size
        max_radius_cell = 20
        device = pc1.device
        self.device = device
        self.max_radius_cell = max_radius_cell
        # Construction params
        size = ((max_range - min_range) / cell_size).astype(int)
        self.t_size = torch.as_tensor(size, device=device)
        index_grid = cp.ones(cp.asnumpy(size), dtype=cp.float32)
        origin = cp.array([0, 0, 0])
        origin_coors = (- min_range / cell_size).astype(int)
        self.origid_coors = origin_coors

        pc1_grid_coors = ((pc1 - min_range) / cell_size).astype(int)
        pc2_grid_coors = ((pc2 - min_range) / cell_size).astype(int)

        # pc1_grid_coors = torch.as_tensor(pc1_grid_coors, device=device)
        # pc2_grid_coors = torch.as_tensor(pc2_grid_coors, device=device)

        self.pc1_grid_coors = pc1_grid_coors
        self.pc2_grid_coors = pc2_grid_coors


        # Prepare index grid
        self.orig_index_grid = torch.zeros((self.t_size[0], self.t_size[1], self.t_size[2]), dtype=torch.long,
                                           device=device)

        self.orig_index_grid[pc2_grid_coors[0, :, 0], pc2_grid_coors[0, :, 1], pc2_grid_coors[0, :, 2]] = torch.arange(
                pc2_grid_coors.shape[1], device=device)  # ordering sequentially

        index_grid[pc2_grid_coors[0, :, 0], pc2_grid_coors[0, :, 1], pc2_grid_coors[0, :, 2]] = 0  # cp.arange(pc2.shape[1])

        # _ = time.time()
        # torch.as_tensor(index_grid, device=device)
        # print('index grid trans: ', time.time() - _)

        # We take two times the truncated CD to be sure the points will have precise neighbours. Then we just overlap it (union should be same then)
        # Calculate DT in halves to fit the gpu card and prevent curse of dimensionality

        ###     _________
        ###     | 1 | 3 |
        ###     | 2 | 4 |
        ###     ---------

        self.first_q = index_grid[0: origin_coors[0] + 2 * max_radius_cell, 0: origin_coors[1] + 2 * max_radius_cell, :]
        self.second_q = index_grid[0: origin_coors[0] + 2 * max_radius_cell, origin_coors[1] - 2 * max_radius_cell:, :]
        self.third_q = index_grid[origin_coors[0] - 2 * max_radius_cell:, : origin_coors[1] + 2 * max_radius_cell, :]
        self.fourth_q = index_grid[origin_coors[0] - 2 * max_radius_cell:, origin_coors[1] - 2 * max_radius_cell:, :]

        origin_coors = torch.as_tensor(origin_coors, device=device)

        ### Shift
        first_ind_shift = torch.tensor([0, 0, 0], device=device).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        second_ind_shift = torch.tensor([0, origin_coors[1] - 2 * max_radius_cell, 0], device=device).unsqueeze(
            1).unsqueeze(1).unsqueeze(1)
        third_ind_shift = torch.tensor([origin_coors[0] - 2 * max_radius_cell, 0, 0], device=device).unsqueeze(
            1).unsqueeze(1).unsqueeze(1)
        fourth_ind_shift = torch.tensor(
                [origin_coors[0] - 2 * max_radius_cell, origin_coors[1] - 2 * max_radius_cell, 0],
                device=device).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # ------------------ End of Init ----------------------- #

        # ------------------ Beginning of Update ----------------------- #
        f_dt, f_inds = morphology.distance_transform_edt(self.first_q, return_indices=True, float64_distances=False)
        s_dt, s_inds = morphology.distance_transform_edt(self.second_q, return_indices=True, float64_distances=False)
        t_dt, t_inds = morphology.distance_transform_edt(self.third_q, return_indices=True, float64_distances=False)
        fou_dt, fou_inds = morphology.distance_transform_edt(self.fourth_q, return_indices=True,
                                                             float64_distances=False)

        f_dt, f_inds = torch.as_tensor(f_dt, device=device), torch.as_tensor(f_inds, device=device)
        s_dt, s_inds = torch.as_tensor(s_dt, device=device), torch.as_tensor(s_inds, device=device)
        t_dt, t_inds = torch.as_tensor(t_dt, device=device), torch.as_tensor(t_inds, device=device)
        fou_dt, fou_inds = torch.as_tensor(fou_dt, device=device), torch.as_tensor(fou_inds, device=device)

        f_inds += first_ind_shift
        s_inds += second_ind_shift
        t_inds += third_ind_shift
        fou_inds += fourth_ind_shift

        # Concatenation of all indices across the map
        first_half = torch.cat((f_inds[:, : (self.t_size[0] / 2).long(), : (self.t_size[1] / 2).long()],
                                s_inds[:, : (self.t_size[0] / 2).long(), 2 * self.max_radius_cell:]), dim=2)

        second_half = torch.cat((t_inds[:, 2 * self.max_radius_cell:, : (self.t_size[1] / 2).long()],
                                 fou_inds[:, 2 * self.max_radius_cell:, 2 * self.max_radius_cell:]), dim=2)


        # Free memory
        del self.first_q, self.second_q, self.third_q, self.fourth_q, f_inds, s_inds, t_inds, fou_inds, f_dt, s_dt, t_dt, fou_dt

        self.full_ids = torch.cat((first_half, second_half), dim=1)
        # print('init: ', time.time() - st)
        # ------------------ End of Update ----------------------- #

    def forward(self, pc1, pred_flow, pc2):
        # ------------------ Beginning of Forward -------------- #
        # st = time.time()
        deformed_pc = pc1 + pred_flow

        deformed_pc_grid_coors = ((deformed_pc.detach() - self.min_range) / self.cell_size).astype(int)

        deformed_pc_grid_coors = torch.as_tensor(deformed_pc_grid_coors, device=pc1.device)

        # clip takes relatively long time, maybe tune it ?
        deformed_pc_grid_coors[..., 0] = deformed_pc_grid_coors[..., 0].clip(0, self.t_size[0] - 1)
        deformed_pc_grid_coors[..., 1] = deformed_pc_grid_coors[..., 1].clip(0, self.t_size[1] - 1)
        deformed_pc_grid_coors[..., 2] = deformed_pc_grid_coors[..., 2].clip(0, self.t_size[2] - 1)

        NN_idx = self.full_ids[:, deformed_pc_grid_coors[0, :, 0], deformed_pc_grid_coors[0, :, 1],
                 deformed_pc_grid_coors[0, :, 2]].T

        # Gather results with gradient operations
        NN_indices = self.orig_index_grid[NN_idx[:, 0], NN_idx[:, 1], NN_idx[:, 2]]
        # nn_flow = pc2[:, NN_indices] - pc1
        dist = pc2[:, NN_indices] - deformed_pc

        dist = dist.norm(dim=2)
        # print('forward: ', time.time() - st)
        # ------------------ End of Forward -------------- #
        return dist, NN_indices

class DT:
    def __init__(self, pc1, pc2, grid_factor=10, gamma=0.0):
        ''' works for batch size 1 only - modification to FNSFP'''
        self.grid_factor = grid_factor
        self.gamma = gamma
        pts = pc2[0]

        pc1_min = torch.min(pc1.squeeze(0), 0)[0]
        pc2_min = torch.min(pc2.squeeze(0), 0)[0]
        pc1_max = torch.max(pc1.squeeze(0), 0)[0]
        pc2_max = torch.max(pc2.squeeze(0), 0)[0]

        xmin_int, ymin_int, zmin_int = torch.floor(torch.where(pc1_min < pc2_min, pc1_min, pc2_min) * 10 - 1) / 10
        xmax_int, ymax_int, zmax_int = torch.ceil(torch.where(pc1_max > pc2_max, pc1_max, pc2_max) * 10 + 1) / 10

        pmin = (xmin_int, ymin_int, zmin_int)
        pmax = (xmax_int, ymax_int, zmax_int)

        sample_x = ((pmax[0] - pmin[0]) * grid_factor).ceil().int() + 2
        sample_y = ((pmax[1] - pmin[1]) * grid_factor).ceil().int() + 2
        sample_z = ((pmax[2] - pmin[2]) * grid_factor).ceil().int() + 2

        self.Vx = torch.linspace(0, sample_x, sample_x+1, device=pts.device)[:-1] / grid_factor + pmin[0]
        self.Vy = torch.linspace(0, sample_y, sample_y+1, device=pts.device)[:-1] / grid_factor + pmin[1]
        self.Vz = torch.linspace(0, sample_z, sample_z+1, device=pts.device)[:-1] / grid_factor + pmin[2]

        # NOTE: build a binary image first, with 0-value occuppied points
        grid_x, grid_y, grid_z = torch.meshgrid(self.Vx, self.Vy, self.Vz, indexing="ij")
        self.grid = torch.stack([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1).float().squeeze()
        H, W, D, _ = self.grid.size()
        pts_mask = torch.ones(H, W, D, device=pts.device)
        self.pts_sample_idx_x = ((pts[:,0:1] - self.Vx[0]) * self.grid_factor).round()
        self.pts_sample_idx_y = ((pts[:,1:2] - self.Vy[0]) * self.grid_factor).round()
        self.pts_sample_idx_z = ((pts[:,2:3] - self.Vz[0]) * self.grid_factor).round()
        pts_mask[self.pts_sample_idx_x.long(), self.pts_sample_idx_y.long(), self.pts_sample_idx_z.long()] = 0.

        iterations = 1
        image_pts = torch.zeros(H, W, D, device=pts.device).unsqueeze(0).unsqueeze(0)
        pts_mask = pts_mask.unsqueeze(0).unsqueeze(0)
        self.D = FastGeodis.generalised_geodesic3d(
            image_pts, pts_mask, [1./self.grid_factor, 1./self.grid_factor, 1./self.grid_factor], 1e10, self.gamma, iterations
        ).squeeze()

    def torch_bilinear_distance(self, pc_deformed):

        pc_deformed = pc_deformed.squeeze(0)

        H, W, D = self.D.size()
        target = self.D[None, None, ...]

        sample_x = ((pc_deformed[:,0:1] - self.Vx[0]) * self.grid_factor).clip(0, H-1)
        sample_y = ((pc_deformed[:,1:2] - self.Vy[0]) * self.grid_factor).clip(0, W-1)
        sample_z = ((pc_deformed[:,2:3] - self.Vz[0]) * self.grid_factor).clip(0, D-1)

        sample = torch.cat([sample_x, sample_y, sample_z], -1)

        # NOTE: normalize samples to [-1, 1]
        sample = 2 * sample
        sample[...,0] = sample[...,0] / (H-1)
        sample[...,1] = sample[...,1] / (W-1)
        sample[...,2] = sample[...,2] / (D-1)
        sample = sample -1

        sample_ = torch.cat([sample[...,2:3], sample[...,1:2], sample[...,0:1]], -1)

        # NOTE: reshape to match 5D volumetric input
        dist = F.grid_sample(target, sample_.view(1,-1,1,1,3), mode="bilinear", align_corners=True).view(-1)


        return dist.mean(), dist


class SC2_KNN(torch.nn.Module):
    
    ''' Our soft-rigid regularization with neighborhoods
    pc1 : Point cloud
    K : Number of NN for the neighborhood
    use_normals : Whether to use surface estimation for neighborhood construction
    d_thre : constant for working with the displacements as percentual statistics, we use value from https://github.com/ZhiChen902/SC2-PCR
    '''
    def __init__(self, pc1, K=16, use_normals=False, d_thre=0.03):
        super().__init__()
        self.d_thre = d_thre
        self.K = K
        if use_normals:
            l = GeneralLoss(pc1, pc2=None, dist_mode='knn_points', K=K, max_radius=2, loss_norm=1, smooth_weight=0., sm_normals_K=4, forward_weight=0., pc2_smooth=False)
            self.kNN = l.NN_pc1
        else:
            dist, self.kNN, _ = knn_points(pc1, pc1, lengths1=None, lengths2=None, K=K, return_nn=True)

        self.src_keypts = pc1[:, self.kNN[:, :, :]]

    def forward(self, flow):

        target_keypts = self.src_keypts + flow[:, self.kNN[:, :, :]]
        target_keypts = target_keypts[0, 0]
        src_keypts = self.src_keypts[0, 0]

        src_dist = (src_keypts[:, :, None, :] - src_keypts[:, None, :, :]).norm(dim=-1)
        target_dist = (target_keypts[:, :, None, :] - target_keypts[:, None, :, :]).norm(dim=-1)
        cross_dist = (src_dist - target_dist).abs()
        A = torch.clamp(1.0 - cross_dist ** 2 / self.d_thre ** 2, min=0)
        self.A = A

        leading_eig = sc_utils.power_iteration(A)
        # 
        self.leading_eig = leading_eig
        score = sc_utils.spatial_consistency_score(A, leading_eig)
        loss = - torch.log(score).mean()

        return loss


class MBSC(torch.nn.Module):
    # Implementation of https://github.com/kavisha725/MBNSF/
    # Spatial-consistency isometry based on DBSCAN clusters with fixed parameters
    def __init__(self, pc1, eps=0.8, min_samples=30):
        super().__init__()
        self.pc1 = pc1
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pc1[0].detach().cpu().numpy())
        self.labels_t = torch.from_numpy(labels).float().clone().to(pc1.device)
        self.label_ids = torch.unique(self.labels_t)[1:]  # Ignore unclustered points with label = -1.
        self.num_clusters = len(self.label_ids)

    def forward(self, flow):

        pc1_deformed = self.pc1 + flow
        loss_sc = torch.zeros([1, 1], dtype=self.pc1.dtype, device=self.pc1.device, )
        for id in self.label_ids:
            cluster_ids = self.labels_t == id
            num_cluster_points = torch.count_nonzero(cluster_ids)
            if num_cluster_points > 2:
                cluster = self.pc1[0, cluster_ids]
                cluster_deformed = pc1_deformed[0][cluster_ids]
                assert cluster.shape == cluster_deformed.shape
                cluster_cs_loss = sc_utils.spatial_consistency_loss(cluster.unsqueeze(0), cluster_deformed.unsqueeze(0),
                                                           d_thre=0.03)
                loss_sc += cluster_cs_loss
        loss_sc /= self.num_clusters
        loss_sc = loss_sc.squeeze()
        loss_sc = 1 * loss_sc

        return loss_sc


class GeneralLoss(torch.nn.Module):

    # use normals to calculate smoothness loss
    def __init__(self, pc1, pc2=None, dist_mode='DT', cell_size=0.1, K=12, sm_normals_K=0, smooth_weight=1., max_radius=2, loss_norm=1, forward_weight=0., pc2_smooth=False, **kwargs):

        super().__init__()
        self.K = K
        self.max_radius = max_radius
        self.pc1 = pc1
        self.pc2 = pc2
        self.normals_K = sm_normals_K
        self.loss_norm = loss_norm
        self.smooth_weight = smooth_weight
        self.dist_mode = dist_mode

        # normal Smoothness
        if K > 0:

            if self.normals_K > 3:
                self.dist1, self.NN_pc1, _ = self.KNN_with_normals(pc1)
            else:
                self.dist1, self.NN_pc1, _ = knn_points(self.pc1, self.pc1, K=self.K)

            self.NN_pc1 = mask_NN_by_dist(self.dist1, self.NN_pc1, max_radius)

        # ff
        self.forward_weight = forward_weight

        # ff with NNpc2
        self.pc2_smooth = pc2_smooth
        self.NN_pc2 = None

        if pc2_smooth and K > 0:

            if self.normals_K > 3:
                self.dist2, self.NN_pc2, _ = self.KNN_with_normals(pc2)
            else:
                self.dist2, self.NN_pc2, _ = knn_points(self.pc2, self.pc2, K=self.K)

            self.NN_pc2 = mask_NN_by_dist(self.dist2, self.NN_pc2, max_radius)

            # if VA:
            #     self.NN_pc2 = self.Visibility_pc2.visibility_aware_smoothness_KNN(self.NN_pc2).unsqueeze(0)

        if dist_mode == 'DT':
            self.f_DT = DT(pc1, pc2, grid_factor= int(1 / cell_size))
            self.b_DT = DT(pc2, pc1, grid_factor= int(1 / cell_size))
        elif dist_mode == 'FastNN':
            self.DT = FastNN(pc1, pc2, cell_size=cell_size)
        elif dist_mode == 'knn_points':
            pass
        else:
            raise NotImplementedError
    def forward(self, pc1, est_flow, pc2):

        # loss = torch.tensor(0, dtype=torch.float32, device=pc1.device)

        # chamfer
        if self.dist_mode == 'DT':
            f_dist_loss, f_per_point = self.f_DT.torch_bilinear_distance(pc1 + est_flow)
            # b_dist_loss, b_per_point = self.b_DT.torch_bilinear_distance(pc1 + est_flow)
            dist_loss = f_per_point[f_per_point < self.max_radius].mean()


        elif self.dist_mode == 'FastNN':
            dist_loss, forward_nn = self.DT(pc1 + est_flow)

        elif self.dist_mode == 'knn_points':
            forw_dist, forward_nn, _ = knn_points(pc1 + est_flow, pc2, lengths1=None, lengths2=None, K=1, norm=1)
            back_dist, backward_nn, _ = knn_points(pc2, pc1 + est_flow, lengths1=None, lengths2=None, K=1, norm=1)

            dist_loss = (forw_dist.mean() + back_dist.mean()) / 2

        loss = dist_loss

        if self.smooth_weight > 0:

            smooth_loss, pp_smooth_loss = self.smoothness_loss(est_flow, self.NN_pc1, self.loss_norm)

            loss += self.smooth_weight * smooth_loss

        if self.forward_weight > 0:
            forward_loss, pp_forward_loss = self.forward_smoothness(pc1, est_flow, pc2, forward_nn=forward_nn)

            loss += self.forward_weight * forward_loss

        return loss

    def KNN_with_normals(self, pc):

        normals = estimate_pointcloud_normals(pc, neighborhood_size=self.normals_K)
        pc_with_norms = torch.cat([pc, normals], dim=-1)

        return knn_points(pc_with_norms, pc_with_norms, K=self.K)

    def smoothness_loss(self, est_flow, NN_idx, loss_norm=1, mask=None):

        bs, n, c = est_flow.shape
        est_flow_neigh = est_flow.reshape(-1, c)[NN_idx.reshape(-1, NN_idx.shape[2])]
        flow_diff = est_flow_neigh[:, :1, :] - est_flow_neigh[:, 1:, :]
        # est_flow_neigh = est_flow_neigh[:, 1:, :]   # drop identity to ease computation
        smooth_flow_per_point = flow_diff.norm(dim=2)
        smooth_flow_loss = smooth_flow_per_point.mean()

        return smooth_flow_loss, smooth_flow_per_point


    def forward_smoothness(self, pc1, est_flow, pc2, forward_nn=None):

        if forward_nn is None:
            _, forward_nn, _ = knn_points(pc1 + est_flow, pc2, lengths1=None, lengths2=None, K=1, norm=1)

        a = est_flow[0]

        ind = forward_nn[0] # more than one?

        if pc1.shape[1] < pc2.shape[1]:
            shape_diff = pc2.shape[1] - ind.shape[0] + 1 # one for dummy    # what if pc1 is bigger than pc2?
            a = torch.nn.functional.pad(a, (0,0,0, shape_diff), mode='constant', value=0)
            a.retain_grad() # padding does not retain grad, need to do it manually. Check it

            ind = torch.nn.functional.pad(ind, (0,0,0, shape_diff), mode='constant', value=pc2.shape[1])  # pad with dummy not in orig

        # storage of same points
        vec = torch.zeros(ind.shape[0], 3, device=pc1.device)

        # this is forward flow withnout NN_pc2 smoothness
        vec = vec.scatter_reduce_(0, ind.repeat(1,3), a, reduce='mean', include_self=False)

        forward_flow_loss = torch.nn.functional.mse_loss(vec[ind[:,0]], a, reduction='none').mean(dim=-1)

        if self.pc2_smooth:
            # rest is pc2 smoothness with pre-computed NN
            keep_ind = ind[ind[:,0] != pc2.shape[1] ,0]

            # znamena, ze est flow body maji tyhle indexy pro body v pc2 a ty indexy maji mit stejne flow.
            n = self.NN_pc2[0, keep_ind, :]

            # beware of zeros!!!
            connected_flow = vec[n] # N x KSmooth x 3 (fx, fy, fz)

            prep_flow = est_flow[0].unsqueeze(1).repeat_interleave(repeats=self.K, dim=1) # correct

            # smooth it, should be fine
            flow_diff = prep_flow - connected_flow  # correct operation, but zeros makes problem

            occupied_mask = connected_flow.all(dim=2).repeat(3,1,1).permute(1,2,0)

            # occupied_mask
            per_flow_dim_diff = torch.masked_select(flow_diff, occupied_mask)

            # per_point_loss = per_flow_dim_diff.norm(dim=-1).mean()
            NN_pc2_loss = (per_flow_dim_diff ** 2).mean()    # powered to 2 because norm will sum it directly

        else:
            NN_pc2_loss = torch.tensor(0.)

        forward_loss = forward_flow_loss.mean() + NN_pc2_loss

        return forward_loss, forward_flow_loss

# class VAChamferLoss(torch.nn.Module):

#     def __init__(self, pc2, fov_up, fov_down, H, W, max_range, pc_scene=None, nn_weight=1, max_radius=2, both_ways=False, free_weight=0, margin=0.001, ch_normals_K=0, **kwargs):
#         super().__init__()
#         self.kwargs = kwargs
#         self.pc2 = pc2
#         self.pc_scene = pc_scene if pc_scene is not None else pc2


#         # todo option of "pushing" points out of the freespace
#         self.fov_up = fov_up
#         self.fov_down = fov_down
#         self.H = H
#         self.W = W
#         self.max_range = max_range
#         self.margin = margin
#         self.free_weight = free_weight

#         # NN component
#         self.normals_K = ch_normals_K
#         self.nn_weight = nn_weight
#         self.nn_max_radius = max_radius
#         self.both_ways = both_ways

#         # torch.use_deterministic_algorithms(mode=True, warn_only=False)  # this ...
#         # pc2_depth, idx_w, idx_h, inside_range_img = range_image_coords(pc2[0], fov_up, fov_down, proj_H, proj_W)

#         # self.range_depth = create_depth_img(pc2_depth, idx_w, idx_h, proj_H, proj_W, inside_range_img)
#         # torch.use_deterministic_algorithms(mode=False, warn_only=False)  # this ...

#     def forward(self, pc1, est_flow, pc2=None):
#         '''

#         Args:
#             pc1:
#             est_flow:

#         Returns:
#         mask whether the deformed point cloud is in freespace visibility area
#         '''
#         # dynamic

#         # assign Kabsch to lonely points or just push them out of freespace?
#         # precompute chamfer, radius
#         chamf_x, chamf_y = self.chamfer_distance_loss(pc1 + est_flow, self.pc2, both_ways=self.both_ways, normals_K=self.normals_K)

#         if self.free_weight > 0:
#             freespace_loss = self.flow_freespace_loss(pc1, est_flow, chamf_x)

#         else:
#             freespace_loss = torch.zeros_like(chamf_x, dtype=torch.float32, device=chamf_x.device)

#         chamf_loss = self.nn_weight * (chamf_x.mean() + chamf_y.mean()) + self.free_weight * freespace_loss.mean()

#         return chamf_loss, freespace_loss


#     def chamfer_distance_loss(self, x, y, x_lengths=None, y_lengths=None, both_ways=False, normals_K=0, loss_norm=1):
#         '''
#         Unique Nearest Neighboors?
#         :param x:
#         :param y:
#         :param x_lengths:
#         :param y_lengths:
#         :param reduction:
#         :return:
#         '''
#         if normals_K >= 3:
#             normals1 = estimate_pointcloud_normals(x, neighborhood_size=normals_K)
#             normals2 = estimate_pointcloud_normals(y, neighborhood_size=normals_K)

#             x = torch.cat([x, normals1], dim=-1)
#             y = torch.cat([y, normals2], dim=-1)


#         x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1, norm=loss_norm)
#         cham_x = x_nn.dists[..., 0]  # (N, P1)
#         # x_nearest_to_y = x_nn[1]

#         if both_ways:
#             y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1, norm=loss_norm)
#             cham_y = y_nn.dists[..., 0]  # (N, P2)
#             # y_nearest_to_x = y_nn[1]
#         else:

#             cham_y = torch.tensor(0, dtype=torch.float32, device=x.device)

#         return cham_x, cham_y

#     def flow_freespace_loss(self, pc1, est_flow, chamf_x):

#         # flow_depth, flow_w, flow_h, flow_inside = self.Visibility.generate_range_coors(pc1 + est_flow)
#         pc2_image_depth = self.Visibility.assign_depth_to_flow((pc1 + est_flow)[0])
#         flow_depth = ((pc1+est_flow)[0]).norm(dim=-1)

#             # use it only for flow inside the image
#         # masked_pc2_depth = self.range_depth[flow_h[flow_inside], flow_w[flow_inside]]
#         compared_depth = pc2_image_depth - flow_depth



#         # if flow point before the visible point from pc2, then it is in freespace
#         # margin is just little number to not push points already close to visible point
#         flow_in_freespace = compared_depth > 0 + self.margin


#         # Indexing flow in freespace
#         # freespace_mask = torch.zeros_like(chamf_x, dtype=torch.bool)[0]
#         # freespace_mask = flow_in_freespace
#         # if repel:
#         freespace_loss = - est_flow[0, flow_in_freespace].norm(dim=-1).mean()

#         # freespace_loss = flow_in_freespace * chamf_x

#         return freespace_loss





# class _old_FastNN(torch.nn.Module):
#     def __init__(self, max_radius=2, max_z_diff=0.2, cell_size=0.1, max_range=200, device='cuda:0'):
#         super().__init__()

#         with torch.no_grad():
#             cell_size = torch.tensor(cell_size, device=device)
#             self.cell_size = cell_size
#             voxel_size = (cell_size, cell_size, cell_size)
#             self.voxel_size = voxel_size

#             range_x, range_y, range_z = max_range, max_range, max_range / 4
#             self.size = (int(range_x / voxel_size[0]), int(range_y / voxel_size[1]), int(range_z / voxel_size[2]))
#             self.mid = torch.tensor([self.size[0] / 2, self.size[1] / 2, self.size[2] / 2], device=device, dtype=torch.long)
#             self.voxel_grid = - torch.ones(self.size, device=device, dtype=torch.long)

#             self.max_radius = torch.tensor(max_radius, device=device)
#             self.max_z_diff = torch.tensor(max_z_diff, device=device)
#             self.device = device
#             inter = int(max_radius / cell_size) + 1
#             # inter_z = int(max_radius_z / cell_size) + 1

#             max_ring_idx = int(max_radius / cell_size) + 1

#             added_balls_x = torch.linspace(-max_radius - 1, max_radius + 1, 5 * inter, device=device)
#             added_balls_y = torch.linspace(-max_radius - 1, max_radius + 1, 5 * inter, device=device)
#             added_balls_z = torch.linspace(-max_radius - 1, max_radius + 1, 5 * inter, device=device)

#             ball_points = torch.cartesian_prod(added_balls_x, added_balls_y, added_balls_z)
#             ball_dist = ball_points.norm(dim=1)
#             ball_points = torch.cat((ball_points, ball_dist.unsqueeze(1)), dim=1)

#             dist_mask = ball_points[:, -1] <= max_radius

#             ball_points = ball_points[dist_mask]
#             subsampling_voxel = torch.zeros((500, 500, 100), device=device)
#             subsampling_voxel[(ball_points[:, 0] / cell_size + 250).to(torch.long),
#             (ball_points[:, 1] / cell_size + 250).to(torch.long),
#             (ball_points[:, 2] / cell_size + 50).to(torch.long)] = 1

#             survived_pts = subsampling_voxel.nonzero()
#             survived_pts = (survived_pts - torch.tensor([250, 250, 50], device=device)) * cell_size
#             dist_survived = survived_pts.norm(dim=1)

#             processed_ball_points = torch.cat(
#                     (survived_pts, dist_survived.unsqueeze(1), torch.zeros_like(dist_survived.unsqueeze(1))), dim=1)

#             self.idx_dict = {}
#             for dist_idx, dist in enumerate(torch.unique(processed_ball_points[:, 3])):
#                 mask = (processed_ball_points[:, 3] == dist) & (torch.abs(processed_ball_points[:, 2]) < max_z_diff)
#                 processed_ball_points[mask, 4] = dist_idx

#                 add_ball = torch.cat(
#                         (processed_ball_points[mask, :3], torch.zeros_like(processed_ball_points[mask, :1])), dim=1)
#                 self.idx_dict[dist_idx] = add_ball

#             del subsampling_voxel

#     def update_pointcloud(self, points):

#         del self.voxel_grid

#         with torch.no_grad():
#             self.voxel_grid = - torch.ones(self.size, device=self.device, dtype=torch.long)
#             pc = points
#             pc = torch.cat((pc, torch.zeros_like(pc[..., :1])), dim=2)
#             pc[..., -1] = torch.arange(len(pc[0]))  # assign index

#             for dist_idx in range(len(self.idx_dict) - 1, -1, -1):
#                 ball_points = self.idx_dict[dist_idx]

#                 # print(pc.shape, ball_points.shape)
#                 # broadcast pc and ball_points

#                 extended_pc = pc[0] + ball_points.unsqueeze(1)

#                 # ball_points to voxel coordinates
#                 # ball_points = (ball_points / cell_size).to(torch.long)
#                 # full_pc = torch.cat((extended_pc, ))
#                 full_pc = extended_pc.view(-1, 4)
#                 # print(full_pc.shape, self.mid.shape)
#                 full_pc[:, :3] = (full_pc[:, :3] / self.cell_size)

#                 # print(full_pc[:, :3].min(), full_pc[:, :3].max())

#                 # transfer full_pc point cloud to voxel grid cells
#                 # full_coors = torch.stack((full_pc[..., 0] / cell_size, full_pc[..., 1] / cell_size, full_pc[..., 2] / cell_size, torch.full_like(full_pc[..., 2],1)), dim=2).to(torch.long)
#                 # full_coors = full_coors.reshape(-1, 4)
#                 idx = full_pc[..., :].to(torch.long)
#                 idx[:, :3] += self.mid
#                 # if len(idx) > 0:
#                 #     print(idx[:,:3].max(), idx[:,:3].min())

#                 self.voxel_grid[idx[:, 0], idx[:, 1], idx[:, 2]] = idx[:, 3]

#             # del full_pc, idx, extended_pc#, self.idx_dict
#             # torch.cuda.empty_cache()

#     def forward(self, pc1, pred_flow, pc2):
#         with torch.no_grad():
#             flow_voxel_indices = ((pc1 + pred_flow) / self.cell_size).to(torch.long) + self.mid
#             point_mask = (flow_voxel_indices[..., 0] >= 0) & (flow_voxel_indices[..., 0] < self.size[0]) & (
#                          flow_voxel_indices[..., 1] >= 0) & (flow_voxel_indices[..., 1] < self.size[1]) & (
#                          flow_voxel_indices[..., 2] >= 0) & (flow_voxel_indices[..., 2] < self.size[2])

#             point_mask = point_mask[0]
#             # print(point_mask.shape, flow_voxel_indices.shape)

#             NN_ind = - torch.ones((pc1.shape[1]), device=self.device, dtype=torch.long)
#             NN_ind[point_mask] = self.voxel_grid[flow_voxel_indices[0, point_mask, 0], flow_voxel_indices[0, point_mask, 1], flow_voxel_indices[0, point_mask, 2]]

#             # Do not forget about the mask
#             mask = (NN_ind != -1)
#             # for bs = 1 for now
#         # dist = torch.zeros_like(pred_flow[0, :, 0], device=self.device)
#         dist = (pc2[0, NN_ind[mask]] - (pc1[0][mask] + pred_flow[0][mask])).norm(dim=1)

#         return dist, NN_ind
