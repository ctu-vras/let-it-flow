# Functions in this file are from: https://github.com/ZhiChen902/SC2-PCR
# Implementation taken from: https://github.com/kavisha725/MBNSF/


import torch
import time

def power_iteration(M, num_iterations=10):
    """
    Calculate the leading eigenvector using power iteration algorithm
    Input:
        - M:      [bs, num_corr, num_corr] the compatibility matrix
    Output:
        - solution: [bs, num_corr] leading eigenvector
    """
    leading_eig = torch.ones_like(M[:, :, 0:1])
    leading_eig_last = leading_eig
    for i in range(num_iterations):
        leading_eig = torch.bmm(M, leading_eig) # vectory poscitane po radcich a sloupcich
        leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6) # prumerny displacement
        if torch.allclose(leading_eig, leading_eig_last):
            break
        leading_eig_last = leading_eig
    leading_eig = leading_eig.squeeze(-1)
    return leading_eig

def spatial_consistency_score(M, leading_eig):
    """
    Calculate the spatial consistency score based on spectral analysis.
    Input:
        - M:          [bs, num_corr, num_corr] the compatibility matrix
        - leading_eig [bs, num_corr]           the leading eigenvector of matrix M
    Output:
        - sc_score
    """
    sc_score = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
    sc_score = sc_score.squeeze(-1) / M.shape[1]
    return sc_score

def spatial_consistency_score(M, leading_eig):
    """
    Calculate the spatial consistency score based on spectral analysis.
    Input:
        - M:          [bs, num_corr, num_corr] the compatibility matrix
        - leading_eig [bs, num_corr]           the leading eigenvector of matrix M
    Output:
        - sc_score
    """
    sc_score = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
    sc_score = sc_score.squeeze(-1) / M.shape[1]
    return sc_score

def spatial_consistency_loss(src_keypts, tgt_keypts, d_thre=0.1, max_points = 3000):
    """
    Input:sc_utils
        - src_keypts: [bs, num_corr, 3] 
        - tgt_keypts: [bs, num_corr, 3]
    Output:
        - sc_loss:   [bs, 1], the spatial consistency loss.
    """
    bs, num_corr = src_keypts.shape[0], tgt_keypts.shape[1]

    # (Optional) random sample points
    if num_corr > max_points:
        rand_perm = torch.randperm(num_corr)
        rand_idx = rand_perm[:max_points]

        src_keypts = src_keypts[:, rand_idx, :]
        tgt_keypts = tgt_keypts[:, rand_idx, :]

    # Spatial Consistency Adjacency Matrix
    src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
    target_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
    cross_dist = torch.abs(src_dist - target_dist)
    adj_mat = torch.clamp(1.0 - cross_dist ** 2 / d_thre ** 2, min=0)

    # Spatial Consistency Loss
    # Time problem is number of clusters
    # pi_st = time.time()
    lead_eigvec = power_iteration(adj_mat)
    # print('Power Iteration Time: ', time.time() - pi_st)
    # torch.eigen is slow
    # breakpoint()
    # sc_s_st = time.time()
    sc_score = spatial_consistency_score(adj_mat, lead_eigvec)
    # print('Spatial Consistency Score Time: ', time.time() - sc_s_st)
    sc_loss = -torch.log(sc_score)

    return sc_loss
