import glob
import numpy as np
import os
import yaml

with open('config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

def sample_argoverse2(folder_path, seq : int, cfg):
    
    max_radius = cfg['max_radius']
    max_height = cfg['max_height']
    min_height = cfg['min_height']
    # device = cfg['device']
    sequence_path = sorted(glob.glob(folder_path + '/*/'))

    file_paths = sorted(glob.glob(sequence_path[seq] + '/*.npz'))

    pc_list = []
    global_list = []
    flow_list = []
    gt_list = []
    compensated_gt_flow_list = []
    dynamic_list = []
    category_indices_list = []
    pose_list = []
    global_pose_list = []
    seq_names = []

    
    for t, d_path in enumerate(file_paths):

        data_file = np.load(d_path)

        pc1 = data_file['pc1'][:,:3]
        pc2 = data_file['pc2'][:,:3]
        ground1 = data_file['ground1']
        ground2 = data_file['ground2']
        gt_flow = data_file['flow']
        dynamic = data_file['dynamic']
        category_indices = data_file['category_indices']
        pose = data_file['pose']
        flow_valid = data_file['flow_valid']
        
        


        mask =(~ground1) & (pc1[..., 2] < max_height) & (pc1[..., 2] > min_height) & (np.linalg.norm(pc1, axis=-1) < max_radius) 
        mask = mask.astype(bool)

        mask2 = (~ground2) & (pc2[..., 2] < max_height) & (pc2[..., 2] > min_height) & (np.linalg.norm(pc2, axis=-1) < max_radius)
        mask2 = mask2.astype(bool)

        pc1 = pc1[mask]
        pc2 = pc2[mask2]
        # pose = torch.tensor(pose)
        # pc2 = pc2[mask2].unsqueeze(0).to(device)

        compensated_pc1 = (pose[:3,:3] @ pc1.T).T + pose[:3, -1][None]


        # compensated GT_flow 
        ego_flow = compensated_pc1 - pc1

        gt_flow = gt_flow[mask] 
        compensated_gt_flow = gt_flow - ego_flow

        

        category_indices = category_indices[mask]
        dynamic = dynamic[mask]
        compensated_pc1 = compensated_pc1

        # cur_pose = first_pose @ np.linalg.inv(pose)


        pc_list.append(pc1)
        flow_list.append(gt_flow)
        compensated_gt_flow_list.append(compensated_gt_flow)
        gt_list.append(gt_flow)
        dynamic_list.append(dynamic)
        category_indices_list.append(category_indices)
        pose_list.append(pose)

        seq_names.append(os.path.dirname(d_path).split('/')[-1] + '_' + os.path.basename(d_path).split('.')[0])

    final_poses = []
    
    # construct path from relative poses
    pose = np.eye(4)[None]
    poses = pose.copy()
    for i in range(len(pose_list)):
        pose = pose @ np.linalg.inv(pose_list[i])
        poses = np.concatenate([poses, pose], axis=0)

    for i in range(len(poses) - 1):
        global_pc = (poses[i, :3,:3] @ pc_list[i].T).T + poses[i, :3, -1][None]
        global_pc = np.insert(global_pc, 3, i, axis=-1)
        global_list.append(global_pc)
    
    return global_list, poses, gt_flow, compensated_gt_flow_list, dynamic_list, category_indices_list, seq_names