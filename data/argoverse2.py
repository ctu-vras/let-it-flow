import glob
import numpy as np
import os
import yaml

with open('config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

def unpack_one_frame(dataset, i):
    ''' 
    Unpacks one frame from the Argoverse SceneFlow dataset.
    
    Args:
        dataset (Dataset): The dataset containing the frames.
        i (int): The index of the frame to unpack.
        
    Returns:
        dict: A dictionary containing the unpacked frame data.
    '''
    
    data = dataset[i]

    # lidar1 = data[0]
    pc1 = data[0].lidar.as_tensor().detach().cpu().numpy()
    ground1 = data[0].is_ground.detach().cpu().numpy()

    # lidar2 = data[1]
    pc2 = data[1].lidar.as_tensor().detach().cpu().numpy()
    ground2 = data[1].is_ground.detach().cpu().numpy()

    uuid1 = data[0].sweep_uuid[0]
    uuid2 = data[1].sweep_uuid[0]
    timestamp = data[0].sweep_uuid[1]

    if uuid1 != uuid2: 
        print('uuid mismatched')

    relative_pose = data[2].matrix().detach().cpu().numpy()[0]	# batch

    pose1 = data[0].city_SE3_ego.matrix()   # compute it w.r.t ego position in first frame
    pose2 = data[1].city_SE3_ego.matrix()

    flow = data[3].flow.detach().cpu().numpy()
    flow_valid = data[3].is_valid.detach().cpu().numpy()
    category_indices = data[3].category_indices.detach().cpu().numpy()
    dynamic = data[3].is_dynamic.detach().cpu().numpy()
    class_names = data[0].cuboids.category

    d_dict = {'pc1' : pc1,
                'pc2' : pc2,
                'pose1' : pose1,
                'pose2' : pose2,
                'relative_pose' : relative_pose,            
                'ground1' : ground1,
                'ground2' : ground2,
                'flow' : flow,
                'flow_valid' : flow_valid,
                'dynamic' : dynamic,
                'category_indices' : category_indices,
                'uuid1' : uuid1,
                'uuid2' : uuid2,
                'class_names' : class_names
                
                }

    # transform to second point cloud, pose is relative pc1 -> pc2
    sync_pts = np.dot(relative_pose[:3, :3], pc1[:,:3].T).T + relative_pose[:3, 3]
    sync_pts = np.concatenate((sync_pts, pc1[:, 3:4]), axis=1)
    
    return d_dict

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