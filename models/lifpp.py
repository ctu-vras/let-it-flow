from vis import *
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader

from data.argoverse2 import unpack_one_frame


dataset_path = "/mnt/personal/vacekpa2/data/"
dataset = SceneFlowDataloader(dataset_path, "argoverse2", "val")

pc_list = []
pose_list = []

sync_pose_list = []
sync_pc_list = []

# TODO, split sequences
for i in range(1000,1010):
    data_dict = unpack_one_frame(dataset, i)
    
    # relative_pose = data_dict['relative_pose']
    pc1 = data_dict['pc1'][data_dict['ground1'] == 0] # TODO split functionality
    pc1 = np.insert(pc1, 3, i, axis=1)
    
    pc_list.append(pc1)
    pose_list.append(data_dict['pose1'])

for i in range(len(pose_list)):
    
    pc1 = pc_list[i]
    pose1 = pose_list[0][0]
    pose2 = pose_list[i][0]
    
    sync_pose = np.dot(np.linalg.inv(pose1), pose2)
    sync_pose_list.append(sync_pose)


    sync_pts = np.dot(sync_pose[:3, :3], pc1[:,:3].T).T + sync_pose[:3, 3]
    sync_pts = np.concatenate((sync_pts, pc1[:, 3:4]), axis=1)
    
    sync_pc_list.append(sync_pts)

visualize_multiple_pcls(*sync_pc_list)
from sklearn.cluster import DBSCAN

to_cluster_pc = np.concatenate(sync_pc_list, axis=0)    # no time so far
to_cluster_pc[:, 3] /= 10  
to_cluster_pc[:, 2] /= 2 
# # TODO scale time
# # TODO init flow with open3d icp?
# # cluster using open3d dbscan

init_ids = DBSCAN(eps=0.3, min_samples=1).fit_predict(to_cluster_pc)

from vis.deprecated_vis import *
idx = init_ids[557877]

# visualize_points3D(to_cluster_pc[:, :3][init_ids==idx], to_cluster_pc[:, 3][init_ids==idx])
# visualize_points3D(to_cluster_pc[:, :3], to_cluster_pc[:, 3])

pcds = to_cluster_pc[:][init_ids==idx]
# Method
# 1) build map from one frame
# 2) ICP to align of instances to the reconstructed map
# 3) Downsample map
# 4) Transfer instances to other frames based on overlaps
# 5) repeat 2-4
# opt) Freespace as place in the map and it deletes the map points?


# Trajectory really seems to be good only for init
# KISS-ICP wont probably work, as the easy cluster is still noise and do not use all times at once
obj_pts_list = []
for i, t in enumerate(np.unique(pcds[:,3])):

#     pcd = o3d.geometry.PointCloud()
    obj_pts_list.append(pcds[pcds[:, 3] == t][:,:3])


# visualize_points3D(to_cluster_pc[:,:3], init_ids)

visualize_multiple_pcls(*obj_pts_list)
visualize_points3D(to_cluster_pc[:,:3], init_ids)
# KISS-ICP
# visualize_multiple_pcls(*obj_pts_list)
# store obj_pts_list as pcd points
# no gpu?
import numpy as np
import open3d as o3d

pts_list = []
orig_mean_list = []

# spatio-temporal pc
# visualize_multiple_pcls(*obj_pts_list)
for i in range(len(obj_pts_list)):
    
    obj_mean = obj_pts_list[i].mean(axis=0) 
    pts = obj_pts_list[i] - obj_mean #.var(axis=0)  # Treat only dynamic this way

    
    pts_list.append(pts)
    orig_mean_list.append(obj_mean)

# shifted by own means
# visualize_multiple_pcls(*pts_list)
trans_list = []

reference_frame = 0 # len(obj_pts_list) - 1

for t in range(0, len(obj_pts_list)):
    if t == reference_frame: 
        trans_list.append(np.eye(4))
        continue
    
    # if t != 4: continue 

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pts_list[t])
    # source.estimate_normals(
    # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn))

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pts_list[reference_frame])   # compute to
    # target.estimate_normals(
    # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn))
    
    threshold = 1.0 # should be same parameter as spatio-temporal clustering?
    trans_init = np.eye(4)  # reinit?
    
    icp_reg = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
                                                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    
    trans_list.append(icp_reg.transformation)
    # print(icp_reg, icp_reg.transformation)


# vis
global_list = [(trans_list[t][:3,:3] @ pts_list[t].T).T + trans_list[t][:3,-1] for t in range(len(trans_list))]

# reinited transforms
from scipy.signal import savgol_filter
# print(trans_list)
x = np.array([i[:3, -1] for i in trans_list], dtype=np.float64)
smoothed_x = savgol_filter(x, 3, 1, axis=0)

# global_list.insert(0, smoothed_x)


reconstructed_poses = x + np.array(orig_mean_list)
smoothed_x = savgol_filter(reconstructed_poses, 4, 2, axis=0)

# try to use smoothed path as initial guess

# visualize_multiple_pcls(*global_list)


# Downsample in open3d
stacked_obj = np.concatenate(global_list, axis=0)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(stacked_obj)
pcd = pcd.voxel_down_sample(voxel_size=0.2)

# print(reconstructed_poses)
# visualize_multiple_pcls(*[np.array(pcd.points)] + obj_pts_list)
# visualize_multiple_pcls(*[np.concatenate(obj_pts_list), reconstructed_poses])

# unroll with reconstructed poses
shape_along_time = [np.asarray(pcd.points) + reconstructed_poses[i] for i in range(len(reconstructed_poses))]

visualize_multiple_pcls(*[np.concatenate(shape_along_time)] + obj_pts_list)
# visualize_multiple_pcls(*[np.concatenate(obj_pts_list), smoothed_x])
# visualize_points3D(np.array(pcd.points), np.ones(len(pcd.points)))

from vis.deprecated_vis import imshow
plt.close()

plt.plot(reconstructed_poses[:,0], reconstructed_poses[:,1], label='orig')
plt.plot(smoothed_x[:,0], smoothed_x[:,1], label='smoothed')

imshow(plt)