import torch
from lietorch import SO3
from itertools import chain

idx = 86 # 18
obj_pts = pc1.view(-1,3)[clusters == idx]
obj_pts_list = [pc1[t][clusters.reshape(len(pc1), -1)[t] == idx] for t in range(len(pc1))]
time_means = torch.stack([pc1[t][clusters.reshape(len(pc1), -1)[t] == idx].mean(dim=0) for t in range(len(pc1))])

time_means[..., 2] = time_means[..., 2].median()    # can be initialized more intelligently

# yaw_angle = torch.atan2(flow[:,:,0], flow[:,:,1] + 1e-6) # T x N    
    
# point_displacements = time_means.diff(dim=0)
# yaws = torch.atan2(point_displacements[..., 0], point_displacements[..., 1] + 1e-6) # smooth yaws


# rot_vectors = torch.zeros(len(time_means), 3, device=pc1.device)
# rot_vectors[:-1, 2] = yaws
# rot_vectors[-1, 2] = rot_vectors[-2, 2] # reassign last as the previous
# rot_vectors.requires_grad_(True)

target_mean = time_means[-1]
# shift_to_target = target_mean - time_means

for i in range(len(obj_pts_list)):
    # obj_pts_list[i] = obj_pts_list[i] - obj_pts_list[i].mean(axis=0)
    # geometrical mean
    geometrical_middle = obj_pts_list[i].max(0)[0] - obj_pts_list[i].min(0)[0]
    obj_pts_list[i] = obj_pts_list[i] - geometrical_middle
    

# translation = time_means + shift_to_target
translation = torch.zeros((len(pc1), 3), device=pc1.device)
translation.requires_grad_(True)

# 1) posunu vsechno do stredu jako init ---> hodila by se inicializace ze sceneflow?
# 2) optimalizuju hledani stredu tak, aby body sedeli na sobe a zaroven by byl stred ve stejnem miste (shape reconstruct)

yaws = torch.zeros((len(time_means), 1), device=pc1.device, requires_grad=True)

icp_optimizer = torch.optim.Adam(chain([translation, yaws]), lr=0.1)

max_iters = 2
# with torch.autograd.set_detect_anomaly(True):
for it in tqdm(range(max_iters)):
    
    rot_vectors = torch.cat((torch.zeros((len(time_means), 2), device=yaws.device), yaws), dim=1)      # constrained icp
    
    R = SO3.exp(rot_vectors)
    R_mat = R.matrix()
    
    loss = 0
    transformed_pts = []

    for t in range(len(pc1) - 1, -1, -1):
        # do not forget to shift to center
        
        pts = obj_pts_list[t] 
        out = (R_mat[t][:3,:3] @ (pts - translation[t]).T).T + translation[t]  # not checked! add translation etc.
        
        dist, _, _ = knn_points(out.unsqueeze(0), obj_pts_list[-1].unsqueeze(0), K=1)

        per_point_dist = dist[0,:,0]
        loss += per_point_dist[per_point_dist < 0.3].mean()    # max correspondence limit
        
        # TODO - initialize with trajectory, diff2, shift directly to center of last pt? global pts
        # TODO - init target as the most dense point cloud? Shape registration ordering; https://arxiv.org/pdf/2210.08061.pdf
        # lots of things still missing
        # grid registration?
        # similar angles
        # if it < 300:    # for init
        loss += 0.5 * rot_vectors[:-1].diff(dim=0).norm(dim=-1).mean()
            # acceleration - not checked!
        loss += (translation[:-1] - translation[:-1].mean(dim=0)).norm(dim=0).mean() #.diff(dim=0).diff(dim=0).norm(dim=-1).mean()
        # loss += translation[:-1].diff(dim=0).diff(dim=0).norm(dim=0).mean()

        if it == max_iters - 1:
            transformed_pts.append(out.detach().cpu().numpy())
        # print(dist.shape)
    # print(loss.item())
    loss.mean().backward()
    
    icp_optimizer.step()
    icp_optimizer.zero_grad()


transformed_pts.append(translation)
# print(translation)

