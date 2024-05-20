import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pptk

def visualize_points3D(points, labels=None, point_size=0.01, **kwargs):

    if type(points) is not np.ndarray:
        points = points.detach().cpu().numpy()

    if type(labels) is not np.ndarray and labels is not None:
        labels = labels.detach().cpu().numpy()

    if labels is None:
        v = pptk.viewer(points[:, :3])
    else:
        v = pptk.viewer(points[:, :3], labels)
    v.set(point_size=point_size, **kwargs)

    return v

def visualize_multiple_pcls(*args, **kwargs):
    p = []
    l = []

    for n, points in enumerate(args):

        p.append(points[:, :3])
        l.append(n * np.ones((points.shape[0])))

    p = np.concatenate(p)
    l = np.concatenate(l)
    v = visualize_points3D(p, l)
    v.set(**kwargs)

def visualize_flow3d(pts1, pts2, frame_flow, flow_mag_thres=0.05, **kwargs):
    # flow from multiple pcl vis
    # valid_flow = frame_flow[:, 3] == 1
    # vis_flow = frame_flow[valid_flow]
    # threshold for dynamic is flow larger than 0.05 m
    dist = np.sqrt((frame_flow[:, :3] ** 2).sum(1))
    mask = dist > flow_mag_thres
    vis_flow = frame_flow[mask]
    vis_pts = pts1[mask, :3]
    # dist_mask = dist > 0.1
    # vis_flow = frame_flow[dist_mask]

    # todo color for flow estimate
    # for raycast
    # vis_pts = pts1[valid_flow, :3]
    # vis_pts = pts1[dist_mask, :3]

    all_rays = []
    # breakpoint()
    for x in range(1, int(20)):
        ray_points = vis_pts + (vis_flow[:, :3]) * (x / int(20))
        all_rays.append(ray_points)

    all_rays = np.concatenate(all_rays)

    visualize_multiple_pcls(*[pts1, all_rays, pts2], **kwargs)



def fw_bw_2D_vis(pts1, pts2, est_flow, gt_flow, save_path='flow.png'):
    # write function for visualization of point cloud with flow. point cloud is set of points and flow is 2D arrows in top down view.
    # Point cloud is transformed into the top down view first, then plot it together.
    # transform point cloud into top down view

    # should you add rigid flow to dynamic flow?

    # matplotlib.use('Agg')

    pts_2D = pts1[:, :2]
    pts2_2D = pts2[:, :2]
    # est_flow = est_flow[:, :2]
    # gt_flow = gt_flow[:, :2]
    # plot point cloud and flow
    fig, ax = plt.subplots(1,2, dpi=300)


    ax[0].plot(pts_2D[:, 0], pts_2D[:, 1], 'b.', alpha=0.3)
    ax[0].plot(pts2_2D[:, 0], pts2_2D[:, 1], 'r.', alpha=0.1)

    ax[0].quiver(pts_2D[:, 0], pts_2D[:, 1], est_flow[:, 0], est_flow[:, 1], color='g', scale=1, units='xy')
    # ax[0].set_aspect('equal', 'box')

    ax[1].plot(pts_2D[:, 0], pts_2D[:, 1], 'b.', alpha=0.3)
    ax[1].plot(pts2_2D[:, 0], pts2_2D[:, 1], 'r.', alpha=0.1)

    ax[1].quiver(pts2_2D[:, 0], pts2_2D[:, 1], gt_flow[:, 0], gt_flow[:, 1], color='g', scale=1, units='xy')
    # ax[1].set_aspect('equal', 'box')
    # plt.quiver(pts_2D[:, 0], pts_2D[:, 1], fw_flow[:, 0], fw_flow[:, 1], color='g', scale=1, units='xy')
    # plt.savefig(save_path)
    plt.show()
    plt.close()
