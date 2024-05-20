import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from ops.boxes import connect_3d_corners

def visualize_boxes_and_masks(pc, boxes=None, id_mask=None):
    
    assert id_mask is not None

    plt.close()

    cmap = plt.cm.jet  # define the colormap
    
    cmaplist = [cmap(i) for i in range(cmap.N)]
    
    cmaplist[0] = (.5, .5, .5, 1.0)
    
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    
    bounds = np.linspace(0, int(id_mask.max()), int(id_mask.max()) + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    plt.scatter(pc[0, :, 0].detach().cpu(), pc[0, :, 1].detach().cpu(), c=id_mask.detach().cpu().numpy(), cmap=cmap, norm=norm, s=1)
    
    if boxes is not None:
        vis_boxes = boxes.detach().clone() 
        bbox = vis_boxes.detach().cpu().numpy()
        box_pts = connect_3d_corners(vis_boxes.detach().cpu(), fill_points=50)
        plt.scatter(box_pts[:,0], box_pts[:,1], c=box_pts[:,-1], cmap=cmap, norm=norm, s=1)

        # Ego
        plt.quiver(bbox[0,0], bbox[0,1], 1, 0, color='green', alpha=1, scale=1, angles='xy', scale_units='xy')
        plt.quiver(bbox[0,0], bbox[0,1], 0, 1, color='green', alpha=1, scale=1, angles='xy', scale_units='xy')

        # Boxes
        plt.quiver(bbox[1:,0], bbox[1:,1], bbox[1:,3] / 2 * np.cos(bbox[1:,6]), bbox[1:,3] / 2 * np.sin(bbox[1:,6]), color='teal', alpha=1, scale=1, angles='xy', scale_units='xy')
        plt.quiver(bbox[1:,0], bbox[1:,1], bbox[1:,4] / 2 * (- np.sin(bbox[1:,6])), bbox[1:,4] / 2 * np.cos(bbox[1:,6]), color='orange', alpha=1, scale=1, angles='xy', scale_units='xy')

    return plt

def visualize_flow_and_instances(pc1, flow=None, mask=None):
    plt.close()
    cmap = plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)
    
    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize
    bounds = np.linspace(0, mask.shape[-1], mask.shape[-1] + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    # video rovnou na serveru
    plt.figure(dpi=250)
    plt.scatter(pc1[:, 0], pc1[:, 1], c= mask.argmax(dim=-1), s=1, cmap=cmap, norm=norm, alpha=1)
    
    if flow is not None:
        plt.quiver(pc1[:, 0], pc1[:, 1], flow[:, 0], flow[:, 1], color='green', angles=1, alpha=1, scale=1, scale_units='xy')
    
    plt.axis('equal')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    
    return plt
    