import open3d as o3d
import numpy as np
from matplotlib import cm
import torch
import time


__all__ = [
    'visualize_points3D',
    'visualize_cloud_sequence',
    'visualize_bbox',
    'visualize_bboxes',
    'visualize_poses',
    'map_colors',
]

def map_colors(values, colormap=cm.gist_rainbow, min_value=None, max_value=None):
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    assert callable(colormap) or isinstance(colormap, torch.Tensor)
    if min_value is None:
        min_value = values[torch.isfinite(values)].min()
    if max_value is None:
        max_value = values[torch.isfinite(values)].max()
    scale = max_value - min_value
    a = (values - min_value) / scale if scale > 0.0 else values - min_value
    if callable(colormap):
        colors = colormap(a.squeeze())[:, :3]
        return colors
    # TODO: Allow full colormap with multiple colors.
    assert isinstance(colormap, torch.Tensor)
    num_colors = colormap.shape[0]
    a = a.reshape([-1, 1])
    if num_colors == 2:
        # Interpolate the two colors.
        colors = (1 - a) * colormap[0:1] + a * colormap[1:]
    else:
        # Select closest based on scaled value.
        i = torch.round(a * (num_colors - 1))
        colors = colormap[i]
    return colors


def visualize_points3D(x, value=None, normals=None, min=None, max=None, colormap=cm.jet, vis=True):
    assert x.ndim == 2
    assert x.shape[1] == 3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    if value is not None:
        assert len(value) == len(x)
        if isinstance(value, torch.Tensor):
            value = value.float()
            value = value.detach().cpu().numpy()
        assert isinstance(value, np.ndarray)
        value = np.asarray(value, dtype=float)
        if value.ndim == 2:
            assert value.shape[1] == 3
            colors = value
        elif value.ndim == 1:
            colors = map_colors(value, colormap=colormap, min_value=min, max_value=max)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    if vis:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def visualize_cloud_sequence(clouds, sleep_time=1.0):
    assert isinstance(clouds, list) or isinstance(clouds, np.ndarray)
    assert isinstance(clouds[0], np.ndarray)
    assert clouds[0].ndim == 2
    assert clouds[0].shape[1] == 3

    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(clouds[0])
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(geometry)

    for pt_cloud in clouds:  # pt_clouds are the point cloud data from several .pcf files

        geometry.points = o3d.utility.Vector3dVector(pt_cloud)
        normals = np.random.rand(len(pt_cloud), 3)
        geometry.normals = o3d.utility.Vector3dVector(normals)
        vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(sleep_time)

    vis.destroy_window()

def box_to_vertices(lwh=(1, 1, 1), pose=np.eye(4)):
    l, w, h = lwh
    vertices = np.array([[l / 2, w / 2, h / 2],
                         [l / 2, w / 2, -h / 2],
                         [l / 2, -w / 2, -h / 2],
                         [l / 2, -w / 2, h / 2],
                         [-l / 2, w / 2, h / 2],
                         [-l / 2, w / 2, -h / 2],
                         [-l / 2, -w / 2, -h / 2],
                         [-l / 2, -w / 2, h / 2]], dtype=pose.dtype)
    vertices = pose[:3, :3] @ vertices.T + pose[:3, 3:4]
    vertices = vertices.T
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]])

    return vertices, lines

def visualize_bbox(lwh=(1, 1, 1), pose=np.eye(4), color=(0, 0, 0), vis=True):
    # plot cube vertices as points and connect them with lines
    # lwh: length, width, height
    # pose: (4 x 4) pose of the cube
    vertices, lines = box_to_vertices(lwh, pose)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

    if vis:
        o3d.visualization.draw_geometries([line_set])

    return line_set


def visualize_bboxes(lwhs, poses, colors=None, vis=True):
    # plot multiple boxes
    # lwhs: list of tuples (l, w, h)
    # poses: list of (4 x 4) poses of the boxes
    # colors: list of colors for each box
    if colors is None:
        colors = [(0, 0, 0) for _ in range(len(lwhs))]

    line_sets = []
    for lwh, pose, color in zip(lwhs, poses, colors):
        vertices, lines = box_to_vertices(lwh, pose)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
        line_sets.append(line_set)

    if vis:
        o3d.visualization.draw_geometries(line_sets)

    return line_sets


def visualize_poses(poses, value=None, min=None, max=None, colormap=cm.jet, vis=True):
    assert isinstance(poses, np.ndarray)
    n_poses = len(poses)
    assert poses.shape == (n_poses, 4, 4)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
    line_set.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(n_poses - 1)])
    if value is not None:
        assert isinstance(value, np.ndarray)
        if value.ndim == 2:
            assert value.shape[1] == 3
            colors = value
        elif value.ndim == 1:
            colors = map_colors(value, colormap=colormap, min_value=min, max_value=max)
        assert colors.shape == (n_poses - 1, 3)
        line_set.colors = o3d.utility.Vector3dVector(colors)
    # line_set.colors = o3d.utility.Vector3dVector([np.random.random(3) for _ in range(n_poses - 1)])

    if vis:
        o3d.visualization.draw_geometries([line_set])

    return line_set

# print('here')
if '__main__' == __name__:
    # pcl_list = [np.random.rand(100,3) for i in range(10)]
    # visualize_cloud_sequence(pcl_list, sleep_time=1.0)

    pts = np.random.rand(100,3)
    rgbs = np.random.rand(100,3)
    normals = np.ones((100,3)) * 3
    visualize_points3D(pts, value=rgbs, normals=normals)
    # visualize points sequences the same way
    # boxes can be points?

    # RCI did not see this?
    