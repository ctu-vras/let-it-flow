import numpy as np
from mayavi import mlab
import torch

def visualize_PCA(pc, eigen_vectors):
    figure = mlab.figure(1, bgcolor=(1, 1, 1), size=(640, 480))
    vis_pc = pc.detach().cpu().numpy()
    vis_eigen_vectors = eigen_vectors.detach().cpu().numpy()


    mlab.points3d(0, 0, 0, color=(0, 0, 1), scale_factor=0.3, mode='axes')
    # mlab.points3d(vis_pc[0,:,0], vis_pc[0,:,1], vis_pc[0,:,2], color=(0,0,1), scale_factor=0.1)
    mlab.quiver3d(np.zeros(3), np.zeros(3), np.zeros(3), vis_eigen_vectors[:,0], vis_eigen_vectors[:,1], vis_eigen_vectors[:,2], color=(0,1,0), scale_factor=1)

    mlab.show()

def visualize_flow_frame(pc1, pc2, est_flow1, pose=None, pose2=None, normals1=None, normals2=None):

    figure = mlab.figure(1, bgcolor=(1, 1, 1), size=(640, 480))
    vis_pc1 = pc1.detach().cpu().numpy()
    vis_pc2 = pc2.detach().cpu().numpy()

    vis_est_rigid_flow = est_flow1.detach().cpu().numpy()

    mlab.points3d(0, 0, 0, color=(0, 0, 1), scale_factor=0.3, mode='axes')
    if pose2 is not None:
        mlab.points3d(pose2.detach()[0, 0, -1], pose2.detach()[0, 1, -1], pose2.detach()[0, 2, -1], color=(1, 0, 0), scale_factor=0.3, mode='axes')
    mlab.points3d(vis_pc1[0,:,0], vis_pc1[0,:,1], vis_pc1[0,:,2], color=(0,0,1), scale_factor=0.1)
    mlab.points3d(vis_pc2[0,:,0], vis_pc2[0,:,1], vis_pc2[0,:,2], color=(1,0,0), scale_factor=0.1)
    mlab.quiver3d(vis_pc1[0,:,0], vis_pc1[0,:,1], vis_pc1[0,:,2], vis_est_rigid_flow[0, :,0], vis_est_rigid_flow[0, :,1], vis_est_rigid_flow[0, :,2], color=(0,1,0), scale_factor=1)

    if normals1 is not None:
        vis_normals1 = normals1.detach().cpu().numpy()
        mlab.quiver3d(vis_pc1[0, :, 0], vis_pc1[0, :, 1], vis_pc1[0, :, 2], vis_normals1[0, :, 0], vis_normals1[0, :, 1], vis_normals1[0, :, 2], color=(0, 0, 0.4), scale_factor=0.4)

    if normals2 is not None:
        vis_normals2 = normals2.detach().cpu().numpy()
        mlab.quiver3d(vis_pc2[0,:,0], vis_pc2[0,:,1], vis_pc2[0,:,2], vis_normals2[0, :,0], vis_normals2[0, :,1], vis_normals2[0, :,2], color=(0.4,0,0), scale_factor=0.4)
    mlab.show()

if __name__ == '__main__':
    from traits.api import HasTraits, Range, Instance, \
        on_trait_change
    from traitsui.api import View, Item, HGroup
    from tvtk.pyface.scene_editor import SceneEditor
    from mayavi.tools.mlab_scene_model import \
        MlabSceneModel
    from mayavi.core.ui.mayavi_scene import MayaviScene

    from numpy import linspace, pi, cos, sin




    class Visualization(HasTraits):
        frame = Range(1, 30, 6)
        flow = Range(0, 30, 11)
        scene = Instance(MlabSceneModel, ())

        def __init__(self):
            # Do not forget to call the parent's __init__
            HasTraits.__init__(self)
            # x, y, z, t = curve(self.meridional, self.transverse)
            # self.plot = self.scene.mlab.plot3d(x, y, z, t, colormap='Spectral')
            self.data = np.random.rand(31, 10000, 3) * 10
            init_data = self.data[0]
            self.plot = self.scene.mlab.points3d(init_data[:,0], init_data[:,1], init_data[:,2])

        @on_trait_change('frame, flow')
        def update_plot(self):
            print('Updating')
            cur_data = self.data[self.frame]
            self.plot.mlab_source.set(x=cur_data[:,0], y=cur_data[:,1], z=cur_data[:,2])

        # the layout of the dialog created
        view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                         height=250, width=300, show_label=False),
                    HGroup(
                        '_', 'frame', 'flow',
                    ),
                    )


    visualization = Visualization()
    visualization.configure_traits()


def draw_coord_frame(pose, scale=0.5):
    t, R = pose[:3, 3], pose[:3, :3]
    # draw coordinate frame
    x_axis = torch.tensor([1., 0., 0.], dtype=pose.dtype)
    y_axis = torch.tensor([0., 1., 0.], dtype=pose.dtype)
    z_axis = torch.tensor([0., 0., 1.], dtype=pose.dtype)
    x_axis = R @ x_axis
    y_axis = R @ y_axis
    z_axis = R @ z_axis
    mlab.quiver3d(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], color=(1, 0, 0), scale_factor=scale)
    mlab.quiver3d(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], color=(0, 1, 0), scale_factor=scale)
    mlab.quiver3d(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], color=(0, 0, 1), scale_factor=scale)


def draw_coord_frames(poses, scale=0.1):
    assert poses.ndim == 3
    assert poses.shape[-2:] == (4, 4)

    for pose in poses:
        draw_coord_frame(pose, scale=scale)


def draw_bbox(lwh=(1, 1, 1), pose=np.eye(4), color=(0, 0, 0)):
    # plot cube vertices as points and connect them with lines
    # lwh: length, width, height
    # pose: (4 x 4) pose of the cube
    l, w, h = lwh
    vertices = torch.tensor([[l / 2, w / 2, h / 2],
                             [l / 2, w / 2, -h / 2],
                             [l / 2, -w / 2, -h / 2],
                             [l / 2, -w / 2, h / 2],
                             [-l / 2, w / 2, h / 2],
                             [-l / 2, w / 2, -h / 2],
                             [-l / 2, -w / 2, -h / 2],
                             [-l / 2, -w / 2, h / 2]], dtype=pose.dtype)
    vertices = pose[:3, :3] @ vertices.T + pose[:3, 3:4]
    vertices = vertices.T
    lines = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]])

    mlab.points3d(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=color, scale_factor=0.1)
    for line in lines:
        mlab.plot3d(vertices[line, 0], vertices[line, 1], vertices[line, 2], color=color, tube_radius=0.01)

def draw_bboxes(lwhs, poses, colors=None):
    # plot multiple cubes
    # lwhs: list of tuples (l, w, h)
    # poses: list of (4 x 4) poses of the cubes
    # colors: list of colors for each cube
    if colors is None:
        colors = [(0, 0, 0) for _ in range(len(lwhs))]
    for lwh, pose, color in zip(lwhs, poses, colors):
        draw_bbox(lwh, pose, color)

def draw_cloud(points, rgbs=None, **kwargs):
    # points: (N, 3)
    # rgbs: (N, 3)
    assert points.shape[1] == 3
    if rgbs is not None:
        assert rgbs.shape[1] == 3
        assert points.shape[0] == rgbs.shape[0]

        color_n = np.arange(len(points))
        lut = np.zeros((len(color_n), 4))
        lut[:, :3] = rgbs
        lut[:, 3] = 255

        p3d = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color_n, mode='point', **kwargs)
        p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut)
        p3d.module_manager.scalar_lut_manager.lut.table = lut
    else:
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], **kwargs)

    mlab.show()


if __name__ == '__main__':
    pts = np.random.rand(100,3)
    rgbs = np.random.rand(100,3)
    print('here')
    draw_cloud(pts, rgbs) 