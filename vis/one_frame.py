# pyline: disable=no-member
""" plot3d using existing visuals : LinePlotVisual """

import numpy as np
import sys

from vispy import app, visuals, scene

# build visuals
Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)

# build canvas
canvas = scene.SceneCanvas(keys='interactive', title='plot3d', show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 6

# prepare data
N = 60
x = np.sin(np.linspace(-2, 2, N)*np.pi)
y = np.cos(np.linspace(-2, 2, N)*np.pi)
z = np.linspace(-2, 2, N)

# plot

data = np.load('/home/patrik/cmp/experiments/SCOOP/2023-07-11-17-11-13-450/inference/000000.npz', allow_pickle=True)
pc1 = data['pc1']
gt_flow = data['gt_flow']


Plot3D(gt_flow, width=2.0, color='green',
       edge_color='w', symbol='>', face_color=(0.2, 0.2, 1, 0.8),
       parent=view.scene)


if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()
