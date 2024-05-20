import os

import numpy as np
import vispy.scene
from vispy.scene.visuals import Arrow
from vispy.scene import visuals
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# https://vispy.org/gallery/scene/complex_image.html

# this will work
# Make a canvas and add simple view



# visual scenes result scenes. Find worst cases recursively?
# order input list by shit epe

# for multiple?
def gen_flow_visualization(pc1, pc2, flow):

    points = np.concatenate((pc1, pc2), axis=0)
    vis_flow = np.concatenate((flow, np.zeros(pc2.shape)), axis=0)
    all_rays = []

    for x in range(1, 20):
        ray_points = points + ((vis_flow[:, :3]) * (x / 20))
        all_rays.append(ray_points)

    all_rays = np.concatenate(all_rays)

    vis_points = np.concatenate([points, all_rays])
    vis_color = np.concatenate(
            [np.ones((pc1.shape[0], 4)) * (0,0,1,0.8),   # pts color blue
             np.ones((pc2.shape[0], 4)) * (1,0,0,0.8),   # pts color red
             np.ones((all_rays.shape[0], 4)) * (0,1,0,0.8)]    # flow color green
    )

    return vis_points, vis_color


def assign_flow_to_view(view, pc1, pc2, flow):
    vis_points, vis_color = gen_flow_visualization(pc1, pc2, flow)

    scatter = visuals.Markers()
    symbols = np.random.choice(['o'], len(vis_points))
    scatter.set_data(vis_points, edge_width=0, face_color=vis_color, size=5, symbol=symbols)

    view.add(scatter)
    view.camera = 'turntable'  # or try 'arcball'
    axis = visuals.XYZAxis(parent=view.scene)

    return view

def get_img_from_mplfig(fig):
    fig.savefig(os.path.expanduser('~') + '/tmp/fig.png')
    img_mpl = plt.imread(os.path.expanduser('~') + '/tmp/fig.png')

    return img_mpl

# canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
# canvas.size = 600, 600

# grid = canvas.central_widget.add_grid()



# view = grid.add_view(row=0,col=0)
# # view2 = grid.add_view(row=0,col=1)
# # view3 = grid.add_view(row=0,col=2)
# # canvas2 = vispy.scene.SceneCanvas(keys='interactive', show=True)

# view = assign_flow_to_view(view, pc1, pc2, est_flow1)
# # view2 = assign_flow_to_view(view2, pc1, pc2, our_flow1)
# # view3 = assign_flow_to_view(view3, pc1, pc2, gt_flow1)


# interpolation = 'nearest'
# img_data = (255 * np.random.normal(size=(100, 100, 3), scale=2, loc=128)).astype(np.ubyte)

# fig, axes = plt.subplots(1,2, dpi=400)
# axes[0].imshow(img_data)
# axes[1].plot(range(10))
# # fig.tight_layout()

# img_mpl = get_img_from_mplfig(fig)

# arrows = np.concatenate([np.zeros(points.shape), points], axis=1)   # x,y,z,x2,y2,z2


# arr = Arrow(pos=pt, color='teal', method='gl', width=5., arrows=arrow,
#             arrow_type="angle_30", arrow_size=5.0, arrow_color='teal', antialias=True, parent=view.scene)



# image = visuals.Image(img_mpl, interpolation=interpolation,
#                             parent=view3.scene, method='subdivide')
#
# view3.camera = vispy.scene.PanZoomCamera(aspect=1)
# flip y-axis to have correct aligment
# view3.camera.flip = (0, 1, 0)
# view3.camera.set_range()
# view3.camera.zoom(1)





# vispy.app.run()
# canvas2 = vispy.scene.SceneCanvas(keys='interactive', show=True)
# view2 = canvas2.central_widget.add_view()
# generate data
# interpolation = 'nearest'
# img_data = (255 * np.random.normal(size=(100, 100, 3), scale=2, loc=128)).astype(np.ubyte)
#
# image = visuals.Image(img_data, interpolation=interpolation,
#                             parent=view2.scene, method='subdivide')
#
# view2.camera = vispy.scene.PanZoomCamera(aspect=1)
# # flip y-axis to have correct aligment
# view2.camera.flip = (0, 1, 0)
# view2.camera.set_range()
# view2.camera.zoom(1)
#

# import vispy.plot as vp
# from vispy import color
# from vispy.util.filter import gaussian_filter
# import numpy as np
#
# z = np.random.normal(size=(250, 250), scale=200)
# z[100, 100] += 50000
# z = gaussian_filter(z, (10, 10))
#
# fig = vp.Fig(show=False)
# cnorm = z / abs(np.amax(z))
# c = color.get_colormap("hsl").map(cnorm).reshape(z.shape + (-1,))
# c = c.flatten().tolist()
# c=list(map(lambda x,y,z,w:(x,y,z,w), c[0::4],c[1::4],c[2::4],c[3::4]))
#
# #p1 = fig[0, 0].surface(z, vertex_colors=c) # why doesn't vertex_colors=c work?
# p1 = fig[0, 0].surface(z)
# p1.mesh_data.set_vertex_colors(c)
# fig.show()


# one could stop here for the data generation, the rest is just to make the
# data look more interesting. Copied over from magnify.py
# centers = np.random.normal(size=(50, 3))
# indexes = np.random.normal(size=100000, loc=centers.shape[0] / 2,
#                            scale=centers.shape[0] / 3)
# indexes = np.clip(indexes, 0, centers.shape[0] - 1).astype(int)

# scales = 10**(np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
# pos *= scales
# pos += centers[indexes]

# create scatter object and fill in the data


# view2.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation


import numpy as np
from PyQt5 import QtWidgets

from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app

IMAGE_SHAPE = (600, 800)  # (height, width)
CANVAS_SIZE = (800, 600)  # (width, height)
NUM_LINE_POINTS = 200

COLORMAP_CHOICES = ["viridis", "reds", "blues"]
LINE_COLOR_CHOICES = ["black", "red", "blue"]


class MyMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()

        self._controls = Controls()
        main_layout.addWidget(self._controls)
        self._canvas_wrapper = CanvasWrapper()
        main_layout.addWidget(self._canvas_wrapper.canvas.native)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self._connect_controls()

    def _connect_controls(self):
        self._controls.colormap_chooser.currentTextChanged.connect(self._canvas_wrapper.set_image_colormap)
        self._controls.line_color_chooser.currentTextChanged.connect(self._canvas_wrapper.set_line_color)


class Controls(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout()
        self.colormap_label = QtWidgets.QLabel("Image Colormap:")
        layout.addWidget(self.colormap_label)
        self.colormap_chooser = QtWidgets.QComboBox()
        self.colormap_chooser.addItems(COLORMAP_CHOICES)
        layout.addWidget(self.colormap_chooser)

        self.line_color_label = QtWidgets.QLabel("Line color:")
        layout.addWidget(self.line_color_label)
        self.line_color_chooser = QtWidgets.QComboBox()
        self.line_color_chooser.addItems(LINE_COLOR_CHOICES)
        layout.addWidget(self.line_color_chooser)

        layout.addStretch(1)
        self.setLayout(layout)


class CanvasWrapper:
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', size=CANVAS_SIZE, show=True)
        self.grid = self.canvas.central_widget.add_grid()

        self.view = grid.add_view(row=0, col=0)
        

        self.view = assign_flow_to_view(self.view, pc1, pc2, est_flow1)


        self.view_bot = self.grid.add_view(1, 0, bgcolor='#c0c0c0')
        line_data = _generate_random_line_positions(NUM_LINE_POINTS)
        self.line = visuals.Line(line_data, parent=self.view_bot.scene, color=LINE_COLOR_CHOICES[0])
        self.view_bot.camera = "panzoom"
        self.view_bot.camera.set_range(x=(0, NUM_LINE_POINTS), y=(0, 1))

    def set_image_colormap(self, cmap_name: str):
        print(f"Changing image colormap to {cmap_name}")
        self.image.cmap = cmap_name

    def set_line_color(self, color):
        print(f"Changing line color to {color}")
        self.line.set_data(color=color)


def _generate_random_image_data(shape, dtype=np.float32):
    rng = np.random.default_rng()
    data = rng.random(shape, dtype=dtype)
    return data


def _generate_random_line_positions(num_points, dtype=np.float32):
    rng = np.random.default_rng()
    pos = np.empty((num_points, 2), dtype=np.float32)
    pos[:, 0] = np.arange(num_points)
    pos[:, 1] = rng.random((num_points,), dtype=dtype)
    return pos


# if __name__ == "__main__":
#     app = use_app("pyqt5")
#     app.create()
#     win = MyMainWindow()
#     win.show()
#     app.run()

if __name__ == '__main__':
    import sys

    if sys.flags.interactive != 1:
        vispy.app.run()
