import socket
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import time
import torch
import vispy
from vispy import visuals, app
from vispy.scene import SceneCanvas
from vispy.scene.cameras import ArcballCamera, TurntableCamera
from vispy.scene.visuals import Arrow

from data.PATHS import VIS_PATH
from ops.boxes import connect_3d_corners



def retype_for_numpy_store(data_dict):
    store_data = {}

    for k,v in data_dict.items():
        
        if type(data_dict[k]) is list:
            # print(k)

            store_data[k] = []
            for idx in range(len(data_dict[k])):
                
                if type(data_dict[k][idx]) is torch.Tensor:
                    store_data[k].append(data_dict[k][idx].detach().cpu().numpy())

                else:
                    store_data[k].append(data_dict[k][idx])


        if type(data_dict[k]) is torch.Tensor:
            
            store_data[k] = v.detach().cpu().numpy()
            # print(k)

        if type(data_dict[k]) is np.ndarray:
            store_data[k] = v

    return store_data

def vis_data(data_dict=None, path='000000.npz'):
    
    store_dict = retype_for_numpy_store(data_dict)
    # if socket.gethostname().startswith("Pat") and data_dict is None:
        # Vis_function()
    # if type(data_dict['pts'][0]) is torch.Tensor:
        # tmp = [i.detach().cpu().numpy() for i in data_dict['pts']]
        # data_dict['pts'] = tmp
    
    folder_path = VIS_PATH + '/tmp_vis/'
    # file = f'{time.time()}_cur.npz'
    file = path
    command = 'visualize_points3D'
    
    np.savez(folder_path + '/' + file, **store_dict)

def wait_for_visuals():
        # tmp_folder = os.path.expanduser("~") + '/rci/data/tmp_vis/'
    tmp_folder = os.path.expanduser("~") + '/cmp/visuals/tmp_vis/'

    # command = 'visualize_points3D'
    # print(getattr(sys.modules[__name__], command))

    # Clean tmp folder before starting
    files = glob.glob(tmp_folder + '/*.npz') + glob.glob(tmp_folder + '/*.png')

    for file in files:
        os.remove(file)

    print('waiting for files in folder ', tmp_folder, ' ...')


    while True:
        time.sleep(0.8)

        files = glob.glob(tmp_folder + '/*.npz') + glob.glob(tmp_folder + '/*.png')

        _ = os.stat(tmp_folder).st_mtime    # to refresh the cache

        if len(files) > 0:

            for file in files:
                print('Loading file: ', file)


                if file.endswith('.npz'):
                    data_dict = np.load(file, allow_pickle=True)
                    Vis_function(data_dict)
                    # command = data['command'].item()

                    # kwargs = {name: data[name] for name in data.files if name not in ['command']}

                    # if 'multiple' in command:
                        # pcs = [data['args'][i] for i in range(len(data['args']))]
                        # breakpoint()
                        # visualize_multiple_pcls(*pcs)
                    # else:
                        # getattr(sys.modules[__name__], command)(**kwargs)

                elif file.endswith('.png'):
                    from PIL import Image
                    img = Image.open(file)
                    img.show()

                else:
                    raise ValueError('Unknown file type: ', file)

                os.remove(file)






class Visual_PCL_Template:
    """Class that creates and handles a visualizer for a pointcloud"""

    def __init__(self):
        pass

    def reset(self):
        """ Reset. """
        # last key press (it should have a mutex, but visualization is not
        # safety critical, so let's do things wrong)
        self.action = "no"  # no, next, back, quit are the possibilities

        # new canvas prepared for visualizing data
        self.canvas = SceneCanvas(keys='interactive', size=(1024, 800), show=True, bgcolor=(0.9, .9, .9))
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()

        # laserscan part
        self.scan_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.scan_view, 0, 0)
        self.scan_vis = vispy.scene.visuals.Markers()
        
        # self.scan_view.camera = 'turntable'
        self.scan_view.camera = TurntableCamera(
        fov=45, distance=20, interactive=True, parent=self.canvas.scene)

        self.scan_view.add(self.scan_vis)
        vispy.scene.visuals.XYZAxis(parent=self.scan_view.scene)

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()

        vispy.app.quit()

    def run(self):
        vispy.app.run()



class Vis_function(Visual_PCL_Template):
    def __init__(self, data_dict=None, function=None, config=None, offset=0):
        super().__init__()
        self.data = data_dict
        self.function = function

        # self.original_config = config.copy()
        # self.config = config

        self.offset = offset
        self.total = 100 # len(dataset)

        # value init
        # self.keys = list(self.config.keys())
        self.keys = ['pts']
        self.flow_value = 0
        self.flow_features = ['pred_flow', 'gt_flow']
        self.cur_key = 0
        # self.cur_value = self.config[self.keys[self.cur_key]]
        self.cur_value = 0

        self.reset()
        self.update_scan()
        self.run()

    def refresh_files(self):
        tmp_folder = VIS_PATH + '/tmp_vis/'

        # command = 'visualize_points3D'
        # print(getattr(sys.modules[__name__], command))
        # Clean tmp folder before starting
        _ = os.stat(tmp_folder).st_mtime    # to refresh the cache
        files = glob.glob(tmp_folder + '/*.npz') + glob.glob(tmp_folder + '/*.png')
        self.files = sorted(files)
        self.total = len(self.files)
        
        # wait for files
        if len(self.files) == 0:
            time.sleep(0.2)

            self.refresh_files()
        # for file in files:
            # os.remove(file)

        # print('waiting for files in folder ', tmp_folder, ' ...')


    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def update_scan(self):
        
        self.refresh_files()

        # if len(self.files) == 0:
        #     data = {'pts' : np.random.rand(10,3)}
        #     idx_color = np.zeros(10)
        
        # else:

        data = np.load(self.files[self.offset], allow_pickle=True)
        # data = self.dataset.get_multi_frame(self.offset)
        
        
        # print(data.files)
        
        feature_name = 'id_mask'

        feature = data[feature_name]
        # for keys in data.files:
            # setattr(self, data[keys], keys)

        # if type(data['pts']) is list or data['pts'].dtype == 'O': # "O" is object dtype in numpy as object
            # pts = np.concatenate([p for p in data['pts']])
            # idx_color = np.concatenate([idx * np.ones(p.shape[0]) for idx, p in enumerate(data['pts'])])

        # else:
            # pts = data['pts'] 
            # idx_color = np.zeros(len(pts))
        # self.config[self.keys[self.cur_key]] = self.cur_value
        
        range_data = data['pts'][:,2]
        range_data = range_data / range_data.max()


        # then change names
        title = f"{self.keys[self.cur_key]} \t ---> {self.cur_value:.2f} \t Scan {self.offset} out of {self.total}"
        self.canvas.title = title


        # print(f"\r \n{self.keys[self.cur_key]}\n \t {self.cur_value:.2f}", end='')

        # print_str = [f"{self.keys[i]} \t {self.config[self.keys[i]]}\n" for i in range(len(self.keys))]
        # text_to_print = "".join(print_str)
        # print(f'\r{text_to_print}', end='')
        # print(self.config)

        # pts -= pts.mean(0)  # for global?
        
        # id_mask = np.random.randint(0,256, len(data['points']))
        cmap = plt.cm.jet  # define the colormap

        
        # arrow2 = np.array([(0, 0, 0, -1, -0.5, 1)])  # Arrow direction, position
        # arr = Arrow(pos=np.array([(5, 5, 5), (-1, -0.5, 1)]), color='green', method='gl', width=5., arrows=arrow2,
                    # arrow_type="angle_30", arrow_size=5.0, arrow_color='blue', antialias=True, parent=self.)

        # normalized_colors = (feature - feature.min()) / (feature.max() - feature.min())
        normalized_colors = feature
        colors = cmap(normalized_colors.astype('float'))

        # green flow
        if 'flow' in data.files or 'gt_flow' in data.files or 'pred_flow' in data.files:
            all_rays = []
            flow_mask = np.linalg.norm(data[self.flow_features[self.flow_value]][:,:3], axis=1) > 0.05
            for x in range(1, int(20)):
                ray_points = data['pts'][flow_mask] + (data[self.flow_features[self.flow_value]][:, :3][flow_mask]) * (x / int(20))
                all_rays.append(ray_points)

            all_rays = np.concatenate(all_rays).reshape(-1, 3)[:,:3]
            flow_color = np.ones((all_rays.shape[0], 4)) * np.array((0,1,0,1))
            flow_size = np.ones(all_rays.shape[0])
            # data['pts'] = all_rays

            pts_size = np.ones(data['pts'].shape[0]) * 3

            pts = np.concatenate((data['pts'][:,:3], all_rays))
            colors = np.concatenate((colors, flow_color))
            pts_size = np.concatenate((pts_size, flow_size))


        if 'boxes' in data.files:
            boxes = data['boxes']
            
            box_pts = connect_3d_corners(torch.from_numpy(boxes), fill_points=50)
            pts = np.concatenate((pts, box_pts[:,:3]))
            box_colors = np.ones((box_pts.shape[0], 4)) * np.array((1,0,0,1))
            colors = np.concatenate((colors, box_colors))
            box_pts_size = np.ones(box_colors.shape[0])
            pts_size = np.concatenate((pts_size, box_pts_size))

        # self.scan_vis.set_data(data['pts'][:,:3],
        #                        face_color=colors,
        #                        edge_color=colors,
        #                        size=np.ones(data['pts'].shape[0]) * 3)
        # breakpoint()
        self.scan_vis.set_data(pts,
                                face_color=colors,
                                edge_color=colors,
                                size=pts_size)


    def key_press(self, event):
        self.canvas.events.key_press.block()
        # self.img_canvas.events.key_press.block()

        # if event.key == 'S' and self.metric is not None:
        #     self.metric.print_stats()
        if event.key == 'S':
            im = self.canvas.render(alpha=True, size=(1024, 800))  # here I can adjust
            file_name = os.path.expanduser("~") + '/visuals/tmp_vis/' + os.path.basename(self.files[self.offset])[:-4] + '.png' 
            
            plt.imshow(im)
            plt.axis('off')
            plt.savefig(file_name, dpi=300)
            plt.close()

        if event.key == 'N':
            self.offset += 1
            if self.offset >= self.total:
                self.offset = 0

        elif event.key == 'B':
            self.offset -= 1
            if self.offset < 0:
                self.offset = self.total - 1

        elif event.key == 'F':
            self.flow_value += 1
            if self.flow_value >= len(self.flow_features):
                self.flow_value = 0

        # Shift by 50 frames
        elif event.key == '0':
            self.offset += 50

        elif event.key == '9':
            self.offset -= 50

        elif event.key == '-':
            self.cur_value -= 0.05

        elif event.key == '=':
            self.cur_value += 0.05

        elif event.key == 'k':
            self.cur_key += 1
            if self.cur_key >= len(self.keys):
                self.cur_key = 0

            self.cur_value = self.config[self.keys[self.cur_key]]

        elif event.key == 'r':
            self.config = self.original_config

        self.update_scan()

        if event.key == 'Q' or event.key == 'Escape':
            self.destroy()

if __name__ == "__main__":
    # wait_for_visuals()
    # Clean tmp folder before starting
    _ = os.stat(VIS_PATH + '/tmp_vis/').st_mtime    # to refresh the cache
    files = glob.glob(VIS_PATH + '/tmp_vis/' + '/*.npz') + glob.glob(VIS_PATH + '/tmp_vis/' + '/*.png')

    for file in files:
        os.remove(file)

    Vis_function()