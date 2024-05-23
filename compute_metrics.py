import numpy as np
import os
import sys
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from ops.metric.sceneflow import ThreewayFlowMetric

model = str(sys.argv[1])
class_remap = {'Background' : 0,
               'Road Sign?' : 5,
               'Truck' : 6,
               'Cyclist' : 15,
               'Pedestrian' : 17,
               'Vehicle' : 19,
               'IDK' : 28,
               'Overall' : None}

foreground_cls = ['Pedestrian', 'Cyclist', 'Vehicle', 'Truck']
background_cls = ['Background']

metric_dict = {k : ThreewayFlowMetric() for k in class_remap.keys()}
metric_dict["Overall"] = ThreewayFlowMetric()


with open('config.yaml') as file:    
    cfg = yaml.load(file, Loader=yaml.FullLoader)['cfg']

exp_folder = cfg['store_path'] + f'/{model}/'
files = sorted(glob.glob(exp_folder + '/*.npz'))
   

for file in files:
   data = np.load(file, allow_pickle=True)

   p1 = data['p1'][0]
   p2 = data['p2'][0]
   f1 = torch.from_numpy(data['f1'])
   dynamic = torch.from_numpy(data['dynamic'])
   category_indices = torch.from_numpy(data['category_indices'])
   gt_flow = torch.from_numpy(data['compensated_gt_flow'][None])  # correct

   
   foreground_points = [category_indices == class_remap[f] for f in foreground_cls]#.astype(bool)
   background_points = [category_indices == class_remap[b] for b in background_cls]#.astype(bool)

   foreground_points = torch.stack(foreground_points, dim=0).any(dim=0)
   background_points = torch.stack(background_points, dim=0).any(dim=0)
   
   ### STORE METRICS
   for k in metric_dict.keys():
      if k == 'Overall': 
      
         dynamic_foreground_mask = dynamic & foreground_points
         static_foreground_mask = ~dynamic & foreground_points # musi patrit do foregroundu
         static_background_mask = ~dynamic & background_points

      else:
         mask = category_indices == class_remap[k]

         dynamic_foreground_mask = dynamic & mask
         static_foreground_mask = ~dynamic & mask # musi patrit do foregroundu
         static_background_mask = ~dynamic & mask

      # Eliminate non-existent metrics
      if k in foreground_cls:
         static_background_mask = torch.zeros_like(static_background_mask)
      elif k in background_cls:
         static_foreground_mask = torch.zeros_like(static_foreground_mask)
         dynamic_foreground_mask = torch.zeros_like(dynamic_foreground_mask)

      # Split overall into foreground and background

      metric_dict[k].update(f1, gt_flow, -1, dynamic_foreground_mask, static_foreground_mask, static_background_mask)
   
   
   ### PRINT METRICS
def print_accumulated_metrics(metric_dict):
   for k in metric_dict.keys():
      T_D = metric_dict[k].DFG_metric.get_metric().mean()['EPE']
      T_S = metric_dict[k].SFG_metric.get_metric().mean()['EPE']
      T_B = metric_dict[k].SBG_metric.get_metric().mean()['EPE']

      # break
      Average = (T_D + T_S + T_B) / 3
   
      print(f'Threeway EPE of {k}:',  f"{Average:.3f}", f'{T_D:.3f}', f'{T_S:.3f}', f'{T_B:.3f}')

print_accumulated_metrics(metric_dict=metric_dict)
