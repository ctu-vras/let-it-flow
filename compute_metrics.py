import numpy as np
import os
import glob
import torch
import pandas as pd
from ops.metric.sceneflow import SceneFlowMetric, ThreewayFlowMetric
# Dummy remap
class_remap = {0 : 'Background',
               5 : 'Road Sign?',
               6 : 'Truck',
            #    11 : "Van",
               15 : 'Cyclist',
               17 : 'Pedestrian',
               19 : 'Vehicle',
            #    21 : 'Sign?',
               28 : 'IDK'}

### USe for visuals
# from vis.utils import flow_to_rgb
# flow_KNN = flow_to_rgb(flow[0].detach().cpu().numpy(), flow_max_radius=None, background='bright') / 255.

exp_folder = 'results/argoverse2/'
files = sorted(glob.glob(exp_folder + '/*.npz'))

ThreewayMetric = ThreewayFlowMetric()
PedsFlowMetric = ThreewayFlowMetric()
VehsFlowMetric =  ThreewayFlowMetric()
CycFlowMetric =  ThreewayFlowMetric()
TruckFlowMetric =  ThreewayFlowMetric()

for file in files:
    data = np.load(file, allow_pickle=True)

    p1 = data['p1']
    p2 = data['p2']
    f1 = torch.from_numpy(data['f1'])
    dynamic = torch.from_numpy(data['dynamic'])
    category_indices = torch.from_numpy(data['category_indices'])
    gt_flow = torch.from_numpy(data['compensated_gt_flow'][None])  # correct

    map_dict = {'foreground_cls' : [6, 19, 17, 15], # bicyclist vs bicycle
                'background_cls' : [0],
                }
    
    # pred_flow, gt_flow, eval_time, dynamic_foreground_mask, static_foreground_mask, static_background_mask
    foreground = (category_indices == 6) | (category_indices == 19) | (category_indices == 17) | (category_indices == 15)
    
    dynamic_foreground_mask = dynamic & foreground
    static_foreground_mask = ~dynamic & foreground
    static_background_mask = ~dynamic & category_indices == 0

    # print(gt_flow.shape, f1.shape)
    ThreewayMetric.update(f1, gt_flow, -1, dynamic_foreground_mask, static_foreground_mask, static_background_mask)

    # peds
    peds_mask = (category_indices == 17)
    dynamic_foreground_mask = dynamic & peds_mask
    static_foreground_mask = ~dynamic & peds_mask
    static_background_mask = torch.zeros_like(dynamic, dtype=bool)

    PedsFlowMetric.update(f1, gt_flow, -1, dynamic_foreground_mask, static_foreground_mask, static_background_mask)

    # vehs
    vehs_mask = (category_indices == 19)
    dynamic_foreground_mask = dynamic & vehs_mask
    static_foreground_mask = ~dynamic & vehs_mask
    static_background_mask = torch.zeros_like(dynamic, dtype=bool)

    VehsFlowMetric.update(f1, gt_flow, -1, dynamic_foreground_mask, static_foreground_mask, static_background_mask)

    # cycs
    cycs_mask = (category_indices == 15)
    dynamic_foreground_mask = dynamic & cycs_mask
    static_foreground_mask = ~dynamic & cycs_mask
    static_background_mask = torch.zeros_like(dynamic, dtype=bool)

    CycFlowMetric.update(f1, gt_flow, -1, dynamic_foreground_mask, static_foreground_mask, static_background_mask)

    # trucks
    trucks_mask = (category_indices == 6)
    dynamic_foreground_mask = dynamic & trucks_mask
    static_foreground_mask = ~dynamic & trucks_mask
    static_background_mask = torch.zeros_like(dynamic, dtype=bool)

    TruckFlowMetric.update(f1, gt_flow, -1, dynamic_foreground_mask, static_foreground_mask, static_background_mask)

T_D = ThreewayMetric.DFG_metric.get_metric().mean()['EPE']
T_S = ThreewayMetric.SFG_metric.get_metric().mean()['EPE']
T_B = ThreewayMetric.SBG_metric.get_metric().mean()['EPE']

# break
Average = (T_D + T_S + T_B) / 3


print('Threeway',  f"{Average:.3f}", f'{T_D:.3f}', f'{T_S:.3f}', f'{T_B:.3f}')

print("THREEWAY METRIC ------")
ThreewayMetric.print_metric()

print('\n\n\n')
print("PEDS METRIC ---------")

T_D = PedsFlowMetric.DFG_metric.get_metric().mean()['EPE']
T_S = PedsFlowMetric.SFG_metric.get_metric().mean()['EPE']
# T_B = PedsFlowMetric.SBG_metric.get_metric().mean()['EPE']

Average = (T_D + T_S) / 2   # there is no background for dynamic class

print( 'Peds', f"{Average:.3f}",  "&",f'{T_D:.3f}', "&", f'{T_S:.3f}', end=' ')
# PedsFlowMetric.print_metric()


# print('\n\n\n')
# print("CYCS METRIC ---------")
T_D = CycFlowMetric.DFG_metric.get_metric().mean()['EPE']
T_S = CycFlowMetric.SFG_metric.get_metric().mean()['EPE']
# T_B = PedsFlowMetric.SBG_metric.get_metric().mean()['EPE']

Average = (T_D + T_S) / 2   # there is no background for dynamic class

print( 'Cycs', f"{Average:.3f}",  "&", f'{T_D:.3f}',  "&", f'{T_S:.3f}', end=' ')

# print('\n\n\n')
# print("VEHS METRIC ---------")
T_D = VehsFlowMetric.DFG_metric.get_metric().mean()['EPE']
T_S = VehsFlowMetric.SFG_metric.get_metric().mean()['EPE']
# T_B = PedsFlowMetric.SBG_metric.get_metric().mean()['EPE']

Average = (T_D + T_S) / 2   # there is no background for dynamic class

print( 'Vehs', f"{Average:.3f}", "&", f'{T_D:.3f}',  "&", f'{T_S:.3f}', end=' ')
# VehsFlowMetric.print_metric()


# CycFlowMetric.print_metric()

# print('\n\n\n')
# print("TRUCKS METRIC ---------")
T_D = TruckFlowMetric.DFG_metric.get_metric().mean()['EPE']
T_S = TruckFlowMetric.SFG_metric.get_metric().mean()['EPE']
T_B = PedsFlowMetric.SBG_metric.get_metric().mean()['EPE']

Average = (T_D + T_S) / 2   # there is no background for dynamic class

# print('Truck', f"{Average:.3f}",  "&",f'{T_D:.3f}',  "&",f'{T_S:.3f}')
# TruckFlowMetric.print_metric()
print('\n------\n')


