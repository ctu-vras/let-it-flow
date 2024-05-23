import torch
import numpy as np
import pandas as pd

def KNN_precision(indices_NN, instance_mask, include_first=False):

    instance_NN = instance_mask[indices_NN]

    # TP = 0
    # total_NN = np.multiply(instance_NN.shape[0], instance_NN.shape[1] - 1)
    correct_mask = np.zeros(instance_NN.shape)

    for col in range(0, instance_NN.shape[1]):  # exluding the original NN (always correct)
        correct_mask[:, col] = instance_NN[:, 0] == instance_NN[:, col]

    Precision = np.mean(correct_mask)
    at_least_one_incorrect = correct_mask.all(axis=1) == False

    return Precision, at_least_one_incorrect


def scene_flow_metrics(pred, labels):
    l2_norm = torch.sqrt(torch.sum((pred - labels) ** 2, 2)).cpu()  # Absolute distance error.
    labels_norm = torch.sqrt(torch.sum(labels * labels, 2)).cpu()
    relative_err = l2_norm / (labels_norm + 1e-20)

    EPE3D = torch.mean(l2_norm).item()  # Mean absolute distance error

    # NOTE: Acc_5
    error_lt_5 = torch.BoolTensor((l2_norm < 0.05))
    relative_err_lt_5 = torch.BoolTensor((relative_err < 0.05))
    acc3d_strict = torch.mean((error_lt_5 | relative_err_lt_5).float()).item()

    # NOTE: Acc_10
    error_lt_10 = torch.BoolTensor((l2_norm < 0.1))
    relative_err_lt_10 = torch.BoolTensor((relative_err < 0.1))
    acc3d_relax = torch.mean((error_lt_10 | relative_err_lt_10).float()).item()

    # NOTE: outliers
    l2_norm_gt_3 = torch.BoolTensor(l2_norm > 0.3)
    relative_err_gt_10 = torch.BoolTensor(relative_err > 0.1)
    outlier = torch.mean((l2_norm_gt_3 | relative_err_gt_10).float()).item()

    # NOTE: angle error
    unit_label = labels / labels.norm(dim=2, keepdim=True)
    unit_pred = pred / pred.norm(dim=2, keepdim=True)
    eps = 1e-7
    dot_product = (unit_label * unit_pred).sum(2).clamp(min=-1+eps, max=1-eps)
    dot_product[dot_product != dot_product] = 0  # Remove NaNs
    angle_error = torch.acos(dot_product).mean().item()

    return EPE3D, acc3d_strict, acc3d_relax, outlier, angle_error

class SceneFlowMetric():
    def __init__(self):
        self.epe_list = []
        self.accs_list = []
        self.accr_list = []
        self.angle_list = []
        self.outlier_list = []
        self.time_list = []
        self.metric_list = []

    def update(self, data):
        if data['gt_flow'].shape[-1] == 4:
            gt_mask = data['gt_flow'][:, :, 3] > 0

            self.epe, self.accs, self.accr, self.angle, self.outlier = scene_flow_metrics(data['pred_flow'][gt_mask].unsqueeze(0), data['gt_flow'][gt_mask][..., :3].unsqueeze(0))
        else:
            self.epe, self.accs, self.accr, self.angle, self.outlier = scene_flow_metrics(data['pred_flow'], data['gt_flow'][:,:,:3])

        # update lists
        self.epe_list.append(self.epe), self.accs_list.append(self.accs), self.accr_list.append(self.accr),
        self.angle_list.append(self.angle), self.outlier_list.append(self.outlier), self.time_list.append(data['eval_time'])




    def get_metric(self):

        # Computation of sceneflow metrics
        epe = np.stack(self.epe_list)
        accs = np.stack(self.accs_list) * 100
        accr = np.stack(self.accr_list) * 100
        angle = np.stack(self.angle_list)
        outlier = np.stack(self.outlier_list)
        eval_time = np.stack(self.time_list)
        # set pandas display options for float precision
        pd.set_option("display.precision", 3)
        metric_df = pd.DataFrame([epe, accs, accr, angle, outlier, eval_time], index=['EPE', 'AS', 'AR', 'Angle' , 'Out', 'Eval_Time']).T

        self.metric_list.append(metric_df)

        # as npz file?
        return metric_df

    def store_metric(self, path):
        df = self.get_metric()
        df.to_csv(path)

class ThreewayFlowMetric():

    def __init__(self):    
        self.DFG_metric = SceneFlowMetric()
        self.SFG_metric = SceneFlowMetric()
        self.SBG_metric = SceneFlowMetric()

    def update(self, pred_flow, gt_flow, eval_time, dynamic_foreground_mask, static_foreground_mask, static_background_mask):
        assert pred_flow.shape == gt_flow.shape
        assert pred_flow.shape[0] == 1

        self.DFG_metric.update({'gt_flow': gt_flow[:1, dynamic_foreground_mask], 'pred_flow': pred_flow[:1, dynamic_foreground_mask], 'eval_time' : eval_time})
        self.SFG_metric.update({'gt_flow': gt_flow[:1, static_foreground_mask], 'pred_flow': pred_flow[:1, static_foreground_mask], 'eval_time' : eval_time})
        self.SBG_metric.update({'gt_flow': gt_flow[:1, static_background_mask], 'pred_flow': pred_flow[:1, static_background_mask], 'eval_time' : eval_time})

    def print_metric(self):
        print('Dynamic FG: \n', self.DFG_metric.get_metric().mean())
        print('Static FG: \n', self.SFG_metric.get_metric().mean())
        print('Static BG: \n', self.SBG_metric.get_metric().mean())

    def store_metric(self, path):
        self.DFG_metric.get_metric().mean().to_csv(path + '/DFG.csv')
        self.SFG_metric.get_metric().mean().to_csv(path + '/SFG.csv')
        self.SBG_metric.get_metric().mean().to_csv(path + '/SBG.csv')