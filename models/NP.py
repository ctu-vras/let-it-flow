# All in Torch
import time
import torch
import torch.nn as nn
from loss.flow import GeneralLoss
from pytorch3d.transforms import euler_angles_to_matrix


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

def construct_transform(rotation_vector: torch.Tensor, translation: torch.Tensor):
    '''
    Construct 4x4 transformation matrix from rotation vector and translation vector while perserving differentiation
    :param rotation_vector:
    :param translation:
    :return: Pose matrix
    '''

    rotation = euler_angles_to_matrix(rotation_vector, convention='XYZ')


    r_t_matrix = torch.hstack([rotation, translation.unsqueeze(1)])

    one_vector = torch.zeros((len(rotation), 4, 1), device=rotation.device)
    one_vector[:, -1, -1] = 1

    pose = torch.cat([r_t_matrix, one_vector], dim=2)


    return pose


class PoseTransform(torch.nn.Module):
    '''
    Pose transform layer
    Works as a differentiable transformation layer to fit rigid ego-motion
    '''

    def __init__(self, BS=1, device='cpu'):
        super().__init__()
        # If not working in sequences, use LieTorch
        self.translation = torch.nn.Parameter(torch.zeros((BS, 3), requires_grad=True, device=device))
        self.rotation_angles = torch.nn.Parameter(torch.zeros((BS, 3), requires_grad=True, device=device))
        

    def construct_pose(self):
        self.pose = construct_transform(self.rotation_angles, self.translation)

        return self.pose

    def forward(self, pc):
        
        pc_to_transform = torch.cat([pc, torch.ones((len(pc), pc.shape[1], 1), device=pc.device)], dim=2)

        pose = construct_transform(self.rotation_angles, self.translation)

        deformed_pc = torch.bmm(pc_to_transform, pose)[:, :, :3]

        return deformed_pc



class ModelTemplate(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model_cfg = self.store_init_params(locals())

        # self.initialize()

    def forward(self, data):
        st = time.time()

        eval_time = time.time() - st
        return data

    def model_forward(self, data):
        return data

    def initialize(self):
        self.apply(init_weights)

    def store_init_params(self, local_variables):
        cfg = {}
        for key, value in local_variables.items():
            if key not in ['self', '__class__', 'args', 'kwargs']:
                setattr(self, key, value)
                cfg[key] = value
            if key == 'kwargs':
                for k, v in value.items():
                    setattr(self, k, v)
                    cfg[k] = v
            if key == 'args':
                setattr(self, 'args', value)
                cfg[key] = value

        return cfg

class NeuralPriorNetwork(torch.nn.Module):
    
    def __init__(self, lr=0.008, early_stop=30, loss_diff=0.001, dim_x=3, filter_size=128, act_fn='relu', layer_size=8, initialize=True,
                 verbose=False, **kwargs):
        super().__init__()
        self.layer_size = layer_size
        bias = True
        self.nn_layers = torch.nn.ModuleList([])

        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size, bias=bias)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size - 1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size, bias=bias)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x, bias=bias))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x, bias=bias)))

        if initialize:
            self.apply(init_weights)

        self.lr = lr
        self.early_stop = early_stop
        self.loss_diff = loss_diff
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.verbose = verbose

    def update(self, pc1=None, pc2=None):
        pass
    
    def forward(self, x):

        for layer in self.nn_layers:
            x = layer(x)

        return x


class PoseNeuralPrior(torch.nn.Module):
    ''' Configurable model with Neural Prior structure '''
    def __init__(self, pc1, pc2=None, eps=0.4, min_samples=5, instances=10, init_transform=0, use_transform=0,
                 ):
        super().__init__()
        self.pc1 = pc1
        self.pc2 = pc2
        self.instances = instances
        self.init_transform = init_transform
        self.use_transform = use_transform

        self.Trans = PoseTransform().to(self.pc1.device)
        if init_transform:
            self.initialize_transform()

        self.FlowModel = NeuralPriorNetwork()


    def forward(self, pc1, pc2=None):

        if self.use_transform == 0:
            final_flow = self.infer_flow(pc1)
            
        elif self.use_transform == 1:
            rigid_flow = self.Trans(pc1) - pc1
            pred_flow = self.infer_flow(pc1)
            final_flow = rigid_flow + pred_flow

        elif self.use_transform == 2:
            deformed_pc1 = self.Trans(pc1)
            rigid_flow = deformed_pc1 - pc1
            pred_flow = self.infer_flow(deformed_pc1)
            final_flow = pred_flow + rigid_flow
        else:
            raise NotImplemented()

        return final_flow

    def infer_flow(self, pc1, pc2=None):

        final_flow = self.FlowModel(pc1)

        return final_flow

    def initialize_transform(self):

        self.Trans = PoseTransform().to(self.pc1.device)
        # Notes:
        # 1) Nechat NN init transformaci
        # 2) Init from Flow model can introduce rotation on KiTTISF making it worse
        trans_iters = 250

        if self.init_transform == 1:
            optimizer = torch.optim.Adam(self.Trans.parameters(), lr=0.03)
            TransLossModule = GeneralLoss(pc1=self.pc1, pc2=self.pc2, dist_mode='DT', K=1, max_radius=2,
                                          smooth_weight=0, forward_weight=0, sm_normals_K=0, pc2_smooth=False)

            for i in range(trans_iters):
                deformed_pc1 = self.Trans(self.pc1)
                rigid_flow = deformed_pc1 - self.pc1
                loss = TransLossModule(self.pc1, rigid_flow, self.pc2)
                # max_points = 5000

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

