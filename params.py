folder_path = 'datasets/argoverse2/'
store_path = 'results/argoverse2/'
device = 'cpu'
nbr_of_devices = 1

cfg = {'frame' : 2,
    'dist_w' : 2.,
    'free_w' : 0.,
    'TEMPORAL_RANGE' : 5,
    'sc_w' : 1.,
    'lr' : 0.03,
    'trunc_dist' : 0.5,
    'passing_ids' : True,
    'K' : 16,
    'd_thre' : 0.03,
    'eps' : 0.3,
    'min_samples' : 1,
    'iters' : 1500,
    
    
    # Choose Models:
    'model' : 'let_it_flow',
    # 'model' : 'MBNSFP',
    # 'model' : 'chodosh',
    # 'model' : 'NP',
    
    }

