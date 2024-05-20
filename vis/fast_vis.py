from data.dataloader import NSF_dataset
import numpy as np

def model_output_demo2(exp_id=17):
    from mayavi import mlab
    from data.PATHS import EXP_PATH

    for model in ['multi-my-SC2', 'SCOOP']:
        exp_id = 0 if model == 'SCOOP' else exp_id
        ds = NSF_dataset(dataset_type='kitti_t')

        for frame_id in [27]:
            data = ds[frame_id]

            out = np.load(os.path.join(EXP_PATH, f'{model}/{exp_id}/inference/sample-{frame_id}.npz'), allow_pickle=True)
            cloud1 = data['pc1'].squeeze()
            cloud2 = data['pc2'].squeeze()
            pred_flow = out['pred_flow'].squeeze()
            cloud_pred = cloud1 + pred_flow[..., :3]

            # rel_pose = np.load(os.path.join(EXP_PATH, 'icp_result.npz'))['rel_pose']
            # cloud1 = transform_cloud(cloud1, rel_pose)

            # visualize in mayavi
            mlab.figure(f'{model}_{frame_id}', bgcolor=(1, 1, 1), size=(1000, 600))
            # mlab.points3d(cloud1[:, 0], cloud1[:, 1], cloud1[:, 2], color=(1, 0, 0), scale_factor=0.1)
            mlab.points3d(cloud2[:, 0], cloud2[:, 1], cloud2[:, 2], color=(0, 0, 1), scale_factor=0.1, opacity=0.8)
            mlab.points3d(cloud_pred[:, 0], cloud_pred[:, 1], cloud_pred[:, 2], color=(0, 1, 0), scale_factor=0.1)
            # set view point
            # mlab.view(azimuth=210, elevation=70, distance=14, focalpoint=(15, -10, 0))
            mlab.view(azimuth=260, elevation=70, distance=40, focalpoint=(20, 0, 0))
    mlab.show()
