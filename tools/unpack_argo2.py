# import 
import os
from tqdm import tqdm
import numpy as np
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader

dataset = SceneFlowDataloader('/home/patrik/rci/mnt/personal/vacekpa2/data', 'argoverse2', 'val')

store_path = '/home/patrik/cmp/data/argoverse2/sensor/processed_val/'
#
last_uuid = '---'
count = 0
max_count = 10

for i in tqdm(range(len(dataset))):
	
	seq_uuid = dataset.get_log_id(i)
	
	if seq_uuid != last_uuid and count < max_count:

		data = dataset[i]



		# lidar1 = data[0]
		pc1 = data[0].lidar.as_tensor().detach().cpu().numpy()
		ground1 = data[0].is_ground.detach().cpu().numpy()

		# lidar2 = data[1]
		pc2 = data[1].lidar.as_tensor().detach().cpu().numpy()
		ground2 = data[1].is_ground.detach().cpu().numpy()

		uuid1 = data[0].sweep_uuid[0]
		uuid2 = data[1].sweep_uuid[0]
		timestamp = data[0].sweep_uuid[1]
		
		if uuid1 != uuid2: 
			print('uuid mismatched')
			continue


		pose = data[2].matrix().detach().cpu().numpy()[0]	# batch
		
		flow = data[3].flow.detach().cpu().numpy()
		flow_valid = data[3].is_valid.detach().cpu().numpy()
		category_indices = data[3].category_indices.detach().cpu().numpy()
		dynamic = data[3].is_dynamic.detach().cpu().numpy()
		class_names = data[0].cuboids.category

		d_dict = {'pc1' : pc1,
				 'pc2' : pc2,
				 'pose' : pose,			
				 'ground1' : ground1,
				 'ground2' : ground2,
				 'flow' : flow,
				 'flow_valid' : flow_valid,
				 'dynamic' : dynamic,
				 'category_indices' : category_indices,
				 'uuid1' : uuid1,
				 'uuid2' : uuid2,
				 'class_names' : class_names
				 
				 }

		sequence = uuid1
		final_path = store_path + '/' + sequence + '/' + str(timestamp)
		os.makedirs(os.path.dirname(final_path), exist_ok=True)
		
		np.savez(final_path, **d_dict)

		if count == max_count - 1:
			last_uuid = seq_uuid
			count = 0
		
		count += 1

	else:
		continue