# Soft Rigidity Scene Flow Regularization

<!-- # Results on StereoKITTI dataset 
![alt text](docs/performance.png) -->



# Installation

Required version of Python >= 3.10, [PyTorch3d](https://github.com/facebookresearch/pytorch3d), [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter/tree/master)
See install.sh for installation of libraries or run it directly:

```console
bash install.sh
```

# DATA
- Setup directory for extracting the data, visuals and experimental results
```console
BASE_PATH='path_where_to_store_data'
```
- Download [Argoverse2](https://login.rci.cvut.cz/data/lidar_intensity/argoverse2.tgz) preprocessed data
<!-- - Download [Data](https://login.rci.cvut.cz/data/lidar_intensity/sceneflow/data_sceneflow.tgz) and unpack it to the folder $BASE_PATH/ -->

<!-- ```console -->
<!-- tar -xvf data_sceneflow.tgz $BASE_PATH/data/sceneflow -->
<!-- ``` -->

