

conda create -n let-it-flow python=3.10
conda activate let-it-flow

conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && python3 -m pip install -e .
cd ..


conda install pytorch-scatter -c pyg
conda install matplotlib 
conda install scikit-learn
