#!/bin/bash

#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=8         # 36 tasks per node
#SBATCH --time=4:00:00               # time limits: 1 hour
#SBATCH --partition=amdgpufast      # partition name
#SBATCH --gres=gpu:1                 # 1 gpu per node
#SBATCH --mem=30G
#SBATCH --error=datasets/log.out            # standard error file
#SBATCH --output=datasets/log.out           # standard output file

######------#SBATCH --array 0-3%4

ml PyTorch3D/0.7.5-foss-2023a-CUDA-12.1.1
ml kornia/0.7.2-foss-2023a-CUDA-12.1.1
ml OpenCV/4.8.1-foss-2023a-CUDA-12.1.1-contrib
ml JupyterLab/4.0.5-GCCcore-12.3.0


cd ~/let-it-flow/

python run_optimization.py

# python run_optimization.py ${SLURM_ARRAY_TASK_ID} 4 # 4 is the number of runs per experiment and indexing is minus 1 basially
