#!/bin/bash

#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=8         # 36 tasks per node
#SBATCH --time=4:00:00               # time limits: 1 hour
#SBATCH --partition=amdgpufast      # partition name
#SBATCH --gres=gpu:1                 # 1 gpu per node
#SBATCH --mem=30G
#SBATCH --error=log/ST_abla_without_K.err            # standard error file
#SBATCH --output=log/ST_abla_without_K.out           # standard output file
#SBATCH --array 0-3%4

ml PyTorch3D/0.7.5-foss-2023a-CUDA-12.1.1
ml scikit-learn/1.3.1-gfbf-2023a
ml JupyterLab/4.0.5-GCCcore-12.3.0



cd ~/let-it-flow/


python run_exps.py ${SLURM_ARRAY_TASK_ID} 4 # 4 is the number of runs per experiment and indexing is minus 1 basially
