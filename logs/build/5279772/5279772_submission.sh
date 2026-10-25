#!/bin/bash

# Parameters
#SBATCH --account=p201236
#SBATCH --error=/mnt/tier2/users/u104045/dev/unitccl/nccl/logs/build/%j/%j_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=unitccl-build-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/mnt/tier2/users/u104045/dev/unitccl/nccl/logs/build/%j/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --signal=USR2@90
#SBATCH --time=60
#SBATCH --wckey=submitit

# setup
source /etc/profile
module load env/release/2025.1
module load OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0
module load CUDA/12.8
module load GCC/14.2.0
module load CMake/3.31.3-GCCcore-14.2.0
export LD_LIBRARY_PATH=/home/users/u104045/dev/unitccl/nccl/build/lib/:$LD_LIBRARY_PATH

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /mnt/tier2/users/u104045/dev/unitccl/nccl/logs/build/%j/%j_%t_log.out --error /mnt/tier2/users/u104045/dev/unitccl/nccl/logs/build/%j/%j_%t_log.err /mnt/tier2/users/u104045/dev/unitccl/.venv/bin/python -u -m submitit.core._submit /mnt/tier2/users/u104045/dev/unitccl/nccl/logs/build/%j
