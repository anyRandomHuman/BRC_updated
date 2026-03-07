#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --output=logs/array_%A_%a.out  # %A is JobID, %a is ArrayID

module load devel/cuda/12.8
eval "$(conda shell.bash hook)"
conda activate py10
export MUJOCO_GL=egl

python train.py "$@" --seed $SLURM_ARRAY_TASK_ID
conda deactivate
