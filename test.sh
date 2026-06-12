#!/bin/bash
set -euo pipefail

#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --output=slurm-%A_%a.out

module load devel/cuda/12.8
eval "$(conda shell.bash hook)"
conda activate py10
export MUJOCO_GL=egl

args=()
for arg in "$@"; do
    case "$arg" in
        --cfg=*|--package=*|--info=*)
            args+=("$arg")
            ;;
        --*=*)
            args+=("${arg#--}")
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

python train.py hydra/launcher=basic "${args[@]}" "seed=${SLURM_ARRAY_TASK_ID}"
conda deactivate
