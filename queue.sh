#!/bin/bash
#SBATCH --time=30:00:00  # Uncomment this line
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

module load devel/cuda/12.8
eval "$(conda shell.bash hook)"
conda activate dime
export MUJOCO_GL=egl

# Translate legacy --key=value arguments into Hydra overrides.
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
