#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

module load devel/cuda/12.8
eval "$(conda shell.bash hook)"
if conda activate hbdime; then
    echo "Activated conda env: hbdime"
elif conda activate dime; then
    echo "Activated conda env: dime"
else
    echo "Failed to activate conda env hbdime or dime" >&2
    exit 1
fi
export MUJOCO_GL=egl

args=()
multirun=false
for arg in "$@"; do
    case "$arg" in
        --multirun)
            multirun=true
            ;;
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

echo "Python executable: $(which python)"
python -c "import sys; print('Python version:', sys.version)"

if $multirun; then
    env_arg_index=-1
    env_arg_value=""
    for i in "${!args[@]}"; do
        if [[ "${args[$i]}" == env_names=* ]]; then
            env_arg_index=$i
            env_arg_value="${args[$i]#env_names=}"
            break
        fi
    done

    if [[ $env_arg_index -lt 0 ]]; then
        echo "Expected env_names=... when using --multirun" >&2
        exit 2
    fi

    IFS=',' read -r -a env_sweep <<< "$env_arg_value"
    for env_name in "${env_sweep[@]}"; do
        env_name="${env_name#"${env_name%%[![:space:]]*}"}"
        env_name="${env_name%"${env_name##*[![:space:]]}"}"
        run_args=("${args[@]}")
        run_args[$env_arg_index]="env_names=${env_name}"
        echo "Starting run with env_names=${env_name} seed=${SLURM_ARRAY_TASK_ID}"
        python train.py hydra/launcher=basic "${run_args[@]}" "seed=${SLURM_ARRAY_TASK_ID}"
    done
else
python train.py hydra/launcher=basic "${args[@]}" "seed=${SLURM_ARRAY_TASK_ID}"
fi
conda deactivate
