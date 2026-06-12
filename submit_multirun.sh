#!/bin/bash
# Usage:
#   ./submit_multirun.sh [sbatch options...] <batch_script> env_names=a,b,c [train args...]
#
# Example:
#   ./submit_multirun.sh --partition=gpu_a100_short --time=00:10:00 test.sh \
#     env_names=h1-walk-v0,h1-stand-v0 job_type=default

set -euo pipefail

sbatch_opts=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            sbatch_opts+=("$1")
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [sbatch options...] <batch_script> env_names=a,b,c [train args...]" >&2
    exit 2
fi

batch_script="$1"
shift

env_arg="$1"
shift

if [[ "$env_arg" != env_names=* ]]; then
    echo "Expected second positional argument to be env_names=..." >&2
    exit 2
fi

env_value="${env_arg#env_names=}"
IFS=',' read -r -a envs <<< "$env_value"

for env_name in "${envs[@]}"; do
    env_name="${env_name#"${env_name%%[![:space:]]*}"}"
    env_name="${env_name%"${env_name##*[![:space:]]}"}"
    echo "Submitting ${batch_script} for env_names=${env_name}"
    sbatch "${sbatch_opts[@]}" "$batch_script" "env_names=${env_name}" "$@"
done
