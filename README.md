# Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners

https://arxiv.org/pdf/2505.23150

This branch contains the implementation of the BRC algorithm.

## Example usage

To run the BRC algorithm in a single task mode, just pass a single task name to the `env_names` variable:

`python3 train.py env_names=dog-run`

By passing a list of task names, multi-task mode will be enabled. 

For the Slurm wrappers, use Hydra overrides as positional arguments:

`sbatch queue.sh env_names=HB_EASY`

The wrapper forces `hydra/launcher=basic` so the training runs inside the allocated Slurm job instead of trying to submit a nested Submitit job.

To record a checkpoint rollout:

MP4 recording requires `imageio-ffmpeg`. To add only the pinned backend to an
existing `dmcdime` environment, preserving all other installed package versions:

```bash
conda run -n dmcdime python -m pip install --no-deps --only-binary=:all: imageio-ffmpeg==0.6.0
```

This targeted install does not require rerunning `pip install -r requirements.txt`.
The rollout script checks the backend before creating environments or evaluating
the policy and reports the Python executable if the backend is unavailable.

```bash
MUJOCO_GL=egl python3 rollout_policy.py env_names=HB_BASIC \
  rollout_checkpoint_path=/path/to/checkpoint/0
```

Use the same environment list (in the same order), `width_actor`, `depth_actor`, and
`width_critic` as training. The checkpoint directory must contain `actor.txt`,
`critic.txt`, `target_critic.txt`, and `temp.txt`; multi-task policies need the
critic's task embeddings as well as the actor.

To reuse a training run's saved configuration, pass
`--config-path /path/to/training/run/.hydra --config-name config` and add
`hydra/launcher=basic rollout_log_to_wandb=false` to the rollout command.
Relative checkpoint and output paths resolve from the directory where the command
was launched. Videos are saved to `./rollout_videos` by default. Override
`rollout_output_dir`, `rollout_episodes`, or `rollout_temperature` as needed;
`rollout_log_to_wandb=true` enables W&B logging. W&B logging is disabled by default.
Without `--multirun`, rollouts execute locally.

DMC videos use model camera `0` by default. For locomotion tasks such as walker,
dog, and humanoid, this camera follows the robot. Set `rollout_dmc_camera_id=1`
for an alternative model camera or `rollout_dmc_camera_id=-1` for the stationary
free camera. Camera names are also accepted if every DMC task in the rollout
defines that name. When using a saved training config without this option, add it
with `+rollout_dmc_camera_id=1`; the default remains `0` when omitted.

To submit a rollout to Slurm, use `--multirun` and override the partition and time
limit on the inherited `submitit_slurm` launcher:

```bash
python3 rollout_policy.py env_names=MYO_HAND_EASY seed=0 \
  rollout_checkpoint_path=/path/to/checkpoint/0 \
  hydra.launcher.partition=gpu_a100_short hydra.launcher.timeout_min=10 --multirun
```

For local multiruns, explicitly select `hydra/launcher=basic`.

For Slurm rollouts using `dmcdime`, add
`hydra/launcher=submitit_slurm_dmc` to the command above. This launcher's setup
activates `dmcdime`; the default `submitit_slurm` launcher activates `dime`.
Partition and timeout overrides work with either Slurm launcher.

## Citation

If you find this repository useful, feel free to cite our paper using the following bibtex.

```
@article{nauman2025bigger,
  title={Bigger, Regularized, Categorical: High-Capacity Value Functions are Efficient Multi-Task Learners},
  author={Nauman, Michal and Cygan, Marek and Sferrazza, Carmelo and Kumar, Aviral and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2505.23150},
  year={2025}
}
```
