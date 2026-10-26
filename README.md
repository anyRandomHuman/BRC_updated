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
`rollout_log_to_wandb=true` enables W&B logging. The default rollout configuration
runs locally with W&B logging disabled.

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
