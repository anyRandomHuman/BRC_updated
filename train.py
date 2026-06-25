import os
import hydra

# os.environ['MUJOCO_GL'] = 'egl'

from jaxrl.agent.brc_learner import BRC
from jaxrl.replay_buffer import ParallelReplayBuffer
from jaxrl.envs import ParallelEnv
from jaxrl.normalizer import RewardNormalizer
from jaxrl.logger import EpisodeRecorder
from jaxrl.env_names import get_environment_list
import wandb


@hydra.main(config_path='config', config_name='default')
def main(cfg):
    if cfg.log_to_wandb:
        wandb.init(
            config=dict(cfg),
            entity=cfg.wandb.entity,
            project=cfg.project,
            group=f'{cfg.env_names}',
            name=f'{cfg.env_names}_{cfg.seed}',
            job_type=cfg.job_type,
        )

    env_names = get_environment_list(cfg.env_names)
    print(f"Resolved env_names from {cfg.env_names!r} to {env_names!r}", flush=True)
    env = ParallelEnv(env_names, seed=cfg.seed)
    if cfg.offline_evaluation:
        eval_env = ParallelEnv(env_names, seed=cfg.seed + 42)
    else:
        eval_env = None

    eval_interval = cfg.eval_interval

    # Kwargs setup
    kwargs = {}
    kwargs['updates_per_step'] = cfg.updates_per_step
    kwargs['width_critic'] = cfg.width_critic
    kwargs['w_lr'] = cfg.w_lr
    kwargs['w_d'] = cfg.w_d
    kwargs['loss_process'] = cfg.loss_process
    kwargs['warmup_epochs'] = cfg.warmup_epochs
    kwargs['use_separate_actor_grad'] = cfg.use_separate_actor_grad
    kwargs['use_separate_critic_grad'] = cfg.use_separate_critic_grad
    kwargs['separate_grad_every'] = cfg.separate_grad_every
    kwargs['grad_process'] = cfg.grad_process
    kwargs['last_init_str'] = cfg.last_init_str
    kwargs['width_actor'] = cfg.width_actor
    kwargs['depth_actor'] = cfg.depth_actor
    log_bin_interval = getattr(cfg, 'log_bin_interval', 100)

    num_tasks = len(env.envs)

    agent = BRC(
        cfg.seed,
        env.observation_space.sample()[:1],
        env.action_space.sample()[:1],
        num_tasks=num_tasks,
        v_max=cfg.v_max,
        cfg=cfg,
        **kwargs,
    )
    batch_size = 1024 if agent.multitask else 256

    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1], cfg.replay_buffer_size,
                                         num_tasks=num_tasks)
    if cfg.normalize:
        reward_normalizer = RewardNormalizer(num_tasks, target_entropy=agent.target_entropy, discount=agent.discount)
    else:
        reward_normalizer = None
    statistics_recorder = EpisodeRecorder(num_tasks, env_names, cfg.heatmap_max_history)

    observations = env.reset()

    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        submit_dir = os.environ.get('SLURM_SUBMIT_DIR')
        save_space = r'/pfs/work9/workspace/scratch/ka_et4232-tcx/checkpoints/BRC'
    else:
        submit_dir = '.'
        save_space = './checkpoints'
    save_dir = save_space
    # save_dir can be overridden via cfg.save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if cfg.save_path == '':
        cfg.save_path = f'{cfg.env_names}_{cfg.job_type}'
    save_path = f'{save_dir}/{cfg.save_path}/{cfg.seed}'
    save_path = f'{save_dir}/{cfg.save_path}/{cfg.seed}'
    os.makedirs(save_path, exist_ok=True)

    if cfg.disable_jit:
        import jax
        jax.config.update("jax_disable_jit", True)

    for i in range(1, cfg.max_steps + 1):
        actions = env.action_space.sample() if i < cfg.start_training else agent.sample_actions(observations,
                                                                                                  temperature=1.0)
        next_observations, rewards, terms, truns, goals = env.step(actions)
        if cfg.normalize:
            reward_normalizer.update(rewards, terms, truns)
        statistics_recorder.update(rewards, goals, terms, truns)
        masks = env.generate_masks(terms, truns)
        replay_buffer.insert(observations, actions, rewards, masks, next_observations)
        observations = next_observations
        observations, terms, truns = env.reset_where_done(observations, terms, truns)
        if i >= cfg.start_training:
            batches = replay_buffer.sample_equal_task_batches(cfg.batch_size, cfg.updates_per_step)
            if cfg.normalize:
                batches = reward_normalizer.normalize(batches, agent.get_temperature())
            infos = agent.update(batches, cfg.updates_per_step, i)
            if i % log_bin_interval == 0:
                statistics_recorder.histogram_logger._on_step(infos.pop('q_logits'), i)
            if (i % eval_interval == 0 or i % cfg.online_eval_interval == 0) and i >= cfg.start_training:
                # new_dict = {}
                # for key in infos.keys():
                #     if infos[key].ndim == 1 and infos[key].shape[0] == len(env_names):
                #         new_dict |= {f'{env_names[i]}/{key}': infos[key][i] for i, env_name in enumerate(env_names)}
                #     else:
                #         new_dict |= {key: infos[key]}
                # weights_dict = {f'{env_names[i]}/actor_task_weights': infos['actor_task_weights'][i] for i, env_name in enumerate(env_names)}
                # wandb.log(new_dict, step=i)
                info_dict = statistics_recorder.log(cfg, agent, replay_buffer, reward_normalizer, i, eval_env,
                                                    render=cfg.render, infos=infos)
        if i > (cfg.max_steps / 2) and not i % (cfg.max_steps / 3):
            agent.save(save_path)

    agent.save(save_path)

import traceback

if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        error_msg = traceback.format_exc()

        # Print to console so cluster logs capture full traceback.
        print("-- exception occurred. traceback:")
        print(error_msg, flush=True)
        print("--------------------------------\n")

        if wandb.run is not None and not 'test' in wandb.run.project:
            try:
                # Persist crash metadata in the run itself.
                wandb.run.summary["crashed"] = True
                wandb.run.summary["crash_type"] = type(ex).__name__
                wandb.run.summary["crash_message"] = str(ex)
                # Keep traceback length bounded for summary storage.
                wandb.run.summary["crash_traceback"] = error_msg[-64000:]
                wandb.log({"crashed": 1})

                # Alert is best-effort; summary logging above is the source of truth.
                wandb.alert(
                    title="Run Crashed",
                    text=f"{type(ex).__name__}: {ex}",
                    level=wandb.AlertLevel.ERROR,
                )
            except Exception as wandb_ex:
                print(f"Failed to write crash info to wandb: {wandb_ex}", flush=True)
            finally:
                wandb.finish(exit_code=1)

        # Re-raise original exception so job status is failed.
        raise
    else:
        if wandb.run is not None:
            wandb.finish(exit_code=0)
