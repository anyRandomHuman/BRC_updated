import os
import shlex
import sys
import traceback
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
import imageio.v2 as imageio
import numpy as np
import wandb
from omegaconf import OmegaConf

# Uncomment this on headless machines if MuJoCo needs EGL rendering.
# os.environ['MUJOCO_GL'] = 'egl'

from jaxrl.agent.brc_learner import BRC
from jaxrl.env_names import get_environment_list
from jaxrl.envs import ParallelEnv


def _get_cfg(cfg, name, default):
    return OmegaConf.select(cfg, name, default=default)


def _default_checkpoint_path(cfg):
    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        save_space = r'/pfs/work9/workspace/scratch/ka_et4232-tcx/checkpoints/BRC'
    else:
        save_space = './checkpoints'

    save_path = cfg.save_path
    if save_path == '':
        save_path = f'{cfg.env_names}_{cfg.job_type}'
    return os.path.join(save_space, save_path, str(cfg.seed))


def _build_agent(cfg, env, num_tasks):
    kwargs = {
        'updates_per_step': cfg.updates_per_step,
        'width_critic': cfg.width_critic,
        'w_lr': cfg.w_lr,
        'w_d': cfg.w_d,
        'loss_process': cfg.loss_process,
        'warmup_epochs': cfg.warmup_epochs,
        'use_separate_actor_grad': cfg.use_separate_actor_grad,
        'use_separate_critic_grad': cfg.use_separate_critic_grad,
        'separate_grad_every': cfg.separate_grad_every,
        'grad_process': cfg.grad_process,
        'last_init_str': cfg.last_init_str,
        'width_actor': cfg.width_actor,
        'depth_actor': cfg.depth_actor,
    }

    return BRC(
        cfg.seed,
        env.observation_space.sample()[:1],
        env.action_space.sample()[:1],
        num_tasks=num_tasks,
        v_max=cfg.v_max,
        cfg=cfg,
        **kwargs,
    )


def _check_video_backend():
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_version()
    except (ImportError, OSError, RuntimeError) as error:
        install_command = shlex.join([
            sys.executable, '-m', 'pip', 'install', '--no-deps',
            '--only-binary=:all:', 'imageio-ffmpeg==0.6.0',
        ])
        raise RuntimeError(
            f'MP4 recording requires a working imageio-ffmpeg backend in {sys.executable}. '
            f'Install it without changing other packages: {install_command}. '
            f'Backend error: {error}'
        ) from error


def _save_videos(renders, env_names, output_dir, fps):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = []
    for env_name, render in zip(env_names, renders):
        # ParallelEnv.evaluate returns wandb-ready videos as TCHW per task.
        frames = np.transpose(render, (0, 2, 3, 1))
        frames = np.asarray(np.clip(frames, 0, 255), dtype=np.uint8)
        video_path = output_dir / f'{env_name}.mp4'
        imageio.mimsave(video_path, frames, fps=fps, format='FFMPEG')
        video_paths.append(video_path)
    return video_paths


@hydra.main(version_base=None, config_path='config', config_name='rollout')
def main(cfg):
    if cfg.disable_jit:
        import jax
        jax.config.update("jax_disable_jit", True)

    checkpoint_path = _get_cfg(cfg, 'rollout_checkpoint_path', '')
    if not checkpoint_path:
        checkpoint_path = _default_checkpoint_path(cfg)
    checkpoint_path = to_absolute_path(checkpoint_path)
    print(f'Loading checkpoint from {checkpoint_path}', flush=True)
    missing_files = [
        name for name in ('actor.txt', 'critic.txt', 'target_critic.txt', 'temp.txt')
        if not (Path(checkpoint_path) / name).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f'Incomplete BRC checkpoint at {checkpoint_path}; '
            f'missing: {", ".join(missing_files)}. '
            'Use the directory written by agent.save().'
        )

    rollout_episodes = int(_get_cfg(cfg, 'rollout_episodes', cfg.eval_episodes))
    rollout_temperature = float(_get_cfg(cfg, 'rollout_temperature', 0.0))
    max_render_steps = int(_get_cfg(cfg, 'rollout_max_render_steps', 5000))
    render_frameskip = int(_get_cfg(cfg, 'rollout_render_frameskip', 4))
    video_fps = int(_get_cfg(cfg, 'rollout_video_fps', 15))
    output_dir = to_absolute_path(_get_cfg(cfg, 'rollout_output_dir', './rollout_videos'))
    for name, value in (
        ('rollout_episodes', rollout_episodes),
        ('rollout_max_render_steps', max_render_steps),
        ('rollout_render_frameskip', render_frameskip),
        ('rollout_video_fps', video_fps),
    ):
        if value <= 0:
            raise ValueError(f'{name} must be positive, got {value}.')
    if not np.isfinite(rollout_temperature) or rollout_temperature < 0:
        raise ValueError('rollout_temperature must be finite and nonnegative.')

    _check_video_backend()
    env_names = get_environment_list(cfg.env_names)
    print(f"Resolved env_names from {cfg.env_names!r} to {env_names!r}", flush=True)
    env = ParallelEnv(
        env_names, seed=cfg.seed, render_mode='rgb_array',
        dmc_camera_id=_get_cfg(cfg, 'rollout_dmc_camera_id', 0),
        myo_camera_id=_get_cfg(cfg, 'rollout_myo_camera_id', 'auto'),
    )
    try:
        agent = _build_agent(cfg, env, num_tasks=len(env.envs))
        agent.load(checkpoint_path)

        print(
            f'Rolling out {rollout_episodes} episode(s) per task '
            f'with temperature={rollout_temperature}',
            flush=True,
        )
        eval_stats = env.evaluate(
            agent,
            num_episodes=rollout_episodes,
            temperature=rollout_temperature,
            render=True,
            max_render_steps=max_render_steps,
            render_frameskip=render_frameskip,
        )
    finally:
        env.close()

    video_paths = _save_videos(eval_stats['renders'], env_names, output_dir, video_fps)
    for video_path in video_paths:
        print(f'Saved video: {video_path}', flush=True)

    print(f"Returns: {dict(zip(env_names, eval_stats['return']))}", flush=True)
    print(f"Goals: {dict(zip(env_names, eval_stats['goal']))}", flush=True)

    if _get_cfg(cfg, 'rollout_log_to_wandb', False):
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            entity=cfg.wandb.entity,
            project=cfg.project,
            group=f'{cfg.env_names}',
            name=f'rollout_{cfg.env_names}_{cfg.seed}',
            job_type='rollout',
        )
        wandb.log({
            **{f'{env_name}/return': value for env_name, value in zip(env_names, eval_stats['return'])},
            **{f'{env_name}/goal': value for env_name, value in zip(env_names, eval_stats['goal'])},
            **{f'{env_name}/video': wandb.Video(str(path), fps=video_fps, format='mp4')
               for env_name, path in zip(env_names, video_paths)},
        })
        wandb.finish(exit_code=0)


if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        error_msg = traceback.format_exc()

        print("-- exception occurred. traceback:")
        print(error_msg, flush=True)
        print("--------------------------------\n")

        if wandb.run is not None:
            try:
                wandb.run.summary["crashed"] = True
                wandb.run.summary["crash_type"] = type(ex).__name__
                wandb.run.summary["crash_message"] = str(ex)
                wandb.run.summary["crash_traceback"] = error_msg[-64000:]
                wandb.log({"crashed": 1})
                wandb.alert(
                    title="Rollout Crashed",
                    text=f"{type(ex).__name__}: {ex}",
                    level=wandb.AlertLevel.ERROR,
                )
            except Exception as wandb_ex:
                print(f"Failed to write crash info to wandb: {wandb_ex}", flush=True)
            finally:
                wandb.finish(exit_code=1)

        raise
