import argparse
import math
import os
from pathlib import Path

import numpy as np
import yaml

from jaxrl.env_names import get_environment_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained BRC policy, roll it out, and save an MP4 video."
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint directory containing actor.txt, critic.txt, target_critic.txt, and temp.txt.")
    parser.add_argument("--checkpoint-root", type=str, default=None,
                        help="Checkpoint root. Defaults to ./checkpoints locally and the training scratch path on Slurm.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path. Defaults to rollouts/<save_path>/<seed>.mp4.")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root used for default relative output paths. Defaults to SLURM_SUBMIT_DIR on Slurm, otherwise cwd.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to record for each task.")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum environment steps to record.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Policy sampling temperature.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video FPS.")
    parser.add_argument("--render-every", type=int, default=1,
                        help="Record one frame every N environment steps.")
    parser.add_argument("--mujoco-gl", type=str, default=None,
                        help="Optional MUJOCO_GL value, e.g. egl, osmesa, or glfw. Defaults to egl on Slurm.")
    parser.add_argument("overrides", nargs="*",
                        help="Hydra config overrides, e.g. env_names=TEST_MULTI_VARYING seed=0 save_path=...")
    return parser.parse_args()


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def to_attr_dict(value):
    if isinstance(value, dict):
        return AttrDict({key: to_attr_dict(val) for key, val in value.items()})
    if isinstance(value, list):
        return [to_attr_dict(item) for item in value]
    return value


def parse_override_value(value):
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_override(cfg, override):
    if "=" not in override:
        raise ValueError(f"Expected key=value override, got: {override}")

    key, value = override.split("=", 1)
    if key.startswith("hydra/") or key.startswith("+hydra/"):
        return

    target = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = AttrDict()
        target = target[part]
    target[parts[-1]] = to_attr_dict(parse_override_value(value))


def has_override(overrides, key):
    prefix = f"{key}="
    return any(override.replace(" ", "").startswith(prefix) for override in overrides)


def load_cfg(overrides):
    config_dir = Path(__file__).resolve().parent / "config"
    with open(config_dir / "default.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data.pop("defaults", None)
    cfg = to_attr_dict(data)
    for override in overrides:
        apply_override(cfg, override)
    return cfg


def running_on_slurm():
    return os.environ.get("SLURM_JOB_ID") is not None or os.environ.get("SLURM_SUBMIT_DIR") is not None


def submit_dir():
    return Path(os.environ.get("SLURM_SUBMIT_DIR", ".")).resolve()


def default_checkpoint_root():
    if running_on_slurm():
        return Path("/pfs/work9/workspace/scratch/ka_et4232-tcx/checkpoints/BRC")
    return Path("checkpoints")


def run_save_path(cfg):
    save_path = cfg.save_path if cfg.save_path else f"{cfg.env_names}_{cfg.job_type}"
    return Path(str(save_path)) / str(cfg.seed)


def resolve_checkpoint_path(cfg, checkpoint, checkpoint_root):
    if checkpoint is not None:
        return Path(checkpoint).expanduser()
    root = Path(checkpoint_root).expanduser() if checkpoint_root is not None else default_checkpoint_root()
    return root / run_save_path(cfg)


def default_output_path(cfg, output_root):
    root = Path(output_root).expanduser() if output_root is not None else submit_dir()
    return root / "rollouts" / run_save_path(cfg).with_suffix(".mp4")


def resolve_output_path(cfg, output, output_root):
    if output is None:
        return default_output_path(cfg, output_root)

    path = Path(output).expanduser()
    if path.is_absolute():
        return path

    root = Path(output_root).expanduser() if output_root is not None else submit_dir()
    return root / path


def make_agent(cfg, env):
    from jaxrl.agent.brc_learner import BRC

    kwargs = {
        "updates_per_step": cfg.updates_per_step,
        "width_critic": cfg.width_critic,
        "w_lr": cfg.w_lr,
        "w_d": cfg.w_d,
        "loss_process": cfg.loss_process,
        "warmup_epochs": cfg.warmup_epochs,
        "use_separate_actor_grad": cfg.use_separate_actor_grad,
        "use_separate_critic_grad": cfg.use_separate_critic_grad,
        "separate_grad_every": cfg.separate_grad_every,
        "grad_process": cfg.grad_process,
        "last_init_str": cfg.last_init_str,
        "width_actor": cfg.width_actor,
        "depth_actor": cfg.depth_actor,
    }
    return BRC(
        cfg.seed,
        env.observation_space.sample()[:1],
        env.action_space.sample()[:1],
        num_tasks=len(env.envs),
        v_max=cfg.v_max,
        cfg=cfg,
        **kwargs,
    )


def to_uint8_rgb(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        high = 1.0 if frame.max(initial=0.0) <= 1.0 else 255.0
        frame = np.clip(frame, 0.0, high) / high * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def tile_frames(frames):
    frames = [to_uint8_rgb(frame) for frame in frames]
    if len(frames) == 1:
        return frames[0]

    height = max(frame.shape[0] for frame in frames)
    width = max(frame.shape[1] for frame in frames)
    channels = frames[0].shape[2]
    cols = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / cols)
    canvas = np.zeros((rows * height, cols * width, channels), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        row, col = divmod(idx, cols)
        h, w = frame.shape[:2]
        canvas[row * height:row * height + h, col * width:col * width + w] = frame
    return canvas


def render_tiled_frame(env):
    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "env.render() returned None. The environment may need rgb_array render mode."
        )
    if frame.ndim == 3:
        return to_uint8_rgb(frame)
    return tile_frames(frame)


def rollout(env, agent, output_path, episodes, max_steps, temperature, fps, render_every):
    if episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if render_every <= 0:
        raise ValueError("--render-every must be positive.")

    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "imageio is required to write rollout videos. Install the project requirements first."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    observations = env.reset()
    returns = np.zeros(env.num_tasks, dtype=np.float64)
    goals = np.zeros(env.num_tasks, dtype=np.float64)
    completed = np.zeros(env.num_tasks, dtype=np.int32)
    active = np.ones(env.num_tasks, dtype=np.float64)
    frames_written = 0

    with imageio.get_writer(output_path, fps=fps, macro_block_size=1) as writer:
        for step in range(max_steps):
            if step % render_every == 0:
                writer.append_data(render_tiled_frame(env))
                frames_written += 1

            actions = agent.sample_actions(observations, temperature=temperature)
            next_observations, rewards, terminals, truncates, successes = env.step(actions)

            returns += rewards * active
            goals += successes * active
            done = np.logical_or(terminals, truncates)
            completed += np.logical_and(done, active > 0).astype(np.int32)
            active = np.where(completed >= episodes, 0.0, 1.0)

            observations = next_observations
            observations, terminals, truncates = env.reset_where_done(observations, terminals, truncates)

            if np.all(completed >= episodes):
                break

    return {
        "steps": step + 1,
        "frames": frames_written,
        "returns": returns / np.maximum(completed, 1),
        "goals": goals / np.maximum(completed, 1),
        "completed": completed,
    }


def main():
    args = parse_args()
    mujoco_gl = args.mujoco_gl
    if mujoco_gl is None and running_on_slurm():
        mujoco_gl = "egl"
    if mujoco_gl is not None:
        os.environ.setdefault("MUJOCO_GL", mujoco_gl)

    cfg = load_cfg(args.overrides)
    if running_on_slurm() and "SLURM_ARRAY_TASK_ID" in os.environ and not has_override(args.overrides, "seed"):
        cfg.seed = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if cfg.disable_jit:
        import jax
        jax.config.update("jax_disable_jit", True)

    env_names = get_environment_list(cfg.env_names)
    print(f"Slurm mode: {running_on_slurm()}  seed={cfg.seed}  MUJOCO_GL={os.environ.get('MUJOCO_GL')}", flush=True)
    print(f"Resolved env_names from {cfg.env_names!r} to {env_names!r}", flush=True)

    from jaxrl.envs import ParallelEnv

    env = ParallelEnv(env_names, seed=cfg.seed)
    agent = make_agent(cfg, env)

    checkpoint = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_root)
    output_path = resolve_output_path(cfg, args.output, args.output_root)
    print(f"Checkpoint path: {checkpoint}", flush=True)
    print(f"Output path: {output_path}", flush=True)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    agent.load(str(checkpoint))

    stats = rollout(
        env=env,
        agent=agent,
        output_path=output_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        temperature=args.temperature,
        fps=args.fps,
        render_every=args.render_every,
    )

    print(f"Loaded checkpoint from {checkpoint}", flush=True)
    print(f"Saved video to {output_path}", flush=True)
    print(f"Steps: {stats['steps']}  Frames: {stats['frames']}", flush=True)
    for env_name, ret, goal, done in zip(env_names, stats["returns"], stats["goals"], stats["completed"]):
        print(f"{env_name}: episodes={done} return={ret:.3f} goal={goal:.3f}", flush=True)


if __name__ == "__main__":
    main()
