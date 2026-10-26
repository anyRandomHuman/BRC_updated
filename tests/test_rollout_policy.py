import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
from hydra import compose, initialize_config_dir

import rollout_policy
from jaxrl.envs import MyoSuiteRGBWrapper, ParallelEnv


class EpisodeEnv(gym.Env):
    def __init__(self, horizon):
        self.horizon = horizon
        self.observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(1), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(1), 1.0, self.steps == self.horizon, False, {'solved': True}

    def render(self):
        return np.zeros((16, 16, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class RolloutTests(unittest.TestCase):
    def setUp(self):
        config_dir = Path(__file__).resolve().parents[1] / 'config'
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            self.cfg = compose(config_name='rollout')

    def test_local_defaults(self):
        self.assertFalse(self.cfg.rollout_log_to_wandb)
        with initialize_config_dir(
            version_base=None,
            config_dir=str(Path(__file__).resolve().parents[1] / 'config'),
        ):
            cfg = compose(config_name='rollout', return_hydra_config=True)
        self.assertEqual(cfg.hydra.launcher._target_, 'hydra._internal.core_plugins.basic_launcher.BasicLauncher')

    def test_incomplete_checkpoint_fails_before_environment_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'actor.txt').touch()
            self.cfg.rollout_checkpoint_path = directory
            with patch.object(rollout_policy, 'ParallelEnv') as env:
                with self.assertRaisesRegex(FileNotFoundError, 'critic.txt.*target_critic.txt.*temp.txt'):
                    rollout_policy.main.__wrapped__(self.cfg)
            env.assert_not_called()

    def test_environment_closes_when_loading_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('actor', 'critic', 'target_critic', 'temp'):
                (Path(directory) / f'{name}.txt').touch()
            self.cfg.rollout_checkpoint_path = directory
            with patch.object(rollout_policy, 'ParallelEnv') as env, patch.object(
                rollout_policy, '_build_agent'
            ) as build_agent:
                build_agent.return_value.load.side_effect = ValueError('bad checkpoint')
                with self.assertRaisesRegex(ValueError, 'bad checkpoint'):
                    rollout_policy.main.__wrapped__(self.cfg)
                self.assertEqual(env.call_args.kwargs['render_mode'], 'rgb_array')
                env.return_value.close.assert_called_once()

    def test_success_counts_stop_after_requested_episodes(self):
        tasks = [EpisodeEnv(1), EpisodeEnv(3)]
        with patch('jaxrl.envs.make_env', side_effect=tasks):
            env = ParallelEnv(['fast', 'slow'], render_mode='rgb_array')
        agent = Mock()
        agent.sample_actions.return_value = np.zeros((2, 1))
        try:
            stats = env.evaluate(agent, num_episodes=2, render=True,
                                 max_render_steps=3, render_frameskip=2)
        finally:
            env.close()
        np.testing.assert_array_equal(stats['return'], [1, 3])
        np.testing.assert_array_equal(stats['goal'], [1, 1])
        self.assertEqual(stats['renders'].shape, (2, 2, 3, 16, 16))
        self.assertTrue(all(task.closed for task in tasks))

    def test_myosuite_uses_offscreen_renderer(self):
        task = EpisodeEnv(1)
        task.sim = Mock()
        expected = np.zeros((480, 640, 3), dtype=np.uint8)
        task.sim.renderer.render_offscreen.return_value = expected
        env = MyoSuiteRGBWrapper(task)
        self.assertEqual(env.render_mode, 'rgb_array')
        self.assertIs(env.render(), expected)
        task.sim.renderer.render_offscreen.assert_called_once_with(width=640, height=480)
        env.close()
        task.sim.renderer.close.assert_called_once()
        self.assertTrue(task.closed)


if __name__ == '__main__':
    unittest.main()
