import unittest

import numpy as np
from dm_control.mujoco import engine

from jaxrl.envs import ParallelEnv, _make_env_dmc


class DMCCameraTests(unittest.TestCase):
    def test_locomotion_camera_follows_robot_translation(self):
        for task, root_x_index in (
            ('walker-walk', 1), ('dog-walk', 0), ('humanoid-walk', 0),
        ):
            with self.subTest(task=task):
                env = _make_env_dmc(task, render_mode='rgb_array')
                try:
                    env.reset(seed=0)
                    adapter = env.unwrapped
                    physics = adapter._env.physics
                    camera = engine.Camera(
                        physics, height=240, width=320,
                        camera_id=adapter.render_kwargs['camera_id'],
                    )
                    camera.update()
                    before = camera.scene.camera[0].pos.copy()
                    physics.data.qpos[root_x_index] += 3.0
                    physics.forward()
                    camera.update()
                    np.testing.assert_allclose(
                        camera.scene.camera[0].pos - before, [3, 0, 0], atol=1e-5,
                    )
                    self.assertEqual(env.render().shape, (240, 320, 3))
                finally:
                    env.close()

    def test_free_camera_can_be_selected(self):
        env = _make_env_dmc('walker-walk', render_mode='rgb_array', camera_id=-1)
        try:
            env.reset(seed=0)
            physics = env.unwrapped._env.physics
            camera = engine.Camera(
                physics, height=240, width=320,
                camera_id=env.unwrapped.render_kwargs['camera_id'],
            )
            camera.update()
            before = camera.scene.camera[0].pos.copy()
            physics.data.qpos[1] += 3.0
            physics.forward()
            camera.update()
            np.testing.assert_allclose(camera.scene.camera[0].pos, before)
        finally:
            env.close()

    def test_named_camera_reaches_dmc_renderer(self):
        env = ParallelEnv(['walker-walk'], render_mode='rgb_array', dmc_camera_id='back')
        try:
            env.reset()
            self.assertEqual(env.envs[0].unwrapped.render_kwargs['camera_id'], 'back')
            self.assertEqual(env.render().shape, (1, 240, 320, 3))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
