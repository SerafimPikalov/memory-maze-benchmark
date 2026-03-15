"""Environment tests -- creation, stepping, and wrapper verification (~10s).

Tests MuJoCo Memory Maze environments. Genesis tests are skipped unless installed.
"""

import numpy as np
import pytest
import gym


class TestMuJoCoEnvironment:
    """Test Memory Maze with MuJoCo backend."""

    @pytest.fixture
    def env(self):
        e = gym.make("memory_maze:MemoryMaze-9x9-v0", disable_env_checker=True, seed=42)
        yield e
        e.close()

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (64, 64, 3)
        assert env.observation_space.dtype == np.uint8

    def test_action_space(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 6

    def test_reset(self, env):
        obs = env.reset()
        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.uint8
        assert obs.min() < 100
        assert obs.max() > 100

    def test_step(self, env):
        env.reset()
        obs, reward, done, info = env.step(1)
        assert obs.shape == (64, 64, 3)
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(done, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_all_actions(self, env):
        env.reset()
        for action in range(6):
            obs, reward, done, info = env.step(action)
            assert obs.shape == (64, 64, 3)
            if done:
                env.reset()

    def test_reward_non_negative(self, env):
        env.reset()
        for _ in range(50):
            obs, reward, done, info = env.step(env.action_space.sample())
            assert reward >= 0.0
            if done:
                env.reset()

    def test_episode_terminates(self, env):
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())
            steps += 1
            if steps > 1100:
                break
        assert done
        assert steps <= 1001


class TestMemoryMazeWrapper:
    """Test the CHW transposition wrapper used by IMPALA training."""

    @pytest.fixture
    def wrapped_env(self):
        from train_impala import MemoryMazeWrapper
        raw = gym.make("memory_maze:MemoryMaze-9x9-v0", disable_env_checker=True, seed=42)
        e = MemoryMazeWrapper(raw)
        yield e
        e.close()

    def test_observation_space(self, wrapped_env):
        assert wrapped_env.observation_space.shape == (3, 64, 64)

    def test_reset_shape(self, wrapped_env):
        obs = wrapped_env.reset()
        assert obs.shape == (3, 64, 64)
        assert obs.dtype == np.uint8

    def test_step_shape(self, wrapped_env):
        wrapped_env.reset()
        obs, _, _, _ = wrapped_env.step(0)
        assert obs.shape == (3, 64, 64)

    def test_contiguous(self, wrapped_env):
        obs = wrapped_env.reset()
        assert obs.flags["C_CONTIGUOUS"]

    def test_transpose_correctness(self):
        from train_impala import MemoryMazeWrapper
        raw = gym.make("memory_maze:MemoryMaze-9x9-v0", disable_env_checker=True, seed=42)
        wrapped = MemoryMazeWrapper(raw)
        raw_obs = raw.reset()
        wrapped_obs = wrapped.reset()
        np.testing.assert_array_equal(wrapped_obs, raw_obs.transpose(2, 0, 1))
        raw.close()
        wrapped.close()


class TestTorchBeastEnvironment:
    """Test the TorchBeast Environment wrapper."""

    @pytest.fixture
    def tb_env(self):
        from train_impala import MemoryMazeWrapper
        from torchbeast.core.environment import Environment
        raw = gym.make("memory_maze:MemoryMaze-9x9-v0", disable_env_checker=True, seed=42)
        e = Environment(MemoryMazeWrapper(raw))
        yield e
        e.close()

    def test_initial_returns_dict(self, tb_env):
        obs = tb_env.initial()
        assert isinstance(obs, dict)
        assert "frame" in obs
        assert "reward" in obs
        assert "done" in obs

    def test_initial_frame_shape(self, tb_env):
        import torch
        obs = tb_env.initial()
        assert obs["frame"].shape == (1, 1, 3, 64, 64)
        assert obs["frame"].dtype == torch.uint8

    def test_step(self, tb_env):
        import torch
        tb_env.initial()
        step_obs = tb_env.step(torch.tensor([[0]], dtype=torch.int64))
        assert step_obs["frame"].shape == (1, 1, 3, 64, 64)
