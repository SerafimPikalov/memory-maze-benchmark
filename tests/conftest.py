"""Test configuration for memory-maze-benchmark.

Sets up rendering backend env vars (must happen before any MuJoCo/dm_control import),
registers custom pytest markers, and provides shared fixtures.
"""

import os
import sys

# Rendering backend — must be set before any MuJoCo/dm_control/Genesis import.
if sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
else:
    os.environ.setdefault("MUJOCO_GL", "glfw")

import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "genesis: marks tests requiring the Genesis backend")
    config.addinivalue_line("markers", "gpu: marks tests requiring a CUDA GPU")


@pytest.fixture
def device():
    """Return the best available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def obs_shape():
    """Standard Memory Maze observation shape (CHW after wrapper)."""
    return (3, 64, 64)


@pytest.fixture
def num_actions():
    """Memory Maze action space size."""
    return 6
