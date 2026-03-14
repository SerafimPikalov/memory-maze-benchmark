"""Test configuration for memory-maze-benchmark."""

import os
import sys

if sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
else:
    os.environ.setdefault("MUJOCO_GL", "glfw")
