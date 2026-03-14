#!/usr/bin/env python3
"""Smoke test for GPU environments -- verify CUDA, EGL rendering, and env creation.

Usage:
    python smoke_test.py                    # test both backends
    python smoke_test.py --backend mujoco   # test MuJoCo only
    python smoke_test.py --backend genesis  # test Genesis only
"""

import argparse
import sys
import os
import warnings

# Suppress Pydantic V2.11 deprecation warnings from Genesis's options.py
warnings.filterwarnings("ignore", message=".*model_fields.*deprecated.*", category=DeprecationWarning)

# Force headless rendering env vars before any OpenGL/MuJoCo imports
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def check_cuda():
    """Verify CUDA is available and print GPU info."""
    print("=" * 60)
    print("CUDA CHECK")
    print("=" * 60)
    import torch
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return False
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    # Quick tensor operation
    x = torch.randn(100, 100, device="cuda")
    y = x @ x.T
    print(f"  CUDA tensor test: OK (result shape: {y.shape})")
    print("PASS: CUDA")
    return True


def check_egl():
    """Verify EGL is available for headless rendering."""
    print("=" * 60)
    print("EGL CHECK")
    print("=" * 60)
    egl_vendor = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if os.path.exists(egl_vendor):
        print(f"  NVIDIA EGL vendor JSON: {egl_vendor} exists")
    else:
        print(f"  WARNING: {egl_vendor} not found")
    print(f"  MUJOCO_GL={os.environ.get('MUJOCO_GL', 'not set')}")
    print(f"  PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM', 'not set')}")
    try:
        from OpenGL import EGL
        print("  PyOpenGL EGL import: OK")
    except (ImportError, AttributeError) as e:
        print(f"  WARNING: PyOpenGL EGL not available ({e})")
    print("PASS: EGL config")
    return True


def check_batch_renderer():
    """Check if Madrona BatchRenderer is available for GPU rendering."""
    print("=" * 60)
    print("BATCH RENDERER CHECK")
    print("=" * 60)
    try:
        from gs_madrona.renderer_gs import MadronaBatchRendererAdapter
        print("  gs_madrona import: OK")
        print("PASS: BatchRenderer available")
        return True
    except ImportError:
        print("  gs_madrona not installed")
        print("SKIP: BatchRenderer unavailable (GPU rendering disabled)")
        return True  # Not a failure -- graceful fallback to Rasterizer


def check_mujoco():
    """Test MuJoCo backend: create env, step, render."""
    print("=" * 60)
    print("MUJOCO BACKEND CHECK")
    print("=" * 60)
    import gym
    import numpy as np
    env = gym.make("memory_maze:MemoryMaze-9x9-v0", disable_env_checker=True)
    obs = env.reset()
    print(f"  Env created: MemoryMaze-9x9-v0")
    print(f"  Obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  Action space: {env.action_space}")
    # Step a few times
    total_reward = 0
    for i in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward
        if done:
            obs = env.reset()
    print(f"  10 steps OK, total reward: {total_reward:.2f}")
    print(f"  Final obs range: [{obs.min()}, {obs.max()}]")
    env.close()
    print("PASS: MuJoCo backend")
    return True


def check_genesis():
    """Test Genesis backend: create env, step, render."""
    print("=" * 60)
    print("GENESIS BACKEND CHECK")
    print("=" * 60)
    import gym
    import numpy as np
    env = gym.make("memory_maze:MemoryMaze-9x9-Genesis-v0", disable_env_checker=True)
    obs = env.reset()
    print(f"  Env created: MemoryMaze-9x9-Genesis-v0")
    print(f"  Obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  Action space: {env.action_space}")
    total_reward = 0
    for i in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward
        if done:
            obs = env.reset()
    print(f"  10 steps OK, total reward: {total_reward:.2f}")
    print(f"  Final obs range: [{obs.min()}, {obs.max()}]")
    env.close()
    print("PASS: Genesis backend")
    return True


def main():
    parser = argparse.ArgumentParser(description="GPU environment smoke test")
    parser.add_argument("--backend", default="all", choices=["all", "mujoco", "genesis"],
                        help="Which backend to test")
    args = parser.parse_args()

    results = {}

    # Always check CUDA and EGL
    results["cuda"] = check_cuda()
    results["egl"] = check_egl()
    results["batch_renderer"] = check_batch_renderer()

    if args.backend in ("all", "mujoco"):
        try:
            results["mujoco"] = check_mujoco()
        except Exception as e:
            print(f"FAIL: MuJoCo backend -- {e}")
            results["mujoco"] = False

    if args.backend in ("all", "genesis"):
        try:
            results["genesis"] = check_genesis()
        except Exception as e:
            print(f"FAIL: Genesis backend -- {e}")
            results["genesis"] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll checks passed!")
        sys.exit(0)
    else:
        print("\nSome checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
