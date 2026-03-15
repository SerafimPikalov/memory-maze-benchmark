#!/usr/bin/env python3
"""Benchmark MuJoCo vs Genesis backends on Memory Maze.

Measures steps/second for each backend at various parallelism levels,
producing a comparison table. Useful for quantifying the Genesis speedup.

Usage:
    # Quick comparison (2 episodes each)
    python benchmark_backends.py

    # Thorough comparison
    python benchmark_backends.py --episodes 5 --actors 1,2,4,8

    # Genesis batched only (requires CUDA)
    python benchmark_backends.py --backends genesis-batched --actors 8,16,32

    # All three backends
    python benchmark_backends.py --backends mujoco,genesis,genesis-batched
"""

import argparse
import os
import sys
import time

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["GENESIS_SKIP_TK_INIT"] = "1"

import numpy as np


def benchmark_mujoco(maze_size, num_actors, total_steps, seed=42):
    """Benchmark MuJoCo backend with N sequential environments."""
    import gym

    envs = []
    for i in range(num_actors):
        env = gym.make(
            f"memory_maze:MemoryMaze-{maze_size}-v0",
            disable_env_checker=True,
            seed=seed + i,
        )
        env.reset()
        envs.append(env)

    rng = np.random.RandomState(seed)
    steps = 0
    total_reward = 0.0

    start = time.monotonic()
    while steps < total_steps:
        for env in envs:
            action = rng.randint(env.action_space.n)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                env.reset()
            if steps >= total_steps:
                break
    elapsed = time.monotonic() - start

    for env in envs:
        env.close()

    return {
        "backend": "mujoco",
        "num_envs": num_actors,
        "total_steps": steps,
        "elapsed_s": elapsed,
        "sps": steps / elapsed,
        "mean_reward_per_step": total_reward / steps,
    }


def benchmark_genesis_single(maze_size, num_actors, total_steps, physics_timestep, seed=42):
    """Benchmark Genesis single-env backend with N sequential environments."""
    import gym

    envs = []
    for i in range(num_actors):
        env = gym.make(
            f"memory_maze:MemoryMaze-{maze_size}-Genesis-v0",
            disable_env_checker=True,
            seed=seed + i,
            physics_timestep=physics_timestep,
        )
        env.reset()
        envs.append(env)

    rng = np.random.RandomState(seed)
    steps = 0
    total_reward = 0.0

    start = time.monotonic()
    while steps < total_steps:
        for env in envs:
            action = rng.randint(env.action_space.n)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                env.reset()
            if steps >= total_steps:
                break
    elapsed = time.monotonic() - start

    for env in envs:
        env.close()

    return {
        "backend": "genesis",
        "num_envs": num_actors,
        "total_steps": steps,
        "elapsed_s": elapsed,
        "sps": steps / elapsed,
        "mean_reward_per_step": total_reward / steps,
    }


def benchmark_genesis_batched(maze_size, num_envs, total_steps, physics_timestep, seed=42):
    """Benchmark Genesis batched backend (GPU physics + BatchRenderer)."""
    import torch
    if not torch.cuda.is_available():
        return {
            "backend": "genesis-batched",
            "num_envs": num_envs,
            "total_steps": 0,
            "elapsed_s": 0,
            "sps": 0,
            "mean_reward_per_step": 0,
            "error": "CUDA not available",
        }

    import genesis as gs
    if not gs._initialized:
        gs.init(backend=gs.cuda, logging_level='warning')

    from memory_maze.genesis_backend import BatchGenesisMemoryMazeEnv

    env = BatchGenesisMemoryMazeEnv(
        n_envs=num_envs,
        maze_size=int(maze_size.split("x")[0]),
        camera_resolution=64,
        seed=seed,
        physics_timestep=physics_timestep,
    )
    env.reset()

    rng = np.random.RandomState(seed)
    steps = 0
    total_reward = 0.0

    start = time.monotonic()
    while steps < total_steps:
        actions = rng.randint(0, 6, size=num_envs)
        obs, rewards, dones, infos = env.step(actions)
        total_reward += rewards.sum()
        steps += num_envs
    elapsed = time.monotonic() - start

    env.close()

    return {
        "backend": "genesis-batched",
        "num_envs": num_envs,
        "total_steps": steps,
        "elapsed_s": elapsed,
        "sps": steps / elapsed,
        "mean_reward_per_step": total_reward / steps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MuJoCo vs Genesis backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backends", default="mujoco,genesis-batched",
                        help="Comma-separated: mujoco, genesis, genesis-batched (default: mujoco,genesis-batched)")
    parser.add_argument("--actors", default="1,4,8",
                        help="Comma-separated actor/env counts (default: 1,4,8)")
    parser.add_argument("--steps", default=4000, type=int,
                        help="Total steps per benchmark (default: 4000)")
    parser.add_argument("--maze-size", default="9x9")
    parser.add_argument("--physics-timestep", default=0.05, type=float,
                        help="Genesis physics timestep (default: 0.05)")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",")]
    actor_counts = [int(a.strip()) for a in args.actors.split(",")]

    print(f"Benchmark: {args.maze_size} maze, {args.steps} steps/run, seed={args.seed}")
    print(f"Backends: {backends}")
    print(f"Actor/env counts: {actor_counts}")
    if "genesis" in backends or "genesis-batched" in backends:
        print(f"Genesis physics_timestep: {args.physics_timestep}")
    print()

    results = []

    for backend in backends:
        for n in actor_counts:
            label = f"{backend} (n={n})"
            print(f"  {label}...", end="", flush=True)
            try:
                if backend == "mujoco":
                    r = benchmark_mujoco(args.maze_size, n, args.steps, args.seed)
                elif backend == "genesis":
                    r = benchmark_genesis_single(args.maze_size, n, args.steps,
                                                  args.physics_timestep, args.seed)
                elif backend == "genesis-batched":
                    r = benchmark_genesis_batched(args.maze_size, n, args.steps,
                                                   args.physics_timestep, args.seed)
                else:
                    print(f" unknown backend '{backend}', skipping")
                    continue

                if r.get("error"):
                    print(f" SKIPPED ({r['error']})")
                else:
                    print(f" {r['sps']:.1f} SPS ({r['elapsed_s']:.1f}s)")
                results.append(r)
            except Exception as e:
                print(f" FAILED: {e}")
                results.append({
                    "backend": backend, "num_envs": n, "total_steps": 0,
                    "elapsed_s": 0, "sps": 0, "mean_reward_per_step": 0, "error": str(e),
                })

    # Print comparison table
    print()
    print(f"{'Backend':<20} {'Envs':>5} {'Steps':>7} {'Time (s)':>9} {'SPS':>8} {'Notes'}")
    print("-" * 65)
    for r in results:
        notes = r.get("error", "")
        print(f"{r['backend']:<20} {r['num_envs']:>5} {r['total_steps']:>7} "
              f"{r['elapsed_s']:>9.2f} {r['sps']:>8.1f} {notes}")


if __name__ == "__main__":
    main()
