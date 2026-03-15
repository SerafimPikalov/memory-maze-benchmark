#!/usr/bin/env python3
"""Benchmark MuJoCo vs Genesis backends on Memory Maze.

Measures steps/second for each backend at various parallelism levels,
matching real IMPALA training patterns:
- MuJoCo / Genesis single-env: N forked processes, each stepping its own env
- Genesis batched: single process, vectorized step over N envs on GPU

Usage:
    # Quick comparison
    python benchmark_backends.py

    # Thorough comparison
    python benchmark_backends.py --actors 1,2,4,8,16 --steps 8000

    # Genesis batched only (requires CUDA)
    python benchmark_backends.py --backends genesis-batched --actors 8,16,32

    # All three backends
    python benchmark_backends.py --backends mujoco,genesis,genesis-batched
"""

import argparse
import multiprocessing as mp
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


# ---------------------------------------------------------------------------
# Worker for forked multiprocess benchmarks (MuJoCo / Genesis single-env)
# ---------------------------------------------------------------------------

def _worker(env_id, steps_per_worker, seed, result_queue, env_kwargs=None):
    """Run in a forked process: create env, step, report results."""
    if sys.platform != "darwin":
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import gym
    kwargs = {"disable_env_checker": True, "seed": seed}
    if env_kwargs:
        kwargs.update(env_kwargs)

    env = gym.make(env_id, **kwargs)
    rng = np.random.RandomState(seed)

    env.reset()
    steps = 0
    total_reward = 0.0

    start = time.monotonic()
    while steps < steps_per_worker:
        action = rng.randint(env.action_space.n)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            env.reset()
    elapsed = time.monotonic() - start

    env.close()
    result_queue.put({"steps": steps, "reward": total_reward, "elapsed": elapsed})


def benchmark_multiprocess(env_id, num_actors, total_steps, seed=42, env_kwargs=None):
    """Benchmark with N forked processes (matches real IMPALA actor setup)."""
    ctx = mp.get_context("fork" if sys.platform == "linux" else "spawn")
    result_queue = ctx.Queue()
    steps_per_worker = total_steps // num_actors

    processes = []
    for i in range(num_actors):
        p = ctx.Process(
            target=_worker,
            args=(env_id, steps_per_worker, seed + i, result_queue, env_kwargs),
        )
        p.start()
        processes.append(p)

    # Wait for all workers
    for p in processes:
        p.join(timeout=300)

    # Collect results
    total_steps_done = 0
    total_reward = 0.0
    max_elapsed = 0.0
    while not result_queue.empty():
        r = result_queue.get_nowait()
        total_steps_done += r["steps"]
        total_reward += r["reward"]
        max_elapsed = max(max_elapsed, r["elapsed"])

    # SPS = total steps across all workers / wall-clock time of slowest worker
    sps = total_steps_done / max_elapsed if max_elapsed > 0 else 0

    return {
        "total_steps": total_steps_done,
        "elapsed_s": max_elapsed,
        "sps": sps,
        "mean_reward_per_step": total_reward / total_steps_done if total_steps_done > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Backend-specific benchmarks
# ---------------------------------------------------------------------------

def benchmark_mujoco(maze_size, num_actors, total_steps, seed=42):
    """MuJoCo: N forked processes, each with its own MuJoCo env."""
    env_id = f"memory_maze:MemoryMaze-{maze_size}-v0"
    r = benchmark_multiprocess(env_id, num_actors, total_steps, seed)
    r["backend"] = "mujoco"
    r["num_envs"] = num_actors
    return r


def benchmark_genesis_single(maze_size, num_actors, total_steps, physics_timestep, seed=42):
    """Genesis single-env: N forked processes, each with its own Genesis env."""
    env_id = f"memory_maze:MemoryMaze-{maze_size}-Genesis-v0"
    env_kwargs = {"physics_timestep": physics_timestep}
    r = benchmark_multiprocess(env_id, num_actors, total_steps, seed, env_kwargs)
    r["backend"] = "genesis"
    r["num_envs"] = num_actors
    return r


def benchmark_genesis_batched(maze_size, num_envs, total_steps, physics_timestep, seed=42):
    """Genesis batched: single process, vectorized step over N envs on GPU."""
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

    # Warmup (Genesis JIT + BatchRenderer init)
    for _ in range(5):
        env.step(rng.randint(0, 6, size=num_envs))

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MuJoCo vs Genesis backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backends", default="mujoco,genesis-batched",
                        help="Comma-separated: mujoco, genesis, genesis-batched (default: mujoco,genesis-batched)")
    parser.add_argument("--actors", default="1,4,8,32",
                        help="Comma-separated actor/env counts (default: 1,4,8,32)")
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
    print()
    print("Note: MuJoCo and Genesis single-env use N forked processes (like real IMPALA).")
    print("      Genesis batched uses a single process with vectorized GPU step.")
    if "genesis" in backends or "genesis-batched" in backends:
        print(f"      Genesis physics_timestep: {args.physics_timestep}")
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
    print("-" * 70)
    for r in results:
        notes = r.get("error", "")
        print(f"{r['backend']:<20} {r['num_envs']:>5} {r['total_steps']:>7} "
              f"{r['elapsed_s']:>9.2f} {r['sps']:>8.1f} {notes}")

    # Print speedup summary if we have both mujoco and genesis-batched
    mujoco_results = [r for r in results if r["backend"] == "mujoco" and r["sps"] > 0]
    batched_results = [r for r in results if r["backend"] == "genesis-batched" and r["sps"] > 0]
    if mujoco_results and batched_results:
        print()
        print("Speedup (Genesis batched vs MuJoCo at matching env count):")
        for br in batched_results:
            # Find MuJoCo result with same env count, or closest
            mr = next((m for m in mujoco_results if m["num_envs"] == br["num_envs"]), None)
            if mr:
                speedup = br["sps"] / mr["sps"]
                print(f"  n={br['num_envs']}: {br['sps']:.0f} vs {mr['sps']:.0f} SPS = {speedup:.1f}x")


if __name__ == "__main__":
    main()
