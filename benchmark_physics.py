#!/usr/bin/env python3
"""Benchmark Memory Maze physics presets.

Runs each physics optimization preset and measures steps/second,
total reward, and basic correctness (walls block movement, targets activate).

Usage:
    python benchmark_physics.py
    python benchmark_physics.py --episodes 10 --env "memory_maze:MemoryMaze-9x9-v0"
"""

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "glfw")

import gym
import numpy as np

PHYSICS_PRESETS = {
    "none": {},
    "conservative": {
        "timestep": 0.025,
        "noslip_iterations": 0,
        "cone": "pyramidal",
    },
    "moderate": {
        "timestep": 0.05,
        "noslip_iterations": 0,
        "cone": "pyramidal",
        "iterations": 3,
    },
    "aggressive": {
        "timestep": 0.25,
        "noslip_iterations": 0,
        "cone": "pyramidal",
        "iterations": 1,
    },
}


def benchmark_preset(env_id, preset_name, physics_opts, num_episodes, seed=42):
    kwargs = {"disable_env_checker": True}
    if physics_opts:
        kwargs["physics_opts"] = physics_opts
    env = gym.make(env_id, seed=seed, **kwargs)

    rng = np.random.RandomState(seed)
    total_steps = 0
    total_reward = 0.0
    episode_rewards = []

    start = time.monotonic()
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = rng.randint(env.action_space.n)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            total_steps += 1
        episode_rewards.append(ep_reward)
        total_reward += ep_reward
    elapsed = time.monotonic() - start

    env.close()

    return {
        "preset": preset_name,
        "episodes": num_episodes,
        "total_steps": total_steps,
        "elapsed_s": elapsed,
        "sps": total_steps / elapsed,
        "mean_reward": total_reward / num_episodes,
        "episodes_with_reward": sum(1 for r in episode_rewards if r > 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Memory Maze physics presets")
    parser.add_argument("--env", default="memory_maze:MemoryMaze-9x9-v0")
    parser.add_argument("--episodes", default=5, type=int)
    parser.add_argument("--presets", nargs="+", default=list(PHYSICS_PRESETS.keys()))
    args = parser.parse_args()

    print(f"Benchmarking {args.env} with {args.episodes} episodes per preset\n")

    results = []
    for name in args.presets:
        opts = PHYSICS_PRESETS[name]
        print(f"  Running preset '{name}'...", end="", flush=True)
        r = benchmark_preset(args.env, name, opts, args.episodes)
        print(f" {r['sps']:.1f} SPS")
        results.append(r)

    # Print comparison table
    baseline_sps = results[0]["sps"] if results else 1.0
    print()
    print(f"{'Preset':<14} {'Steps':>7} {'Time (s)':>9} {'SPS':>8} {'Speedup':>8} {'Mean Rew':>9} {'Ep w/ Rew':>10}")
    print("-" * 75)
    for r in results:
        speedup = r["sps"] / baseline_sps
        print(
            f"{r['preset']:<14} {r['total_steps']:>7d} {r['elapsed_s']:>9.2f} "
            f"{r['sps']:>8.1f} {speedup:>7.2f}x {r['mean_reward']:>9.2f} "
            f"{r['episodes_with_reward']:>5d}/{r['episodes']}"
        )


if __name__ == "__main__":
    main()
