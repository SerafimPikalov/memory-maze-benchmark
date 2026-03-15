#!/usr/bin/env python3
"""Train IMPALA (V-trace) agent on Memory Maze using TorchBeast MonoBeast.

Adapts Facebook's TorchBeast MonoBeast (single-machine IMPALA) to work with
the Memory Maze benchmark. Uses an IMPALA-style ResNet encoder with LSTM
for long-term memory — critical for the revisitation task in Memory Maze.

Setup:
    conda create -n impala python=3.11 -y
    conda activate impala
    pip install torch numpy "gym>=0.21,<1.0"
    cd memory-maze && pip install -e . && cd ..
    # TorchBeast modules are vendored in torchbeast/ — no build needed.

Usage:
    # Train on 9x9 maze (default)
    python train_impala.py --num_actors 8 --total_steps 10_000_000

    # Smoke test
    python train_impala.py --num_actors 2 --total_steps 1000 --batch_size 2

    # Evaluate a trained agent
    python train_impala.py --mode test --xpid <experiment_id>

Based on: github.com/facebookresearch/torchbeast (Apache 2.0)
"""

import argparse

import logging
import math
import os
import pprint
import signal
import sys
import threading
import time
import timeit
import traceback
import typing

os.environ["OMP_NUM_THREADS"] = "1"
# macOS needs glfw; headless Linux needs egl
if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
# Prevent matplotlib (imported by Genesis) from initializing Tk in spawned
# subprocesses — Tk requires macOS main thread and crashes in child processes.
os.environ.setdefault("MPLBACKEND", "Agg")
# Prevent Genesis's viewer.py from creating a Tk root at import time —
# this crashes in spawned subprocesses on macOS (Cocoa NSApplication error).
os.environ["GENESIS_SKIP_TK_INIT"] = "1"


# ─── Import TorchBeast core modules ─────────────────────────────────
# TorchBeast pure-Python core modules are vendored in torchbeast/.

from torchbeast.core import vtrace
from torchbeast.core import environment as _env_mod
from torchbeast.core import file_writer as _fw_mod
from torchbeast.core import prof as _prof_mod

import gym
import numpy as np
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F


# ─── CLI ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="IMPALA on Memory Maze (MonoBeast)")

parser.add_argument("--env", type=str, default="memory_maze:MemoryMaze-9x9-v0",
                    help="Gym environment.")
parser.add_argument("--mode", default="train", choices=["train", "test"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: auto-generated).")
parser.add_argument("--physics_preset", default="none",
                    choices=["none", "conservative", "moderate", "aggressive"],
                    help="MuJoCo physics optimization preset for faster simulation.")
parser.add_argument("--backend", default="mujoco", choices=["mujoco", "genesis"],
                    help="Physics backend: 'mujoco' (default) or 'genesis' (GPU-accelerated).")
parser.add_argument("--batched", action="store_true",
                    help="Use batched Genesis backend (requires --backend genesis).")
parser.add_argument("--envs_per_batch", default=None, type=int,
                    help="Envs per batched actor (default: num_actors, i.e. single actor).")
parser.add_argument("--physics_timestep", default=None, type=float,
                    help="Override Genesis physics timestep (default: 0.005, use 0.05 for 10x fewer substeps).")
parser.add_argument("--n_batched_actors", default=1, type=int,
                    help="Number of batched actor processes (default: 1, single actor on main thread).")
parser.add_argument("--use_batch_renderer", action="store_true",
                    help="Use Madrona BatchRenderer (GPU rasterizer) instead of OpenGL Rasterizer. "
                         "Requires CUDA. Automatically enabled in --batched mode.")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir",
                    default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=128, type=int,
                    help="Number of actors (default: 128, paper uses 128).")
parser.add_argument("--total_steps", default=100_000_000, type=int,
                    help="Total environment steps to train for (paper: 100M).")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=100, type=int,
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", default=2, type=int,
                    help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.001, type=float,
                    help="Entropy cost/multiplier (paper: 0.001).")
parser.add_argument("--baseline_cost", default=0.5, type=float,
                    help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99, type=float,
                    help="Discounting factor.")
parser.add_argument("--reward_clipping", default="none",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings (paper Table E.2: Adam, lr=2e-4, eps=1e-7).
parser.add_argument("--learning_rate", default=0.0002, type=float,
                    help="Learning rate (paper: 2e-4).")
parser.add_argument("--epsilon", default=1e-7, type=float,
                    help="Adam epsilon (paper: 1e-7).")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")

# Reproducibility.
parser.add_argument("--seed", type=int, default=None,
                    help="Global seed for reproducible runs (derives maze_seed=N, training_seed=N+1000000).")

# Episode recording.
parser.add_argument("--record_interval", default=100, type=int,
                    help="Record one episode every N completed episodes (0 = disabled).")
parser.add_argument("--record_dir", default=None, type=str,
                    help="Directory for episode recordings (default: {savedir}/{xpid}/recordings/).")

# Weights & Biases.
parser.add_argument("--wandb", action="store_true",
                    help="Enable Weights & Biases logging.")
parser.add_argument("--wandb_project", default="memorymaze",
                    help="W&B project name.")
parser.add_argument("--wandb_entity", default=None,
                    help="W&B entity (team or username).")


# Physics optimization presets — reduce MuJoCo computation for the simple
# rolling-ball agent. The ball has heavy damping (roll=5, steer=20) which
# acts as a natural stabilizer, making aggressive timestep increases safe.
PHYSICS_PRESETS = {
    "none": {},
    "conservative": {       # ~5-8x physics speedup
        "timestep": 0.025,  # 10 substeps (was 50)
        "noslip_iterations": 0,
        "cone": "pyramidal",
    },
    "moderate": {            # ~10-15x physics speedup
        "timestep": 0.05,   # 5 substeps
        "noslip_iterations": 0,
        "cone": "pyramidal",
        "iterations": 3,
    },
    "aggressive": {          # ~25-50x physics speedup
        "timestep": 0.25,   # 1 substep (control == physics timestep)
        "noslip_iterations": 0,
        "cone": "pyramidal",
        "iterations": 1,
    },
}


logging.basicConfig(
    format="[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] %(message)s",
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


# ─── Environment ─────────────────────────────────────────────────────

class MemoryMazeWrapper(gym.Wrapper):
    """Transpose Memory Maze observation from HWC uint8 to CHW uint8."""

    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=np.uint8,
        )

    def _transpose(self, obs):
        return np.ascontiguousarray(np.transpose(obs, (2, 0, 1)))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._transpose(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._transpose(obs), reward, done, info


def create_env(flags, actor_index=0):
    """Create Memory Maze gym env with CHW observation layout."""
    # disable_env_checker: gym 0.26's checker uses np.bool8, removed in NumPy 2.
    kwargs = {"disable_env_checker": True}
    env_id = flags.env
    if getattr(flags, "backend", "mujoco") == "genesis" and "-Genesis-" not in env_id:
        # Route to Genesis-backed environment
        env_id = env_id.replace("-v0", "-Genesis-v0")
        if getattr(flags, "physics_timestep", None) is not None:
            kwargs["physics_timestep"] = flags.physics_timestep
        if getattr(flags, "use_batch_renderer", False):
            kwargs["use_batch_renderer"] = True
    else:
        physics_opts = PHYSICS_PRESETS.get(getattr(flags, "physics_preset", "none"), {})
        if physics_opts:
            kwargs["physics_opts"] = physics_opts
    # Guard: ExtraObs variants return Dict obs, incompatible with training
    if "-ExtraObs-" in env_id or env_id.endswith("-ExtraObs-v0") or env_id.endswith("-ExtraObs-Genesis-v0"):
        raise ValueError(
            f"Environment '{env_id}' uses ExtraObs (dict observations) which is "
            "incompatible with training. ExtraObs is for analysis/notebooks only. "
            "Use a standard variant (e.g. MemoryMaze-9x9-v0) for training."
        )
    if flags.seed is not None:
        kwargs["seed"] = flags.seed + actor_index
    gym_env = gym.make(env_id, **kwargs)
    return MemoryMazeWrapper(gym_env)


def get_env_metadata(flags):
    """Determine obs_shape and num_actions without creating a rendering env.

    On macOS, creating a MuJoCo/dm_control env initializes GLFW which uses
    Cocoa/CoreFoundation.  This state cannot survive fork(), so we must NOT
    create any env in the parent process before forking actor processes.
    Instead, parse the env ID to get the metadata we need.
    """
    env_id = flags.env
    if ":" in env_id:
        env_id = env_id.split(":")[1]

    if "MemoryMaze" in env_id:
        num_actions = 6  # All Memory Maze envs use 6 discrete actions.
        if "-HD-" in env_id or "-Top-" in env_id:
            obs_shape = (3, 256, 256)
        else:
            obs_shape = (3, 64, 64)
        return obs_shape, num_actions

    # Fallback: create env in a subprocess to probe its spec.
    import multiprocessing
    def _probe(q, env_name):
        if sys.platform == "darwin":
            os.environ.setdefault("MUJOCO_GL", "glfw")
        else:
            os.environ.setdefault("MUJOCO_GL", "egl")
        _env = create_env(type("F", (), {"env": env_name})())
        q.put((_env.observation_space.shape, _env.action_space.n))
        _env.close()

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_probe, args=(q, flags.env))
    p.start()
    result = q.get(timeout=60)
    p.join()
    return result


# ─── Network ─────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block: ReLU → Conv3x3 → ReLU → Conv3x3 + skip."""

    def __init__(self, depth):
        super().__init__()
        self.conv0 = nn.Conv2d(depth, depth, 3, padding=1)
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv0(out)
        out = F.relu(out)
        out = self.conv1(out)
        return out + x


class ConvBlock(nn.Module):
    """IMPALA conv block: Conv3x3 → MaxPool → 2x ResBlock."""

    def __init__(self, in_channels, depth):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, depth, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res0 = ResBlock(depth)
        self.res1 = ResBlock(depth)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res0(x)
        x = self.res1(x)
        return x


class MemoryMazeNet(nn.Module):
    """IMPALA-style ResNet + LSTM for Memory Maze.

    Architecture:
        RGB (3,64,64) → ResNet [16,32,32] → FC(256)
        → concat(features, one_hot_action, reward)
        → LSTM(256) → policy(6) + value(1)
    """

    def __init__(self, observation_shape, num_actions):
        super().__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        c, h, w = observation_shape  # (3, 64, 64)

        # IMPALA ResNet: 3 blocks, each halves spatial dims.
        self.block0 = ConvBlock(c, 16)    # 64 → 32
        self.block1 = ConvBlock(16, 32)   # 32 → 16
        self.block2 = ConvBlock(32, 32)   # 16 → 8

        cnn_out_size = 32 * (h // 8) * (w // 8)  # 32*8*8 = 2048
        self.fc = nn.Linear(cnn_out_size, 256)

        # LSTM input: visual features + one-hot last action + scalar reward.
        lstm_input_size = 256 + num_actions + 1
        self.core = nn.LSTM(lstm_input_size, 256, num_layers=1)

        self.policy = nn.Linear(256, num_actions)
        self.baseline = nn.Linear(256, 1)

    def initial_state(self, batch_size):
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W]
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # [T*B, C, H, W]
        x = x.float() / 255.0

        # IMPALA ResNet encoder.
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        # Concat auxiliary inputs.
        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        # Step-by-step LSTM with state reset on episode boundaries.
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for inp, nd in zip(core_input.unbind(), notdone.unbind()):
            # Zero out LSTM state when an episode ended.
            nd = nd.view(1, -1, 1)
            core_state = tuple(nd * s for s in core_state)
            output, core_state = self.core(inp.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1
            )
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


Net = MemoryMazeNet


# ─── Losses ──────────────────────────────────────────────────────────

def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


# ─── Locks for learner threads ────────────────────────────────────────
_get_batch_lock = threading.Lock()
_learn_lock = threading.Lock()

# ─── Episode recording ────────────────────────────────────────────────

def save_recording(record_dir, episode_count, step, actor_index,
                   frames, actions, episode_return, wandb_video=False):
    """Save an episode recording as compressed npz. Runs on a background thread.

    If wandb_video=True, also log the episode as a wandb.Video to the active
    W&B run (only works when wandb.init() was called in this process).
    """
    try:
        os.makedirs(record_dir, exist_ok=True)
        tag = f"ep{episode_count}_step{step}_actor{actor_index}"
        path = os.path.join(record_dir, f"{tag}.npz")
        stacked_frames = np.stack(frames)                  # (T, H, W, 3) uint8
        np.savez_compressed(path,
            frames=stacked_frames,
            actions=np.array([a for a, _ in actions]),     # (T,) int
            probs=np.stack([p for _, p in actions]),        # (T, 6) float32
            episode_return=episode_return,
        )
        logging.info("Saved recording: %s (%d frames)", path, len(frames))

        if wandb_video:
            try:
                import wandb
                if wandb.run is not None:
                    # wandb.Video expects (T, C, H, W) uint8
                    video_data = stacked_frames.transpose(0, 3, 1, 2)
                    wandb.log({
                        "episode_video": wandb.Video(video_data, fps=4, format="mp4"),
                        "episode_video_return": episode_return,
                        "episode_video_step": step,
                    }, commit=False)
                    logging.info("Logged video to W&B: %s (return=%.1f)", tag, episode_return)
            except Exception:
                logging.warning("Failed to log video to W&B", exc_info=True)
    except Exception:
        logging.error("Failed to save recording", exc_info=True)


# ─── MonoBeast core (adapted from torchbeast/monobeast.py) ───────────

def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = _prof_mod.Timings()

        gym_env = create_env(flags, actor_index=actor_index)
        env = _env_mod.Environment(gym_env)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)

        # Recording state (actor 0 only, non-batched mode)
        should_record = (actor_index == 0 and flags.record_interval > 0)
        rec_episode_count = 0
        rec_is_recording = False
        rec_start_on_next_reset = False
        rec_next_record_at = flags.record_interval if should_record else float("inf")
        rec_frames = []
        rec_actions = []
        rec_episode_return = 0.0
        if should_record:
            rec_dir = flags.record_dir or os.path.join(
                flags.savedir, flags.xpid, "recordings")

        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")

                env_output = env.step(agent_output["action"])

                timings.time("step")

                # --- Episode recording (actor 0 only) ---
                if should_record:
                    done = env_output["done"].item()
                    rec_just_started = False
                    if done:
                        rec_episode_count += 1
                        if rec_is_recording:
                            # End recording — save on background thread
                            threading.Thread(
                                target=save_recording,
                                args=(rec_dir, rec_episode_count, 0, actor_index,
                                      rec_frames, rec_actions, rec_episode_return),
                                daemon=True,
                            ).start()
                            rec_is_recording = False
                        if not rec_is_recording and rec_episode_count >= rec_next_record_at:
                            rec_start_on_next_reset = True
                            rec_next_record_at = rec_episode_count + flags.record_interval
                    if rec_start_on_next_reset and done:
                        # TorchBeast auto-resets: frame is already the new episode's
                        # first obs, but action/reward belong to the old episode.
                        # Start recording on the *next* step when action matches frame.
                        rec_is_recording = True
                        rec_start_on_next_reset = False
                        rec_frames = []
                        rec_actions = []
                        rec_episode_return = 0.0
                        rec_just_started = True
                    if rec_is_recording and not rec_just_started:
                        # Frame: (1, 1, C, H, W) → (H, W, C) uint8
                        frame = env_output["frame"].squeeze().numpy().transpose(1, 2, 0)
                        probs = F.softmax(agent_output["policy_logits"].squeeze(), dim=-1).cpu().numpy()
                        action = agent_output["action"].item()
                        rec_frames.append(frame)
                        rec_actions.append((action, probs))
                        rec_episode_return += env_output["reward"].item()

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def _parse_maze_size(env_id):
    """Extract maze size from env ID string, e.g. 'memory_maze:MemoryMaze-9x9-v0' → 9."""
    import re
    m = re.search(r'(\d+)x\d+', env_id)
    if m:
        return int(m.group(1))
    return 9  # default


def act_batched(
    flags,
    actor_index: int,
    free_queue,
    full_queue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
    n_envs: int,
    batch_env=None,
    step_ref=None,
    actor_step_counter=None,
):
    """Batched actor: manages n_envs environments in a single Genesis scene.

    Replaces n_envs separate act() processes with one thread that steps all
    environments simultaneously. Fills n_envs buffer slots per unroll.
    """
    try:
        logging.info("Batched actor %i started with %d envs.", actor_index, n_envs)
        timings = _prof_mod.Timings()

        if batch_env is None:
            # In multi-process mode, set per-process kernel cache to avoid
            # race conditions when multiple processes compile Taichi kernels.
            if actor_step_counter is not None:
                os.environ["QD_OFFLINE_CACHE_FILE_PATH"] = f"/tmp/genesis_cache_{os.getpid()}"
            from memory_maze.genesis_backend import BatchGenesisMemoryMazeEnv
            maze_size = _parse_maze_size(flags.env)
            seed = flags.seed if flags.seed is not None else actor_index * 10000
            batch_kwargs = dict(
                n_envs=n_envs, maze_size=maze_size,
                camera_resolution=64, seed=seed,
            )
            if flags.physics_timestep is not None:
                batch_kwargs["physics_timestep"] = flags.physics_timestep
            batch_env = BatchGenesisMemoryMazeEnv(**batch_kwargs)

        # Initial reset — (N, H, W, 3) HWC
        obs_all = batch_env.reset()

        # Determine device from model parameters (CUDA when available)
        device = next(model.parameters()).device

        # Batched state tracking — all N envs packed into single tensors.
        # env_outputs:  dict of [1, N, ...] tensors
        # agent_states: tuple of (num_layers, N, hidden) tensors
        # agent_outputs: dict of [1, N, ...] tensors
        frames_init = torch.from_numpy(
            np.ascontiguousarray(obs_all.transpose(0, 3, 1, 2))  # (N, C, H, W)
        ).unsqueeze(0).to(device)  # (1, N, C, H, W)
        env_outputs = {
            "frame": frames_init,
            "reward": torch.zeros(1, n_envs, device=device),
            "done": torch.zeros(1, n_envs, dtype=torch.bool, device=device),
            "episode_return": torch.zeros(1, n_envs, device=device),
            "episode_step": torch.zeros(1, n_envs, dtype=torch.int32, device=device),
            "last_action": torch.zeros(1, n_envs, dtype=torch.int64, device=device),
        }
        agent_states = tuple(
            s.to(device) for s in model.initial_state(batch_size=n_envs)
        )
        with torch.no_grad():
            agent_outputs, _ = model(env_outputs, agent_states)

        # Pre-compute initial LSTM state template for episode resets (shape: num_layers, 1, hidden)
        _init_state = tuple(s.to(device) for s in model.initial_state(batch_size=1))

        # Per-env episode tracking
        episode_returns = np.zeros(n_envs, dtype=np.float32)
        episode_steps = np.zeros(n_envs, dtype=np.int32)

        # Recording state (batched mode)
        should_record = (flags.record_interval > 0)
        rec_episode_count = 0
        rec_is_recording = False
        rec_start_on_next_reset = False
        rec_next_record_at = flags.record_interval if should_record else float("inf")
        rec_env_idx = 0
        rec_frames = []
        rec_actions = []
        rec_episode_return = 0.0
        if should_record:
            rec_dir = flags.record_dir or os.path.join(
                flags.savedir, flags.xpid, "recordings")

        def _get_step(ref):
            if ref is None:
                return 0
            return ref.value if hasattr(ref, "value") else ref[0]

        while step_ref is None or _get_step(step_ref) < flags.total_steps:
            # Grab n_envs free buffer indices
            indices = []
            for _ in range(n_envs):
                index = free_queue.get()
                if index is None:
                    # Sentinel received — return any already-grabbed indices
                    for grabbed in indices:
                        free_queue.put(grabbed)
                    return
                indices.append(index)

            # Write old rollout ends to buffer slots
            for i in range(n_envs):
                idx = indices[i]
                for key in env_outputs:
                    buffers[key][idx][0, ...] = env_outputs[key][0, i]
                for key in agent_outputs:
                    buffers[key][idx][0, ...] = agent_outputs[key][0, i]
                for j in range(len(agent_states)):
                    initial_agent_state_buffers[idx][j][...] = agent_states[j][:, i:i+1, :]

            # Do new rollout
            for t in range(flags.unroll_length):
                timings.reset()

                # Batched model inference — env_outputs already [1, N, ...],
                # agent_states already (num_layers, N, hidden). No assembly needed.
                with torch.no_grad():
                    agent_outputs, agent_states = model(env_outputs, agent_states)

                actions = agent_outputs["action"].view(n_envs).cpu().numpy().astype(np.int64)

                timings.time("model")

                # Single batched step — returns (N, H, W, 3) HWC
                obs_all, rewards, dones, infos = batch_env.step(actions)

                timings.time("step")

                # --- Vectorized tensor construction (no per-env Python loop) ---
                frames_all = torch.from_numpy(
                    np.ascontiguousarray(obs_all.transpose(0, 3, 1, 2))  # (N, C, H, W)
                ).unsqueeze(0).to(device)  # (1, N, C, H, W)
                rewards_t = torch.from_numpy(
                    rewards.astype(np.float32)
                ).unsqueeze(0).to(device)  # (1, N)
                dones_t = torch.from_numpy(
                    dones.astype(bool)
                ).unsqueeze(0).to(device)  # (1, N)
                actions_t = torch.from_numpy(actions).unsqueeze(0).to(device)  # (1, N)

                # Vectorized episode tracking
                episode_returns += rewards
                episode_steps += 1
                ep_ret_snapshot = episode_returns.copy()
                ep_step_snapshot = episode_steps.copy()
                done_mask = dones.astype(bool)

                # --- Episode recording ---
                if should_record:
                    rec_just_started = False
                    n_done = int(done_mask.sum())
                    if n_done > 0:
                        rec_episode_count += n_done
                        if rec_is_recording and done_mask[rec_env_idx]:
                            threading.Thread(
                                target=save_recording,
                                args=(rec_dir, rec_episode_count,
                                      _get_step(step_ref), actor_index,
                                      rec_frames, rec_actions, rec_episode_return),
                                kwargs=dict(wandb_video=getattr(flags, "wandb", False)),
                                daemon=True,
                            ).start()
                            rec_is_recording = False
                        if not rec_is_recording and rec_episode_count >= rec_next_record_at:
                            rec_start_on_next_reset = True
                            rec_env_idx = np.random.randint(n_envs)
                            rec_next_record_at = rec_episode_count + flags.record_interval
                        if rec_start_on_next_reset and done_mask[rec_env_idx]:
                            # batch_env auto-resets: obs_all has post-reset frame but
                            # actions/logits belong to the old episode. Start collecting
                            # on the *next* step when action matches observation.
                            rec_is_recording = True
                            rec_start_on_next_reset = False
                            rec_frames = []
                            rec_actions = []
                            rec_episode_return = 0.0
                            rec_just_started = True
                    if rec_is_recording and not rec_just_started:
                        rec_frames.append(obs_all[rec_env_idx].copy())
                        probs = F.softmax(
                            agent_outputs["policy_logits"][0, rec_env_idx], dim=-1
                        ).cpu().numpy()
                        rec_actions.append((actions[rec_env_idx], probs))
                        rec_episode_return += rewards[rec_env_idx]

                episode_returns[done_mask] = 0.0
                episode_steps[done_mask] = 0

                # Vectorized LSTM state reset for done envs
                if np.any(done_mask):
                    done_broad = torch.from_numpy(done_mask).to(device).view(1, n_envs, 1)
                    agent_states = tuple(
                        torch.where(done_broad, _init_state[j].expand_as(agent_states[j]),
                                    agent_states[j])
                        for j in range(len(agent_states))
                    )

                ep_ret_t = torch.from_numpy(ep_ret_snapshot).unsqueeze(0).to(device)
                ep_step_t = torch.from_numpy(
                    ep_step_snapshot.astype(np.int32)
                ).unsqueeze(0).to(device)

                env_outputs = {
                    "frame": frames_all,
                    "reward": rewards_t,
                    "done": dones_t,
                    "episode_return": ep_ret_t,
                    "episode_step": ep_step_t,
                    "last_action": actions_t,
                }

                # Write to per-env buffer slots (indices differ, loop required)
                for i in range(n_envs):
                    idx = indices[i]
                    for key in env_outputs:
                        buffers[key][idx][t + 1, ...] = env_outputs[key][0, i]
                    for key in agent_outputs:
                        buffers[key][idx][t + 1, ...] = agent_outputs[key][0, i]

                timings.time("write")

            for idx in indices:
                full_queue.put(idx)

            # Update per-actor step counter for monitoring.
            if actor_step_counter is not None:
                actor_step_counter.value += n_envs * flags.unroll_length

        if actor_index == 0:
            logging.info("Batched actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error("Exception in batched actor %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
):
    with _get_batch_lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {
        k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()
    }
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state


def learn(
    flags,
    actor_model,
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
):
    """Performs a learning (optimization) step."""
    with _learn_lock:
        learner_outputs, learner_final_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1] for key, tensor in learner_outputs.items()
        }

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        # Normalize by batch size so changing --batch_size doesn't change
        # the effective learning rate.  Sum-over-time is kept (standard for RL).
        B = batch["reward"].shape[1]
        total_loss = (pg_loss + baseline_loss + entropy_loss) / B

        episode_returns = batch["episode_return"][batch["done"]]
        episode_lengths = batch["episode_step"][batch["done"]]
        # LSTM hidden state norm (h is the first element of the LSTM state tuple).
        lstm_hidden_norm = torch.mean(torch.norm(learner_final_state[0], dim=-1)).item()
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)

        # Behavioral metrics — action distribution
        action_probs = F.softmax(learner_outputs["policy_logits"], dim=-1)  # (T, B, 6)
        mean_action_dist = action_probs.mean(dim=(0, 1))  # (6,)

        # Behavioral metrics — action repeat rate
        batch_actions = batch["action"]  # (T, B)
        repeats = (batch_actions[1:] == batch_actions[:-1]).float()  # (T-1, B)
        per_traj_repeat = repeats.mean(dim=0)  # (B,)

        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item() if len(episode_returns) > 0 else float("nan"),
            "mean_episode_length": torch.mean(episode_lengths.float()).item() if len(episode_lengths) > 0 else float("nan"),
            "total_loss": total_loss.item(),
            "pg_loss": (pg_loss / B).item(),
            "baseline_loss": (baseline_loss / B).item(),
            "entropy_loss": (entropy_loss / B).item(),
            "lstm_hidden_norm": lstm_hidden_norm,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "episodes_per_batch": batch["done"].sum().item(),
            "action_dist_noop": mean_action_dist[0].item(),
            "action_dist_forward": mean_action_dist[1].item(),
            "action_dist_left": mean_action_dist[2].item(),
            "action_dist_right": mean_action_dist[3].item(),
            "action_dist_fwd_left": mean_action_dist[4].item(),
            "action_dist_fwd_right": mean_action_dist[5].item(),
            "mean_action_repeat_rate": per_traj_repeat.mean().item(),
            "std_action_repeat_rate": per_traj_repeat.std().item(),
            "forward_fraction": (batch_actions == 1).float().mean().item(),
        }
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


# ─── Train & test ────────────────────────────────────────────────────

def train(flags):
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = _fw_mod.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    # Weights & Biases.
    if flags.wandb:
        import wandb
        wandb.init(
            project=flags.wandb_project,
            entity=flags.wandb_entity,
            name=flags.xpid,
            config={
                "backend": flags.backend,
                "agent": "impala",
                "env": flags.env,
                "batched": getattr(flags, "batched", False),
                "num_actors": flags.num_actors,
                "batch_size": flags.batch_size,
                "total_steps": flags.total_steps,
                "learning_rate": flags.learning_rate,
                "entropy_cost": flags.entropy_cost,
            },
            tags=["impala", flags.backend],
        )

    # Reproducibility seeding.
    if flags.seed is not None:
        training_seed = flags.seed + 1_000_000
        torch.manual_seed(training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_seed)
        logging.info("Seed: maze_seed=%d, training_seed=%d", flags.seed, training_seed)

    if flags.num_buffers is None:
        flags.num_buffers = 2 * flags.num_actors + flags.batch_size
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    obs_shape, num_actions = get_env_metadata(flags)
    model = Net(obs_shape, num_actions)
    buffers = create_buffers(flags, obs_shape, num_actions)

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    actor_threads = []

    if flags.batched:
        assert flags.backend == "genesis", "--batched requires --backend genesis"
        n_batched_actors = getattr(flags, "n_batched_actors", 1)
        envs_per_batch = flags.envs_per_batch or flags.num_actors
        if n_batched_actors > 1:
            # Multi-process batched mode: split envs across N spawned processes.
            if flags.num_actors % n_batched_actors != 0:
                raise ValueError(
                    f"num_actors ({flags.num_actors}) must be divisible by "
                    f"n_batched_actors ({n_batched_actors})"
                )
            envs_per_batch = flags.num_actors // n_batched_actors
            logging.info(
                "Batched mode: %d actor process(es), %d envs each (total %d envs)",
                n_batched_actors, envs_per_batch, n_batched_actors * envs_per_batch,
            )
            # Use spawn context — Genesis/Taichi require clean process state.
            ctx = mp.get_context("spawn")
            free_queue = ctx.SimpleQueue()
            full_queue = ctx.SimpleQueue()
        else:
            # Single batched actor on main thread (original behavior).
            if flags.envs_per_batch and flags.envs_per_batch < flags.num_actors:
                logging.warning(
                    "--envs_per_batch=%d ignored: Genesis only supports one scene "
                    "per process. Using single actor with %d envs.",
                    flags.envs_per_batch, flags.num_actors,
                )
                envs_per_batch = flags.num_actors
            logging.info(
                "Batched mode: 1 actor (main thread), %d envs (total %d envs)",
                envs_per_batch, envs_per_batch,
            )
            # Use threading queues — single batched actor is on main thread
            import queue
            free_queue = queue.SimpleQueue()
            full_queue = queue.SimpleQueue()
    else:
        # On Linux, "fork" avoids the overhead of re-importing modules in each actor.
        # On macOS, "spawn" is required: forked children inherit Metal/OpenGL driver
        # state that crashes when MuJoCo creates a rendering context.
        ctx = mp.get_context("fork" if sys.platform == "linux" else "spawn")
        free_queue = ctx.SimpleQueue()
        full_queue = ctx.SimpleQueue()

        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(
                    flags, i, free_queue, full_queue,
                    model, buffers, initial_agent_state_buffers,
                ),
            )
            actor.start()
            actor_processes.append(actor)

    learner_model = Net(obs_shape, num_actions).to(device=flags.device)

    optimizer = torch.optim.Adam(
        learner_model.parameters(),
        lr=flags.learning_rate,
        eps=flags.epsilon,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "mean_episode_length",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "lstm_hidden_norm",
        "grad_norm",
        "episodes_per_batch",
        "action_dist_noop",
        "action_dist_forward",
        "action_dist_left",
        "action_dist_right",
        "action_dist_fwd_left",
        "action_dist_fwd_right",
        "mean_action_repeat_rate",
        "std_action_repeat_rate",
        "forward_fraction",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}
    # step_counter: shared reference so actors can read the learner's step count.
    # For multi-process batched actors, use multiprocessing.Value (cross-process).
    # For single-actor/threading mode, use a plain list (no IPC overhead).
    n_batched_actors = getattr(flags, "n_batched_actors", 1)
    if flags.batched and n_batched_actors > 1:
        step_counter = ctx.Value("i", 0)
    else:
        step_counter = [0]

    # Auto-resume from checkpoint.
    if os.path.exists(checkpointpath):
        logging.info("Resuming from checkpoint: %s", checkpointpath)
        ckpt = torch.load(checkpointpath, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        step = ckpt.get("step", 0)
        if hasattr(step_counter, "value"):
            step_counter.value = step
        else:
            step_counter[0] = step
        learner_model.load_state_dict(ckpt["model_state_dict"])
        logging.info("Resumed at step %d", step)

    _log_lock = threading.Lock()
    # Accumulate episode returns across learner batches so the monitor thread
    # can report them even if the stats dict gets overwritten.
    _pending_episode_returns = []

    def batch_and_learn(i):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = _prof_mod.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags, free_queue, full_queue,
                buffers, initial_agent_state_buffers, timings,
            )
            stats = learn(
                flags, model, learner_model,
                batch, agent_state, optimizer, scheduler,
            )
            timings.time("learn")
            with _log_lock:
                # Accumulate episode returns for the monitor thread.
                if stats.get("episode_returns"):
                    _pending_episode_returns.extend(stats["episode_returns"])
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B
                if hasattr(step_counter, "value"):
                    step_counter.value = step
                else:
                    step_counter[0] = step

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,),
            daemon=flags.batched,  # daemon in batched mode to avoid deadlock on exit
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
                "step": step,
            },
            checkpointpath,
        )

    # Handle SIGTERM for graceful shutdown on spot instance preemption.
    def _sigterm_handler(signum, frame):
        logging.info("SIGTERM received — saving emergency checkpoint...")
        checkpoint()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    timer = timeit.default_timer

    def monitor_loop():
        """Periodic logging and checkpointing (runs on a thread for batched mode)."""
        nonlocal step, stats
        try:
            last_checkpoint_time = timer()
            while step < flags.total_steps:
                start_step = step
                start_time = timer()
                time.sleep(5)

                if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                    checkpoint()
                    last_checkpoint_time = timer()

                sps = (step - start_step) / (timer() - start_time)
                if stats.get("episode_returns", None):
                    mean_return = (
                        "Return per episode: %.1f. " % stats["mean_episode_return"]
                    )
                else:
                    mean_return = ""
                total_loss = stats.get("total_loss", float("inf"))
                if flags.wandb:
                    wandb_data = {"step": step, "sps": sps}
                    wandb_data.update({k: stats[k] for k in stat_keys if k in stats and not (isinstance(stats[k], float) and math.isnan(stats[k]))})
                    wandb.log(wandb_data, step=step)
                logging.info(
                    "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                    step, sps, total_loss, mean_return,
                    pprint.pformat(stats),
                )
        except Exception:
            pass

    if flags.batched:
        if n_batched_actors > 1:
            # --- Multi-process batched mode ---
            # Each spawned child will call gs.init() inside
            # BatchGenesisMemoryMazeEnv.__init__().
            # Parent does NOT call gs.init().

            # Per-actor step counters for SPS monitoring.
            actor_step_counters = [
                ctx.Value("i", 0) for _ in range(n_batched_actors)
            ]

            for i in range(n_batched_actors):
                actor = ctx.Process(
                    target=act_batched,
                    args=(
                        flags, i, free_queue, full_queue,
                        model, buffers, initial_agent_state_buffers,
                        envs_per_batch,
                    ),
                    kwargs=dict(
                        step_ref=step_counter,
                        actor_step_counter=actor_step_counters[i],
                    ),
                )
                actor.start()
                actor_processes.append(actor)

            # Monitor loop with dead actor detection.
            def monitor_loop_multi():
                nonlocal step, stats
                # Track which .npz recordings we've already logged to W&B.
                wandb_logged_recordings = set()
                last_wandb_step = -1
                try:
                    last_checkpoint_time = timer()
                    while step < flags.total_steps:
                        start_step = step
                        start_time = timer()
                        time.sleep(5)

                        # Dead actor detection.
                        for ai, actor in enumerate(actor_processes):
                            if not actor.is_alive():
                                logging.error(
                                    "Batched actor %d died (exit code %s). "
                                    "Aborting training to prevent deadlock.",
                                    ai, actor.exitcode,
                                )
                                # Signal all remaining actors to stop.
                                for _ in range(n_batched_actors * envs_per_batch):
                                    free_queue.put(None)
                                return

                        if timer() - last_checkpoint_time > 10 * 60:
                            checkpoint()
                            last_checkpoint_time = timer()

                        sps = (step - start_step) / (timer() - start_time)

                        # Per-actor SPS logging.
                        actor_sps_parts = []
                        for ai in range(n_batched_actors):
                            a_steps = actor_step_counters[ai].value
                            actor_sps_parts.append(f"actor{ai}={a_steps}steps")
                        actor_sps_str = ", ".join(actor_sps_parts)

                        # Drain accumulated episode returns from learner thread.
                        with _log_lock:
                            pending_returns = list(_pending_episode_returns)
                            _pending_episode_returns.clear()
                        if pending_returns:
                            mean_ep_return = sum(pending_returns) / len(pending_returns)
                            mean_return = "Return per episode: %.1f. " % mean_ep_return
                        else:
                            mean_ep_return = None
                            mean_return = ""
                        total_loss = stats.get("total_loss", float("inf"))
                        if flags.wandb and step != last_wandb_step:
                            last_wandb_step = step
                            wandb_data = {"step": step, "sps": sps}
                            wandb_data.update({k: stats[k] for k in stat_keys if k in stats and not (isinstance(stats[k], float) and math.isnan(stats[k]))})
                            # Override with accumulated episode returns.
                            if mean_ep_return is not None:
                                wandb_data["mean_episode_return"] = mean_ep_return
                                wandb_data["mean_episode_length"] = 1000.0
                                wandb_data["episodes_per_batch"] = len(pending_returns)
                            # Log new episode recordings saved by actor subprocesses.
                            rec_dir = flags.record_dir or os.path.join(
                                flags.savedir, flags.xpid, "recordings")
                            if os.path.isdir(rec_dir):
                                for fname in sorted(os.listdir(rec_dir)):
                                    if fname.endswith(".npz") and fname not in wandb_logged_recordings:
                                        wandb_logged_recordings.add(fname)
                                        try:
                                            data = np.load(os.path.join(rec_dir, fname))
                                            video_data = data["frames"].transpose(0, 3, 1, 2)  # (T,C,H,W)
                                            ep_ret = float(data["episode_return"])
                                            wandb_data["episode_video"] = wandb.Video(
                                                video_data, fps=4, format="mp4")
                                            wandb_data["episode_video_return"] = ep_ret
                                            logging.info("Logged recording to W&B: %s (return=%.1f)", fname, ep_ret)
                                        except Exception:
                                            logging.warning("Failed to log recording %s to W&B", fname, exc_info=True)
                            wandb.log(wandb_data, step=step)
                        logging.info(
                            "Steps %i @ %.1f SPS. Loss %f. %s[%s] Stats:\n%s",
                            step, sps, total_loss, mean_return, actor_sps_str,
                            pprint.pformat(stats),
                        )
                except Exception:
                    logging.error("Monitor loop exception:", exc_info=True)

            monitor_thread = threading.Thread(
                target=monitor_loop_multi, name="monitor", daemon=True,
            )
            monitor_thread.start()

            try:
                # Wait for learner threads to complete.
                for thread in threads:
                    thread.join()
                logging.info("Learning finished after %d steps.", step)
            except KeyboardInterrupt:
                pass
            finally:
                # Graceful shutdown: send sentinel None per env slot per actor.
                for _ in range(n_batched_actors * envs_per_batch):
                    free_queue.put(None)
                for actor in actor_processes:
                    actor.join(timeout=10)
                    if actor.is_alive():
                        logging.warning("Actor %d did not exit, terminating.", actor.pid)
                        actor.terminate()
        else:
            # --- Single batched actor on main thread (original behavior) ---
            # Initialize Genesis here (after PyTorch optimizer setup) to avoid
            # corrupting setuptools/distutils imports that triton needs.
            import genesis as gs
            try:
                import gs_madrona
                _batch_renderer_available = True
            except ImportError:
                _batch_renderer_available = False
            if not gs._initialized:
                if not torch.cuda.is_available():
                    raise RuntimeError("Batched Genesis mode requires CUDA. No GPU detected.")
                gs.init(backend=gs.cuda, logging_level='warning')

            # Monitor and learner run on background threads.
            monitor_thread = threading.Thread(
                target=monitor_loop, name="monitor", daemon=True,
            )
            monitor_thread.start()

            try:
                act_batched(
                    flags, 0, free_queue, full_queue,
                    model, buffers, initial_agent_state_buffers,
                    envs_per_batch,
                    step_ref=step_counter,
                )
            except KeyboardInterrupt:
                return
            else:
                # Learner threads may be blocked in full_queue.get().
                # Join with timeout; they are daemon threads so they won't
                # prevent process exit.
                for thread in threads:
                    thread.join(timeout=5)
                logging.info("Learning finished after %d steps.", step)
    else:
        try:
            monitor_loop()
            for thread in threads:
                thread.join()
            logging.info("Learning finished after %d steps.", step)
        except KeyboardInterrupt:
            pass  # Fall through to finally for cleanup.
        finally:
            for _ in range(flags.num_actors):
                free_queue.put(None)
            for actor in actor_processes:
                actor.join(timeout=1)

    checkpoint()
    plogger.close()
    if flags.wandb:
        wandb.finish()


def test(flags, num_episodes: int = 10):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser(
                "%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar")
            )
        )

    # For Genesis: init backend before creating env.
    # BatchRenderer requires gs.cuda; OpenGL Rasterizer uses gs.cpu.
    use_cuda = False
    if getattr(flags, "backend", "mujoco") == "genesis":
        import genesis as gs
        use_cuda = getattr(flags, "use_batch_renderer", False)
        if not gs._initialized:
            gs.init(backend=gs.cuda if use_cuda else gs.cpu, logging_level='warning')

    gym_env = create_env(flags)
    env = _env_mod.Environment(gym_env)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    model = Net(gym_env.observation_space.shape, gym_env.action_space.n)
    model.eval()
    checkpoint = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    observation = env.initial()
    observation = {k: v.to(device) for k, v in observation.items()}
    agent_state = tuple(s.to(device) for s in model.initial_state(batch_size=1))
    returns = []

    while len(returns) < num_episodes:
        agent_outputs, agent_state = model(observation, agent_state)
        observation = env.step(agent_outputs["action"].cpu())
        observation = {k: v.to(device) for k, v in observation.items()}
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
            # Reset LSTM state on episode end.
            agent_state = tuple(s.to(device) for s in model.initial_state(batch_size=1))

    env.close()
    logging.info(
        "Average returns over %i episodes: %.1f",
        num_episodes, sum(returns) / len(returns),
    )


def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
