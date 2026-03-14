#!/usr/bin/env python3
"""Train DreamerV2 agent on Memory Maze.

Implements DreamerV2 (world-model based) training on the Memory Maze benchmark,
matching the paper's Table E.1 hyperparameters. Uses the same IMPALA ResNet
encoder as train_impala.py for fair comparison.

Setup:
    conda create -n dreamer python=3.11 -y
    conda activate dreamer
    pip install torch numpy "gym>=0.21,<1.0"
    cd memory-maze && pip install -e . && cd ..

Usage:
    # Train on 9x9 maze (default)
    python train_dreamer.py --num_envs 8 --total_steps 100_000_000

    # Smoke test
    python train_dreamer.py --num_envs 2 --total_steps 2000 --batch_size 2 --sequence_length 8

    # Train with TBTT (truncated backprop through time)
    python train_dreamer.py --tbtt

    # Train with Genesis backend
    python train_dreamer.py --backend genesis

    # Evaluate a trained agent
    python train_dreamer.py --mode test --xpid <experiment_id>

Based on: DreamerV2 (Hafner et al., 2020) applied to Memory Maze (Pasukonis et al., 2022)
"""

import argparse
import collections
import copy

import logging
import os
import pprint
import signal
import sys
import time
import timeit
from typing import Dict, List, NamedTuple, Optional, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("GENESIS_SKIP_TK_INIT", "1")  # prevent Tk crash in subprocesses


# ─── Import TorchBeast core modules ─────────────────────────────────
# TorchBeast pure-Python core modules are vendored in torchbeast/.

from torchbeast.core import file_writer as _fw_mod
from torchbeast.core import prof as _prof_mod

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Bernoulli


# ─── CLI ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="DreamerV2 on Memory Maze")

# Shared with train_impala.py.
parser.add_argument("--env", type=str, default="memory_maze:MemoryMaze-9x9-v0",
                    help="Gym environment.")
parser.add_argument("--mode", default="train", choices=["train", "test"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: auto-generated).")
parser.add_argument("--backend", default="mujoco", choices=["mujoco", "genesis"],
                    help="Physics backend: 'mujoco' (default) or 'genesis'.")
parser.add_argument("--physics_preset", default="none",
                    choices=["none", "conservative", "moderate", "aggressive"],
                    help="MuJoCo physics optimization preset.")
parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")

# Dreamer-specific.
parser.add_argument("--num_envs", default=8, type=int,
                    help="Number of parallel environments (paper: 8).")
parser.add_argument("--total_steps", default=100_000_000, type=int,
                    help="Total environment steps to train for (paper: 100M).")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Number of sequences per training batch (paper: 32).")
parser.add_argument("--sequence_length", default=48, type=int,
                    help="Sequence length for world model training (paper: 48).")
parser.add_argument("--imagination_horizon", default=15, type=int,
                    help="Imagination rollout length for actor-critic (paper: 15).")
parser.add_argument("--replay_capacity", default=10_000_000, type=int,
                    help="Replay buffer capacity in env steps (paper: 10M).")
parser.add_argument("--env_steps_per_update", default=25, type=int,
                    help="Env steps collected per training update (paper: 25).")
parser.add_argument("--prefill_steps", default=5000, type=int,
                    help="Random exploration steps before training starts.")
parser.add_argument("--tbtt", action="store_true",
                    help="Enable truncated backprop through time (carry RSSM state).")
parser.add_argument("--batched", action="store_true",
                    help="Use batched Genesis backend (requires --backend genesis).")
parser.add_argument("--physics_timestep", default=None, type=float,
                    help="Override Genesis physics timestep (default: 0.005, use 0.05 for 10x fewer substeps).")

# Optimizer settings (Table E.1).
parser.add_argument("--wm_lr", default=3e-4, type=float,
                    help="World model learning rate (paper: 3e-4).")
parser.add_argument("--actor_lr", default=1e-4, type=float,
                    help="Actor learning rate (paper: 1e-4).")
parser.add_argument("--critic_lr", default=1e-4, type=float,
                    help="Critic learning rate (paper: 1e-4).")
parser.add_argument("--adam_eps", default=1e-5, type=float,
                    help="AdamW epsilon (paper: 1e-5).")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="AdamW weight decay (paper: 1e-2).")
parser.add_argument("--grad_clip", default=200.0, type=float,
                    help="Global gradient norm clip (paper: 200).")

# Loss settings (Table E.1).
parser.add_argument("--kl_scale", default=1.0, type=float,
                    help="KL loss scale (paper: 1.0).")
parser.add_argument("--kl_balance", default=0.8, type=float,
                    help="KL balancing coefficient (paper: 0.8).")
parser.add_argument("--entropy_scale", default=0.001, type=float,
                    help="Actor entropy regularization (paper: 0.001).")
parser.add_argument("--discount", default=0.995, type=float,
                    help="Discount factor gamma (paper: 0.995).")
parser.add_argument("--lambda_", default=0.95, type=float,
                    help="Lambda for lambda-returns (paper: 0.95).")
parser.add_argument("--slow_critic_interval", default=100, type=int,
                    help="Slow critic EMA update interval (paper: 100).")

# Reproducibility.
parser.add_argument("--seed", type=int, default=None,
                    help="Global seed for reproducible runs (derives maze_seed=N, training_seed=N+1000000).")

# Weights & Biases.
parser.add_argument("--wandb", action="store_true",
                    help="Enable Weights & Biases logging.")
parser.add_argument("--wandb_project", default="memorymaze",
                    help="W&B project name.")
parser.add_argument("--wandb_entity", default=None,
                    help="W&B entity (team or username).")


PHYSICS_PRESETS = {
    "none": {},
    "conservative": {"timestep": 0.025, "noslip_iterations": 0, "cone": "pyramidal"},
    "moderate": {"timestep": 0.05, "noslip_iterations": 0, "cone": "pyramidal", "iterations": 3},
    "aggressive": {"timestep": 0.25, "noslip_iterations": 0, "cone": "pyramidal", "iterations": 1},
}


logging.basicConfig(
    format="[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] %(message)s",
    level=0,
)


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


def create_env(flags):
    """Create Memory Maze gym env with CHW observation layout."""
    kwargs = {"disable_env_checker": True}
    env_id = flags.env
    if getattr(flags, "backend", "mujoco") == "genesis" and "-Genesis-" not in env_id:
        env_id = env_id.replace("-v0", "-Genesis-v0")
        if getattr(flags, "physics_timestep", None) is not None:
            kwargs["physics_timestep"] = flags.physics_timestep
    else:
        physics_opts = PHYSICS_PRESETS.get(getattr(flags, "physics_preset", "none"), {})
        if physics_opts:
            kwargs["physics_opts"] = physics_opts
    gym_env = gym.make(env_id, **kwargs)
    return MemoryMazeWrapper(gym_env)


def get_env_metadata(flags):
    """Determine obs_shape and num_actions without creating a rendering env."""
    env_id = flags.env
    if ":" in env_id:
        env_id = env_id.split(":")[1]

    if "MemoryMaze" in env_id:
        num_actions = 6
        if "-HD-" in env_id or "-Top-" in env_id:
            obs_shape = (3, 256, 256)
        else:
            obs_shape = (3, 64, 64)
        return obs_shape, num_actions

    import multiprocessing
    def _probe(q, env_name):
        os.environ.setdefault("MUJOCO_GL", "glfw")
        _env = create_env(type("F", (), {"env": env_name})())
        q.put((_env.observation_space.shape, _env.action_space.n))
        _env.close()

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_probe, args=(q, flags.env))
    p.start()
    result = q.get(timeout=60)
    p.join()
    return result


# ─── Preprocessing ───────────────────────────────────────────────────

def preprocess(obs):
    """Convert uint8 CHW observation to float tensor in [-0.5, 0.5]."""
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)
    return obs.float() / 255.0 - 0.5


# ─── Utility MLP ─────────────────────────────────────────────────────

class MLP(nn.Module):
    """Multi-layer perceptron with ELU activation and optional LayerNorm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        activation: type = nn.ELU,
        layer_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ─── Encoder (IMPALA ResNet) ─────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block: ReLU -> Conv3x3 -> ReLU -> Conv3x3 + skip."""

    def __init__(self, depth: int):
        super().__init__()
        self.conv0 = nn.Conv2d(depth, depth, 3, padding=1)
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(x)
        out = self.conv0(out)
        out = F.relu(out)
        out = self.conv1(out)
        return out + x


class ConvBlock(nn.Module):
    """IMPALA conv block: Conv3x3 -> MaxPool -> 2x ResBlock."""

    def __init__(self, in_channels: int, depth: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, depth, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res0 = ResBlock(depth)
        self.res1 = ResBlock(depth)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res0(x)
        x = self.res1(x)
        return x


class ConvEncoder(nn.Module):
    """IMPALA ResNet encoder: (3,64,64) -> ConvBlocks(16,32,32) -> FC(1024)."""

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 1024,
        channels: Tuple[int, ...] = (16, 32, 32),
    ):
        super().__init__()
        self.feature_dim = feature_dim
        blocks: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            blocks.append(ConvBlock(c_in, c_out))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.cnn_out_size = channels[-1] * 8 * 8
        self.fc = nn.Linear(self.cnn_out_size, feature_dim)

    def forward(self, obs: Tensor) -> Tensor:
        leading_shape = obs.shape[:-3]
        x = obs.reshape(-1, *obs.shape[-3:])
        x = x.float() / 255.0
        for block in self.blocks:
            x = block(x)
        x = F.relu(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x.reshape(*leading_shape, self.feature_dim)


# ─── Decoder (Transposed Convolutions) ──────────────────────────────

class ConvDecoder(nn.Module):
    """Transposed conv decoder: state_dim -> FC -> reshape -> deconv -> (3,64,64)."""

    def __init__(
        self,
        state_dim: int,
        out_channels: int = 3,
        channels: Tuple[int, ...] = (32, 32, 16),
    ):
        super().__init__()
        self.initial_channels = channels[0]
        self.fc = nn.Linear(state_dim, channels[0] * 8 * 8)
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.append(nn.ConvTranspose2d(channels[i], channels[i + 1], 4, stride=2, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(channels[-1], out_channels, 4, stride=2, padding=1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, state_features: Tensor) -> Tensor:
        leading_shape = state_features.shape[:-1]
        x = state_features.reshape(-1, state_features.shape[-1])
        x = self.fc(x)
        x = x.reshape(-1, self.initial_channels, 8, 8)
        x = self.deconv(x)
        return x.reshape(*leading_shape, *x.shape[-3:])


# ─── RSSM State ──────────────────────────────────────────────────────

class RSSMState(NamedTuple):
    h: Tensor  # [B, gru_hidden]
    z: Tensor  # [B, num_categories, num_classes]


class RSSMDist(NamedTuple):
    logits: Tensor
    state: RSSMState


# ─── RSSM Core ───────────────────────────────────────────────────────

class RSSMCore(nn.Module):
    """Recurrent State Space Model with 32x32 categorical latent variables."""

    def __init__(
        self,
        gru_hidden: int = 2048,
        num_categories: int = 32,
        num_classes: int = 32,
        mlp_hidden: int = 1000,
        action_dim: int = 6,
        feature_dim: int = 1024,
    ):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.z_flat_dim = num_categories * num_classes

        self.input_proj = nn.Sequential(
            nn.Linear(self.z_flat_dim + action_dim, mlp_hidden),
            nn.ELU(),
        )
        self.gru = nn.GRUCell(mlp_hidden, gru_hidden)
        self.prior_mlp = nn.Sequential(
            nn.Linear(gru_hidden, mlp_hidden), nn.ELU(),
            nn.Linear(mlp_hidden, self.z_flat_dim),
        )
        self.posterior_mlp = nn.Sequential(
            nn.Linear(gru_hidden + feature_dim, mlp_hidden), nn.ELU(),
            nn.Linear(mlp_hidden, self.z_flat_dim),
        )

    def initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> RSSMState:
        if device is None:
            device = next(self.parameters()).device
        return RSSMState(
            h=torch.zeros(batch_size, self.gru_hidden, device=device),
            z=torch.zeros(batch_size, self.num_categories, self.num_classes, device=device),
        )

    def _categorical_straight_through(self, logits: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=-1)
        indices = torch.multinomial(
            probs.reshape(-1, self.num_classes), num_samples=1
        ).reshape(-1, self.num_categories, 1)
        one_hot = torch.zeros_like(logits).scatter_(-1, indices, 1.0)
        return one_hot + probs - probs.detach()

    def _reshape_logits(self, flat_logits: Tensor) -> Tensor:
        return flat_logits.reshape(*flat_logits.shape[:-1], self.num_categories, self.num_classes)

    def prior(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self._reshape_logits(self.prior_mlp(h))
        return logits, self._categorical_straight_through(logits)

    def posterior(self, h: Tensor, features: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self._reshape_logits(self.posterior_mlp(torch.cat([h, features], dim=-1)))
        return logits, self._categorical_straight_through(logits)

    def forward_step(
        self, prev_state: RSSMState, action: Tensor, encoded_obs: Optional[Tensor] = None,
    ) -> Tuple[RSSMDist, Optional[RSSMDist]]:
        B = prev_state.h.shape[0]
        action_onehot = F.one_hot(
            action.long(), num_classes=self.input_proj[0].in_features - self.z_flat_dim
        ).float()
        z_flat = prev_state.z.reshape(B, -1)
        gru_input = self.input_proj(torch.cat([z_flat, action_onehot], dim=-1))
        h = self.gru(gru_input, prev_state.h)

        prior_logits, prior_z = self.prior(h)
        prior_dist = RSSMDist(logits=prior_logits, state=RSSMState(h=h, z=prior_z))

        posterior_dist = None
        if encoded_obs is not None:
            post_logits, post_z = self.posterior(h, encoded_obs)
            posterior_dist = RSSMDist(logits=post_logits, state=RSSMState(h=h, z=post_z))

        return prior_dist, posterior_dist

    def observe(
        self, features: Tensor, actions: Tensor, init_state: RSSMState,
    ) -> Tuple[Tensor, Tensor, list]:
        T, B = features.shape[:2]
        state = init_state
        prior_logits_list, posterior_logits_list, states = [], [], []

        for t in range(T):
            action_t = torch.zeros(B, dtype=torch.long, device=features.device) if t == 0 else actions[t - 1]
            prior_dist, posterior_dist = self.forward_step(state, action_t, features[t])
            state = posterior_dist.state
            prior_logits_list.append(prior_dist.logits)
            posterior_logits_list.append(posterior_dist.logits)
            states.append(state)

        return torch.stack(prior_logits_list), torch.stack(posterior_logits_list), states

    def imagine(self, actor: nn.Module, init_state: RSSMState, horizon: int):
        state = init_state
        states = [state]
        actions_list = []
        for _ in range(horizon):
            state_feat = get_state_features(state)
            action_logits = actor(state_feat)
            action = Categorical(logits=action_logits).sample()
            actions_list.append(action)
            prior_dist, _ = self.forward_step(state, action, encoded_obs=None)
            state = prior_dist.state
            states.append(state)
        return states, torch.stack(actions_list)


# ─── Helper ──────────────────────────────────────────────────────────

def get_state_features(state: RSSMState) -> Tensor:
    z_flat = state.z.reshape(state.z.shape[0], -1)
    return torch.cat([state.h, z_flat], dim=-1)


# ─── Reward, Discount, Actor, Critic ────────────────────────────────

class RewardDecoder(nn.Module):
    def __init__(self, state_dim: int = 3072, hidden_dim: int = 400, num_layers: int = 4):
        super().__init__()
        self.mlp = MLP(state_dim, hidden_dim, output_dim=1, num_layers=num_layers)

    def forward(self, state_features: Tensor) -> Tensor:
        return self.mlp(state_features).squeeze(-1)


class DiscountDecoder(nn.Module):
    def __init__(self, state_dim: int = 3072, hidden_dim: int = 400, num_layers: int = 4):
        super().__init__()
        self.mlp = MLP(state_dim, hidden_dim, output_dim=1, num_layers=num_layers)

    def forward(self, state_features: Tensor) -> Tensor:
        return self.mlp(state_features).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, state_dim: int = 3072, num_actions: int = 6, hidden_dim: int = 400, num_layers: int = 4):
        super().__init__()
        self.mlp = MLP(state_dim, hidden_dim, output_dim=num_actions, num_layers=num_layers)

    def forward(self, state_features: Tensor) -> Tensor:
        return self.mlp(state_features)


class Critic(nn.Module):
    def __init__(self, state_dim: int = 3072, hidden_dim: int = 400, num_layers: int = 4):
        super().__init__()
        self.mlp = MLP(state_dim, hidden_dim, output_dim=1, num_layers=num_layers)

    def forward(self, state_features: Tensor) -> Tensor:
        return self.mlp(state_features).squeeze(-1)


# ─── KL Balancing Loss ───────────────────────────────────────────────

def kl_balancing_loss(prior_logits, posterior_logits, alpha=0.8, free_nats=0.0):
    """KL with balancing (DreamerV2 Eq. 5)."""
    def _kl(q_logits, p_logits):
        q = F.softmax(q_logits, dim=-1)
        kl_per_cat = (q * (F.log_softmax(q_logits, dim=-1) - F.log_softmax(p_logits, dim=-1))).sum(-1)
        return torch.clamp(kl_per_cat, min=free_nats).sum(-1)

    kl_prior = _kl(posterior_logits.detach(), prior_logits)
    kl_posterior = _kl(posterior_logits, prior_logits.detach())
    return (alpha * kl_prior + (1.0 - alpha) * kl_posterior).mean()


# ─── Lambda Returns ──────────────────────────────────────────────────

def lambda_returns(rewards, values, discounts, bootstrap, lambda_=0.95):
    T = rewards.shape[0]
    next_values = torch.cat([values[1:], bootstrap.unsqueeze(0)], dim=0)
    targets = []
    last = bootstrap
    for t in reversed(range(T)):
        last = rewards[t] + discounts[t] * ((1.0 - lambda_) * next_values[t] + lambda_ * last)
        targets.append(last)
    targets.reverse()
    return torch.stack(targets)


# ─── DreamerModel Container ─────────────────────────────────────────

class DreamerModel(nn.Module):
    """Complete DreamerV2 model wrapping all components."""

    def __init__(
        self,
        num_actions: int = 6,
        gru_hidden: int = 2048,
        num_categories: int = 32,
        num_classes: int = 32,
        rssm_mlp_hidden: int = 1000,
        feature_dim: int = 1024,
        decoder_hidden: int = 400,
        decoder_layers: int = 4,
    ):
        super().__init__()
        self.num_actions = num_actions
        state_dim = gru_hidden + num_categories * num_classes

        self.encoder = ConvEncoder(in_channels=3, feature_dim=feature_dim)
        self.rssm = RSSMCore(
            gru_hidden=gru_hidden, num_categories=num_categories,
            num_classes=num_classes, mlp_hidden=rssm_mlp_hidden,
            action_dim=num_actions, feature_dim=feature_dim,
        )
        self.image_decoder = ConvDecoder(state_dim=state_dim)
        self.reward_decoder = RewardDecoder(state_dim=state_dim, hidden_dim=decoder_hidden, num_layers=decoder_layers)
        self.discount_decoder = DiscountDecoder(state_dim=state_dim, hidden_dim=decoder_hidden, num_layers=decoder_layers)
        self.actor = Actor(state_dim=state_dim, num_actions=num_actions, hidden_dim=decoder_hidden, num_layers=decoder_layers)
        self.critic = Critic(state_dim=state_dim, hidden_dim=decoder_hidden, num_layers=decoder_layers)

    def initial_state(self, batch_size, device=None):
        return self.rssm.initial_state(batch_size, device)

    def encode(self, obs):
        return self.encoder(obs)

    def rssm_observe(self, features, actions, init_state):
        return self.rssm.observe(features, actions, init_state)

    def rssm_imagine(self, init_state, horizon):
        return self.rssm.imagine(self.actor, init_state, horizon)

    def decode(self, state):
        feat = get_state_features(state)
        return {
            "image": self.image_decoder(feat),
            "reward": self.reward_decoder(feat),
            "discount_logit": self.discount_decoder(feat),
        }

    def world_model_parameters(self):
        yield from self.encoder.parameters()
        yield from self.rssm.parameters()
        yield from self.image_decoder.parameters()
        yield from self.reward_decoder.parameters()
        yield from self.discount_decoder.parameters()

    def actor_parameters(self):
        yield from self.actor.parameters()

    def critic_parameters(self):
        yield from self.critic.parameters()


# ─── Slow Critic (EMA Target) ───────────────────────────────────────

class SlowCritic:
    """Exponential moving average of critic weights for stable targets."""

    def __init__(self, critic, update_interval=100, decay=0.98):
        self.critic = critic
        self.target = copy.deepcopy(critic)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.update_interval = update_interval
        self.decay = decay
        self._step = 0

    def maybe_update(self, step=None):
        if step is not None:
            self._step = step
        else:
            self._step += 1
        if self._step % self.update_interval == 0:
            for t_param, o_param in zip(self.target.parameters(), self.critic.parameters()):
                t_param.data.mul_(self.decay).add_(o_param.data, alpha=1.0 - self.decay)

    def __call__(self, state_features):
        with torch.no_grad():
            return self.target(state_features)

    def state_dict(self):
        return self.target.state_dict()

    def load_state_dict(self, sd):
        self.target.load_state_dict(sd)


# ─── Replay Buffer ──────────────────────────────────────────────────

class ReplayBuffer:
    """Episode-based circular replay buffer for DreamerV2."""

    def __init__(self, capacity=10_000_000):
        self.capacity = capacity
        self._episodes = collections.deque()
        self._episode_lengths = collections.deque()
        self._total_steps = 0
        self._total_episodes = 0

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def total_episodes(self):
        return self._total_episodes

    @property
    def num_episodes(self):
        return len(self._episodes)

    def add_episode(self, episode):
        length = len(episode["reward"])
        if length == 0:
            return
        stored = {
            "obs": np.asarray(episode["obs"], dtype=np.uint8),
            "action": np.asarray(episode["action"], dtype=np.int64),
            "reward": np.asarray(episode["reward"], dtype=np.float32),
            "done": np.asarray(episode["done"], dtype=np.bool_),
            "discount": np.asarray(episode["discount"], dtype=np.float32),
        }
        self._episodes.append(stored)
        self._episode_lengths.append(length)
        self._total_steps += length
        self._total_episodes += 1
        while self._total_steps > self.capacity and len(self._episodes) > 1:
            evicted_len = self._episode_lengths.popleft()
            self._episodes.popleft()
            self._total_steps -= evicted_len

    def can_sample(self, sequence_length):
        if len(self._episodes) == 0:
            return False
        return max(self._episode_lengths) >= sequence_length

    def sample(self, batch_size, sequence_length):
        if not self.can_sample(sequence_length):
            raise ValueError(
                f"Cannot sample sequences of length {sequence_length}. "
                f"Buffer has {len(self._episodes)} episodes, "
                f"longest is {max(self._episode_lengths) if self._episodes else 0} steps."
            )
        lengths = np.array(self._episode_lengths)
        effective = np.maximum(lengths - sequence_length + 1, 0).astype(np.float64)
        weights = effective / effective.sum()
        episode_idxs = np.random.choice(len(self._episodes), size=batch_size, p=weights)

        batch_obs, batch_action, batch_reward, batch_done, batch_discount = [], [], [], [], []
        for ep_idx in episode_idxs:
            ep = self._episodes[ep_idx]
            ep_len = self._episode_lengths[ep_idx]
            start = np.random.randint(0, ep_len - sequence_length + 1)
            batch_obs.append(ep["obs"][start:start + sequence_length])
            batch_action.append(ep["action"][start:start + sequence_length])
            batch_reward.append(ep["reward"][start:start + sequence_length])
            batch_done.append(ep["done"][start:start + sequence_length])
            batch_discount.append(ep["discount"][start:start + sequence_length])

        return {
            "obs": torch.from_numpy(np.stack(batch_obs)),  # raw uint8; encoder/preprocess handle normalization
            "action": torch.from_numpy(np.stack(batch_action)),
            "reward": torch.from_numpy(np.stack(batch_reward)),
            "done": torch.from_numpy(np.stack(batch_done).astype(np.float32)),
            "discount": torch.from_numpy(np.stack(batch_discount)),
        }


# ─── Environment Runner ─────────────────────────────────────────────

class EnvRunner:
    """Manages N parallel environments for experience collection."""

    def __init__(self, create_env_fn, replay_buffer, num_envs=8, num_actions=6):
        self.replay_buffer = replay_buffer
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.envs = [create_env_fn() for _ in range(num_envs)]
        self._obs_bufs = [[] for _ in range(num_envs)]
        self._action_bufs = [[] for _ in range(num_envs)]
        self._reward_bufs = [[] for _ in range(num_envs)]
        self._done_bufs = [[] for _ in range(num_envs)]
        self._discount_bufs = [[] for _ in range(num_envs)]
        self._current_obs = [env.reset() for env in self.envs]
        self._policy_states = [None] * num_envs

    def collect(self, policy_fn, num_steps):
        steps_taken = 0
        episodes_completed = 0
        episode_returns = []
        episode_lengths = []

        with torch.no_grad():
            while steps_taken < num_steps:
                for i in range(self.num_envs):
                    if steps_taken >= num_steps:
                        break
                    obs_tensor = torch.from_numpy(self._current_obs[i]).unsqueeze(0)  # (1, C, H, W) uint8
                    action_tensor, new_state = policy_fn(obs_tensor, self._policy_states[i])
                    self._policy_states[i] = new_state
                    action = action_tensor.item() if isinstance(action_tensor, torch.Tensor) else int(action_tensor)

                    self._obs_bufs[i].append(self._current_obs[i])
                    self._action_bufs[i].append(action)

                    obs, reward, done, info = self.envs[i].step(action)
                    steps_taken += 1
                    self._reward_bufs[i].append(float(reward))
                    self._done_bufs[i].append(bool(done))
                    self._discount_bufs[i].append(0.0 if done else 1.0)

                    if done:
                        episode = {
                            "obs": np.stack(self._obs_bufs[i]),
                            "action": np.array(self._action_bufs[i]),
                            "reward": np.array(self._reward_bufs[i]),
                            "done": np.array(self._done_bufs[i]),
                            "discount": np.array(self._discount_bufs[i]),
                        }
                        episode_returns.append(sum(self._reward_bufs[i]))
                        episode_lengths.append(len(self._reward_bufs[i]))
                        self.replay_buffer.add_episode(episode)
                        episodes_completed += 1
                        self._obs_bufs[i] = []
                        self._action_bufs[i] = []
                        self._reward_bufs[i] = []
                        self._done_bufs[i] = []
                        self._discount_bufs[i] = []
                        obs = self.envs[i].reset()
                        self._policy_states[i] = None

                    self._current_obs[i] = obs

        mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
        mean_length = float(np.mean(episode_lengths)) if episode_lengths else float("nan")
        return {"steps": steps_taken, "episodes": episodes_completed, "mean_return": mean_return, "mean_episode_length": mean_length}

    def close(self):
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass


def _parse_maze_size(env_id):
    """Extract maze size from env ID string, e.g. 'memory_maze:MemoryMaze-9x9-v0' → 9."""
    import re
    m = re.search(r'(\d+)x\d+', env_id)
    if m:
        return int(m.group(1))
    return 9  # default


class BatchedEnvRunner:
    """Manages N environments via a single BatchGenesisMemoryMazeEnv.

    Drop-in replacement for EnvRunner — same collect() interface.
    """

    def __init__(self, flags, replay_buffer, num_envs=8, num_actions=6):
        from memory_maze.genesis_backend import BatchGenesisMemoryMazeEnv
        maze_size = _parse_maze_size(flags.env)
        seed = getattr(flags, "seed", None)
        self.batch_env = BatchGenesisMemoryMazeEnv(
            n_envs=num_envs, maze_size=maze_size,
            camera_resolution=64, seed=seed,
        )
        self.replay_buffer = replay_buffer
        self.num_envs = num_envs
        self.num_actions = num_actions

        # Per-env episode buffers (same as EnvRunner)
        self._obs_bufs = [[] for _ in range(num_envs)]
        self._action_bufs = [[] for _ in range(num_envs)]
        self._reward_bufs = [[] for _ in range(num_envs)]
        self._done_bufs = [[] for _ in range(num_envs)]
        self._discount_bufs = [[] for _ in range(num_envs)]
        self._policy_states = [None] * num_envs

        # Initial reset — returns (N, H, W, 3) HWC observations
        obs_all = self.batch_env.reset()  # (N, H, W, 3)
        # Transpose each to CHW
        self._current_obs = [
            np.ascontiguousarray(np.transpose(obs_all[i], (2, 0, 1)))
            for i in range(num_envs)
        ]

    def collect(self, policy_fn, num_steps):
        """Collect experience from all envs using batched stepping.

        Args:
            policy_fn: callable(obs_tensor, state) → (action, new_state)
                       where obs_tensor is (1, C, H, W).
            num_steps: total env steps to collect across all envs.

        Returns:
            dict with keys: steps, episodes, mean_return, mean_episode_length.
        """
        steps_taken = 0
        episodes_completed = 0
        episode_returns = []
        episode_lengths = []

        with torch.no_grad():
            while steps_taken < num_steps:
                # Gather actions for all envs
                actions = np.empty(self.num_envs, dtype=np.int64)
                for i in range(self.num_envs):
                    obs_tensor = torch.from_numpy(self._current_obs[i]).unsqueeze(0)  # (1, C, H, W) uint8
                    action_tensor, new_state = policy_fn(obs_tensor, self._policy_states[i])
                    self._policy_states[i] = new_state
                    action = action_tensor.item() if isinstance(action_tensor, torch.Tensor) else int(action_tensor)
                    actions[i] = action
                    self._obs_bufs[i].append(self._current_obs[i])
                    self._action_bufs[i].append(action)

                # Single batched step
                obs_all, rewards, dones, infos = self.batch_env.step(actions)
                steps_taken += self.num_envs

                for i in range(self.num_envs):
                    self._reward_bufs[i].append(float(rewards[i]))
                    self._done_bufs[i].append(bool(dones[i]))
                    self._discount_bufs[i].append(0.0 if dones[i] else 1.0)

                    if dones[i]:
                        episode = {
                            "obs": np.stack(self._obs_bufs[i]),
                            "action": np.array(self._action_bufs[i]),
                            "reward": np.array(self._reward_bufs[i]),
                            "done": np.array(self._done_bufs[i]),
                            "discount": np.array(self._discount_bufs[i]),
                        }
                        episode_returns.append(sum(self._reward_bufs[i]))
                        episode_lengths.append(len(self._reward_bufs[i]))
                        self.replay_buffer.add_episode(episode)
                        episodes_completed += 1
                        self._obs_bufs[i] = []
                        self._action_bufs[i] = []
                        self._reward_bufs[i] = []
                        self._done_bufs[i] = []
                        self._discount_bufs[i] = []
                        # Auto-reset handled by BatchGenesisMemoryMazeEnv
                        self._policy_states[i] = None

                    # Transpose HWC → CHW
                    self._current_obs[i] = np.ascontiguousarray(
                        np.transpose(obs_all[i], (2, 0, 1))
                    )

        mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
        mean_length = float(np.mean(episode_lengths)) if episode_lengths else float("nan")
        return {"steps": steps_taken, "episodes": episodes_completed, "mean_return": mean_return, "mean_episode_length": mean_length}

    def close(self):
        try:
            self.batch_env.close()
        except Exception:
            pass


# ─── World Model Training ───────────────────────────────────────────

def train_world_model(model, batch, optimizer, init_state=None,
                      kl_scale=1.0, kl_alpha=0.8, grad_clip=200.0, device=None):
    """Single world model training step."""
    if device is None:
        device = next(model.parameters()).device

    obs = batch["obs"].to(device)
    actions = batch["action"].to(device)
    rewards = batch["reward"].to(device)
    discounts = batch["discount"].to(device)
    B, L = actions.shape

    if init_state is None:
        state = model.initial_state(B, device)
    else:
        state = RSSMState(h=init_state.h.to(device), z=init_state.z.to(device))

    obs_time_major = obs.permute(1, 0, 2, 3, 4)
    features = model.encode(obs_time_major)
    actions_time_major = actions.permute(1, 0)

    prior_logits, posterior_logits, posterior_states = model.rssm_observe(
        features, actions_time_major, state,
    )

    all_h = torch.stack([s.h for s in posterior_states], dim=0)
    all_z = torch.stack([s.z for s in posterior_states], dim=0)
    all_z_flat = all_z.reshape(L, B, -1)
    state_features = torch.cat([all_h, all_z_flat], dim=-1)

    recon_obs = model.image_decoder(state_features)
    pred_reward = model.reward_decoder(state_features)
    pred_discount_logit = model.discount_decoder(state_features)

    target_obs = preprocess(obs_time_major)
    image_loss = F.mse_loss(recon_obs, target_obs)
    reward_loss = F.mse_loss(pred_reward, rewards.permute(1, 0))
    discount_loss = F.binary_cross_entropy_with_logits(
        pred_discount_logit, discounts.permute(1, 0),
    )
    kl_loss = kl_balancing_loss(prior_logits, posterior_logits, alpha=kl_alpha)
    total_loss = image_loss + reward_loss + discount_loss + kl_scale * kl_loss

    optimizer.zero_grad()
    total_loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(list(model.world_model_parameters()), grad_clip)
    optimizer.step()

    # GRU hidden state norm — tracks recurrent state magnitude over training.
    gru_hidden_norm = torch.mean(torch.norm(all_h, dim=-1)).item()
    stats = {
        "wm_loss": total_loss.item(),
        "image_loss": image_loss.item(),
        "reward_loss": reward_loss.item(),
        "discount_loss": discount_loss.item(),
        "kl_loss": kl_loss.item(),
        "wm_grad_norm": grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm,
        "gru_hidden_norm": gru_hidden_norm,
    }
    final_state = RSSMState(h=posterior_states[-1].h.detach(), z=posterior_states[-1].z.detach())
    return stats, posterior_states, final_state


# ─── Actor-Critic Training ──────────────────────────────────────────

def train_actor_critic(model, posterior_states, actor_optimizer, critic_optimizer,
                       slow_critic, horizon=15, gamma=0.995, lambda_=0.95,
                       entropy_scale=0.001, grad_clip=200.0, train_step=0):
    """Train actor and critic in imagination."""
    all_h = torch.stack([s.h for s in posterior_states], dim=0).detach()
    all_z = torch.stack([s.z for s in posterior_states], dim=0).detach()
    L, B = all_h.shape[:2]
    flat_h = all_h.reshape(L * B, -1)
    flat_z = all_z.reshape(L * B, *all_z.shape[2:])
    init_state = RSSMState(h=flat_h, z=flat_z)

    # --- Critic training ---
    imagined_states, imagined_actions = model.rssm_imagine(init_state, horizon)
    state_feats = torch.stack([get_state_features(s) for s in imagined_states])

    with torch.no_grad():
        imagined_rewards = model.reward_decoder(state_feats[1:])
        imagined_discounts = gamma * torch.sigmoid(model.discount_decoder(state_feats[1:]))

    values = model.critic(state_feats[:-1].detach())
    bootstrap_value = model.critic(state_feats[-1].detach())

    with torch.no_grad():
        slow_values = slow_critic(state_feats[:-1])
        slow_bootstrap = slow_critic(state_feats[-1])
        target_returns = lambda_returns(imagined_rewards, slow_values, imagined_discounts, slow_bootstrap, lambda_)

    critic_loss = F.mse_loss(values, target_returns)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(list(model.critic_parameters()), grad_clip)
    critic_optimizer.step()

    # --- Actor training (REINFORCE for discrete actions, per DreamerV2) ---
    imagined_states2, imagined_actions2 = model.rssm_imagine(init_state, horizon)
    state_feats2 = torch.stack([get_state_features(s) for s in imagined_states2])

    with torch.no_grad():
        actor_rewards = model.reward_decoder(state_feats2[1:])
        actor_discounts = gamma * torch.sigmoid(model.discount_decoder(state_feats2[1:]))
        actor_values = slow_critic(state_feats2[:-1])
        actor_bootstrap = slow_critic(state_feats2[-1])
        actor_returns = lambda_returns(actor_rewards, actor_values, actor_discounts, actor_bootstrap, lambda_)

    # Recompute actor log-probs for the imagined actions
    actor_logits = torch.stack([model.actor(get_state_features(imagined_states2[t])) for t in range(horizon)])
    dist = Categorical(logits=actor_logits)
    log_probs = dist.log_prob(imagined_actions2)  # [horizon, L*B]
    entropy = dist.entropy().mean()
    # REINFORCE: gradient flows through log_prob, returns are treated as scalar weights
    actor_loss = -(log_probs * actor_returns.detach()).mean() - entropy_scale * entropy

    actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(list(model.actor_parameters()), grad_clip)
    actor_optimizer.step()

    slow_critic.maybe_update(train_step)

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": entropy.item(),
        "imagined_return": actor_returns.mean().item(),
    }


# ─── Policy function for env collection ─────────────────────────────

def make_policy_fn(model, device):
    """Create a policy function for env collection using the actor.

    During collection, we run the encoder + RSSM posterior to track the
    latent state, then sample from the actor.
    """
    rssm_state = [None]  # mutable container for closure

    def policy_fn(obs_tensor, env_state):
        """
        Args:
            obs_tensor: (1, C, H, W) raw uint8 obs
            env_state: opaque per-env state (we use rssm_state from closure)

        Returns:
            action: scalar tensor
            new_state: opaque state
        """
        obs_tensor = obs_tensor.to(device)
        features = model.encode(obs_tensor)  # (1, feature_dim); encode handles /255.0

        if env_state is None:
            state = model.initial_state(1, device)
            action_prev = torch.zeros(1, dtype=torch.long, device=device)
        else:
            state, action_prev = env_state

        # RSSM step: get posterior given observation
        prior_dist, posterior_dist = model.rssm.forward_step(state, action_prev, features)
        state = posterior_dist.state

        # Actor
        state_feat = get_state_features(state)
        action_logits = model.actor(state_feat)
        if model.training:
            action = Categorical(logits=action_logits).sample()
        else:
            action = action_logits.argmax(-1)

        return action.squeeze(0), (state, action)

    return policy_fn


# ─── Train ───────────────────────────────────────────────────────────

def train(flags):
    if flags.xpid is None:
        flags.xpid = "dreamer-%s" % time.strftime("%Y%m%d-%H%M%S")
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
                "agent": "dreamer",
                "env": flags.env,
                "batched": getattr(flags, "batched", False),
                "num_envs": flags.num_envs,
                "batch_size": flags.batch_size,
                "sequence_length": flags.sequence_length,
                "total_steps": flags.total_steps,
                "wm_lr": flags.wm_lr,
                "actor_lr": flags.actor_lr,
                "critic_lr": flags.critic_lr,
                "entropy_scale": flags.entropy_scale,
                "imagination_horizon": flags.imagination_horizon,
                "tbtt": flags.tbtt,
            },
            tags=["dreamer", flags.backend],
        )

    # Reproducibility seeding.
    if flags.seed is not None:
        training_seed = flags.seed + 1_000_000
        torch.manual_seed(training_seed)
        np.random.seed(training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_seed)
        logging.info("Seed: maze_seed=%d, training_seed=%d", flags.seed, training_seed)

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    obs_shape, num_actions = get_env_metadata(flags)
    logging.info("Obs shape: %s, Num actions: %d", obs_shape, num_actions)

    # Create model.
    model = DreamerModel(num_actions=num_actions).to(flags.device)
    param_count = sum(p.numel() for p in model.parameters())
    logging.info("DreamerModel: %d parameters", param_count)

    # Optimizers.
    wm_optimizer = torch.optim.AdamW(
        model.world_model_parameters(), lr=flags.wm_lr,
        eps=flags.adam_eps, weight_decay=flags.weight_decay,
    )
    actor_optimizer = torch.optim.AdamW(
        model.actor_parameters(), lr=flags.actor_lr,
        eps=flags.adam_eps, weight_decay=flags.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        model.critic_parameters(), lr=flags.critic_lr,
        eps=flags.adam_eps, weight_decay=flags.weight_decay,
    )
    slow_critic = SlowCritic(model.critic, update_interval=flags.slow_critic_interval)

    # Replay buffer and env runner.
    replay_buffer = ReplayBuffer(capacity=flags.replay_capacity)
    if flags.batched:
        assert flags.backend == "genesis", "--batched requires --backend genesis"
        logging.info("Using BatchedEnvRunner with %d envs", flags.num_envs)
        env_runner = BatchedEnvRunner(
            flags, replay_buffer,
            num_envs=flags.num_envs, num_actions=num_actions,
        )
    else:
        env_runner = EnvRunner(
            create_env_fn=lambda: create_env(flags),
            replay_buffer=replay_buffer,
            num_envs=flags.num_envs,
            num_actions=num_actions,
        )

    # TBTT state tracking.
    tbtt_state = None  # Will hold RSSMState carried across batches.

    # Logging keys.
    stat_keys = [
        "wm_loss", "image_loss", "reward_loss", "discount_loss", "kl_loss",
        "actor_loss", "critic_loss", "entropy", "imagined_return",
        "mean_episode_return", "mean_episode_length",
        "gru_hidden_norm",
    ]
    logger = logging.getLogger("logfile")
    logger.info("# Step\t%s", "\t".join(stat_keys))

    # Policy for collection.
    def random_policy(obs_tensor, state):
        action = torch.randint(0, num_actions, (1,))
        return action.squeeze(0), None

    # Auto-resume from checkpoint.
    _resumed = False
    if os.path.exists(checkpointpath):
        logging.info("Resuming from checkpoint: %s", checkpointpath)
        ckpt = torch.load(checkpointpath, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        wm_optimizer.load_state_dict(ckpt["wm_optimizer_state_dict"])
        actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])
        slow_critic.load_state_dict(ckpt["slow_critic_state_dict"])
        env_steps = ckpt["env_steps"]
        train_step = ckpt["train_step"]
        _resumed = True
        logging.info("Resumed at env_step %d, train_step %d", env_steps, train_step)

    # Prefill replay buffer with random actions (skip if resuming — buffer will refill).
    if not _resumed:
        logging.info("Prefilling replay buffer with %d random steps...", flags.prefill_steps)
        env_runner.collect(random_policy, flags.prefill_steps)
    else:
        # Collect just enough random data to seed the buffer for sampling.
        min_steps = flags.sequence_length * flags.batch_size
        logging.info("Resumed from checkpoint. Collecting minimal seed data (%d steps)...", min_steps)
        env_runner.collect(random_policy, min_steps)
    logging.info("Replay buffer: %d steps, %d episodes",
                 replay_buffer.total_steps, replay_buffer.total_episodes)

    # Main training loop.
    if not _resumed:
        env_steps = 0
        train_step = 0
    timer = timeit.default_timer
    last_checkpoint_time = timer()
    last_log_time = timer()
    all_episode_returns = []

    model.train()

    # Handle SIGTERM for graceful shutdown on spot instance preemption.
    def _sigterm_handler(signum, frame):
        logging.info("SIGTERM received — saving emergency checkpoint...")
        _save_checkpoint(flags, checkpointpath, model, wm_optimizer,
                         actor_optimizer, critic_optimizer, slow_critic,
                         env_steps, train_step)
        env_runner.close()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        while env_steps < flags.total_steps:
            start_time = timer()
            start_env_steps = env_steps

            # 1. Collect experience.
            policy_fn = make_policy_fn(model, flags.device)
            collect_stats = env_runner.collect(policy_fn, flags.env_steps_per_update)
            env_steps += collect_stats["steps"]

            if collect_stats["episodes"] > 0:
                all_episode_returns.append(collect_stats["mean_return"])

            # 2. Train world model (if buffer has enough data).
            wm_stats = {}
            ac_stats = {}
            if replay_buffer.can_sample(flags.sequence_length):
                batch = replay_buffer.sample(flags.batch_size, flags.sequence_length)

                # TBTT: pass carried state from previous batch.
                init_state = tbtt_state if flags.tbtt else None

                wm_stats, posterior_states, final_state = train_world_model(
                    model, batch, wm_optimizer,
                    init_state=init_state,
                    kl_scale=flags.kl_scale,
                    kl_alpha=flags.kl_balance,
                    grad_clip=flags.grad_clip,
                    device=flags.device,
                )

                # TBTT: carry state for next batch (detached).
                if flags.tbtt:
                    tbtt_state = final_state

                # 3. Train actor-critic in imagination.
                ac_stats = train_actor_critic(
                    model, posterior_states,
                    actor_optimizer, critic_optimizer, slow_critic,
                    horizon=flags.imagination_horizon,
                    gamma=flags.discount,
                    lambda_=flags.lambda_,
                    entropy_scale=flags.entropy_scale,
                    grad_clip=flags.grad_clip,
                    train_step=train_step,
                )
                train_step += 1

            # 4. Logging.
            elapsed = timer() - start_time
            sps = collect_stats["steps"] / max(elapsed, 1e-6)

            mean_return = all_episode_returns[-1] if all_episode_returns else float("nan")
            mean_ep_length = collect_stats.get("mean_episode_length", float("nan"))
            log_data = {"step": env_steps, "mean_episode_return": mean_return, "mean_episode_length": mean_ep_length}
            log_data.update(wm_stats)
            log_data.update(ac_stats)
            plogger.log(log_data)
            if flags.wandb:
                wandb_data = {k: v for k, v in log_data.items()
                              if isinstance(v, (int, float)) and v == v}  # skip NaN
                wandb_data["sps"] = sps
                wandb.log(wandb_data, step=env_steps)

            # Periodic console output.
            if timer() - last_log_time > 5.0:
                last_log_time = timer()
                wm_loss = wm_stats.get("wm_loss", float("nan"))
                actor_loss = ac_stats.get("actor_loss", float("nan"))
                return_str = "%.1f" % mean_return if not np.isnan(mean_return) else "N/A"
                logging.info(
                    "Steps %d @ %.1f SPS | WM loss %.4f | Actor loss %.4f | "
                    "Return %s | Buffer %d steps, %d eps | Train step %d",
                    env_steps, sps, wm_loss, actor_loss,
                    return_str, replay_buffer.total_steps,
                    replay_buffer.num_episodes, train_step,
                )

            # 5. Checkpoint.
            if timer() - last_checkpoint_time > 10 * 60:
                _save_checkpoint(flags, checkpointpath, model, wm_optimizer,
                                 actor_optimizer, critic_optimizer, slow_critic,
                                 env_steps, train_step)
                last_checkpoint_time = timer()

    except KeyboardInterrupt:
        logging.info("Training interrupted at step %d.", env_steps)
    finally:
        env_runner.close()

    _save_checkpoint(flags, checkpointpath, model, wm_optimizer,
                     actor_optimizer, critic_optimizer, slow_critic,
                     env_steps, train_step)
    logging.info("Training finished after %d env steps, %d train steps.", env_steps, train_step)
    plogger.close()
    if flags.wandb:
        wandb.finish()


def _save_checkpoint(flags, path, model, wm_opt, actor_opt, critic_opt,
                     slow_critic, env_steps, train_step):
    if flags.disable_checkpoint:
        return
    logging.info("Saving checkpoint to %s", path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "wm_optimizer_state_dict": wm_opt.state_dict(),
        "actor_optimizer_state_dict": actor_opt.state_dict(),
        "critic_optimizer_state_dict": critic_opt.state_dict(),
        "slow_critic_state_dict": slow_critic.state_dict(),
        "env_steps": env_steps,
        "train_step": train_step,
        "flags": vars(flags),
    }, path)


# ─── Test ────────────────────────────────────────────────────────────

def test(flags, num_episodes: int = 10):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser(
                "%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar")
            )
        )

    logging.info("Loading checkpoint from %s", checkpointpath)
    checkpoint = torch.load(checkpointpath, map_location="cpu")

    saved_flags = checkpoint.get("flags", {})
    num_actions = 6  # Memory Maze default

    model = DreamerModel(num_actions=num_actions)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    gym_env = create_env(flags)
    returns = []
    state = None

    obs = gym_env.reset()
    episode_return = 0.0
    episode_step = 0

    while len(returns) < num_episodes:
        obs_uint8 = torch.from_numpy(obs).unsqueeze(0).to(device)
        features = model.encode(obs_uint8)

        if state is None:
            rssm_state = model.initial_state(1, device)
            action_prev = torch.zeros(1, dtype=torch.long, device=device)
        else:
            rssm_state, action_prev = state

        _, posterior_dist = model.rssm.forward_step(rssm_state, action_prev, features)
        rssm_state = posterior_dist.state

        state_feat = get_state_features(rssm_state)
        action_logits = model.actor(state_feat)
        action = action_logits.argmax(-1)

        state = (rssm_state, action)

        obs, reward, done, info = gym_env.step(action.item())
        episode_return += reward
        episode_step += 1

        if done:
            returns.append(episode_return)
            logging.info("Episode ended after %d steps. Return: %.1f",
                         episode_step, episode_return)
            obs = gym_env.reset()
            state = None
            episode_return = 0.0
            episode_step = 0

    gym_env.close()
    logging.info("Average return over %d episodes: %.1f",
                 num_episodes, sum(returns) / len(returns))


# ─── Main ────────────────────────────────────────────────────────────

def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
