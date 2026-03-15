"""Shared utilities for Memory Maze example notebooks.

Provides log loading, plotting, video recording, and engine comparison
helpers used across multiple notebooks.
"""

import os
import csv
import time
from pathlib import Path

import numpy as np

# Default log directory (matches train_impala.py / train_dreamer.py)
DEFAULT_LOGDIR = os.path.expanduser("~/logs/torchbeast")


# ─── Log Loading ─────────────────────────────────────────────────────

def load_training_logs(xpid, logdir=DEFAULT_LOGDIR):
    """Load logs.csv from a training run into a list of dicts.

    Args:
        xpid: Experiment ID (directory name under logdir).
        logdir: Root log directory. Defaults to ~/logs/torchbeast.

    Returns:
        List of dicts, one per logged step. Keys include 'step',
        'mean_episode_return', 'total_loss', etc.
    """
    logs_path = os.path.join(logdir, xpid, "logs.csv")
    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"No logs.csv found at {logs_path}")

    rows = []
    with open(logs_path, "r") as f:
        # The header line starts with "# " — strip the prefix but keep it.
        # Skip any other comment lines.
        lines = []
        for line in f:
            if line.startswith("# ") and not lines:
                lines.append(line[2:])  # strip "# " from header
            elif not line.startswith("#"):
                lines.append(line)
    if not lines:
        return rows

    # First non-comment line is the header
    reader = csv.DictReader(lines)
    for row in reader:
        parsed = {}
        for k, v in row.items():
            if k is None:
                continue
            try:
                parsed[k] = float(v)
            except (ValueError, TypeError):
                parsed[k] = v
        rows.append(parsed)
    return rows


def logs_to_arrays(rows, step_key="step", value_key="mean_episode_return"):
    """Extract parallel arrays from log rows.

    Args:
        rows: List of dicts from load_training_logs().
        step_key: Key for x-axis values.
        value_key: Key for y-axis values.

    Returns:
        (steps, values) as numpy arrays, with NaN entries removed.
    """
    steps, values = [], []
    for row in rows:
        s = row.get(step_key)
        v = row.get(value_key)
        if s is not None and v is not None:
            try:
                sf, vf = float(s), float(v)
                if not np.isnan(vf):
                    steps.append(sf)
                    values.append(vf)
            except (ValueError, TypeError):
                continue
    return np.array(steps), np.array(values)


def smooth(values, window=50):
    """Simple moving average smoothing.

    Args:
        values: 1D array of values.
        window: Smoothing window size.

    Returns:
        Smoothed array (same length, padded at edges).
    """
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


# ─── Plotting ────────────────────────────────────────────────────────

def plot_learning_curve(steps, values, label=None, color=None, smooth_window=50,
                        ax=None, xlabel="Environment Steps", ylabel="Episode Return",
                        title=None, alpha_raw=0.2):
    """Plot a learning curve with raw data + smoothed overlay.

    Args:
        steps: Array of step values (x-axis).
        values: Array of return values (y-axis).
        label: Legend label.
        color: Line color.
        ax: Matplotlib axes. Created if None.
        smooth_window: Window size for moving average.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        alpha_raw: Alpha for raw data points.

    Returns:
        Matplotlib axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Raw data
    ax.plot(steps, values, alpha=alpha_raw, color=color, linewidth=0.5)
    # Smoothed
    smoothed = smooth(values, smooth_window)
    ax.plot(steps, smoothed, label=label, color=color, linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if label:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_loss_components(rows, loss_keys=None, ax=None, smooth_window=50):
    """Plot loss components over training steps.

    Args:
        rows: List of dicts from load_training_logs().
        loss_keys: List of loss key names to plot. Auto-detected if None.
        ax: Matplotlib axes. Created if None.
        smooth_window: Smoothing window.

    Returns:
        Matplotlib axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if loss_keys is None:
        # Auto-detect loss keys from first row
        loss_keys = [k for k in rows[0].keys()
                     if "loss" in k.lower() and k != "total_loss"]

    steps, _ = logs_to_arrays(rows, value_key="total_loss")
    for key in loss_keys:
        _, vals = logs_to_arrays(rows, value_key=key)
        if len(vals) > 0:
            smoothed = smooth(vals, smooth_window)
            ax.plot(steps[:len(smoothed)], smoothed, label=key, linewidth=1.5)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


# ─── Rendering ───────────────────────────────────────────────────────

def render_frame(env, action=None):
    """Take one step and return the rendered frame as HWC uint8 array.

    If the env has already been reset and action is None, just returns the
    current observation (re-transposed to HWC if needed).

    Args:
        env: Gym environment (Memory Maze).
        action: Action to take. If None, uses action 0 (noop).

    Returns:
        RGB frame as numpy array with shape (H, W, 3), dtype uint8.
    """
    if action is not None:
        obs, _, _, _ = env.step(action)
    else:
        obs = env.reset()

    # Handle CHW format (from MemoryMazeWrapper)
    if obs.ndim == 3 and obs.shape[0] in (1, 3):
        obs = np.transpose(obs, (1, 2, 0))
    return obs


def display_frame(frame, ax=None, title=None):
    """Display an RGB frame inline in a notebook.

    Args:
        frame: HWC uint8 RGB array.
        ax: Matplotlib axes. Created if None.
        title: Optional title.

    Returns:
        Matplotlib axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(frame)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


# ─── Video Recording ────────────────────────────────────────────────

def record_episode(env, policy_fn=None, max_steps=1000, fps=10):
    """Record a single episode as a list of frames.

    Args:
        env: Gym environment.
        policy_fn: Callable(obs) -> action, or Callable(obs, reward) -> action.
            Two-arg form receives the previous step's reward (0.0 on first call).
            If None, uses random actions.
        max_steps: Maximum steps per episode.
        fps: Frames per second (for metadata only).

    Returns:
        (frames, total_return, num_steps) where frames is a list of HWC
        uint8 arrays.
    """
    frames = []
    obs = env.reset()
    # Store first frame
    frame = np.transpose(obs, (1, 2, 0)) if obs.ndim == 3 and obs.shape[0] in (1, 3) else obs
    frames.append(frame.copy())

    # Detect if policy_fn accepts a reward argument
    _pass_reward = False
    if policy_fn is not None:
        import inspect
        try:
            sig = inspect.signature(policy_fn)
            _pass_reward = len(sig.parameters) >= 2
        except (ValueError, TypeError):
            pass

    total_return = 0.0
    reward = 0.0
    for step in range(max_steps):
        if policy_fn is not None:
            action = policy_fn(obs, reward) if _pass_reward else policy_fn(obs)
        else:
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_return += reward
        frame = np.transpose(obs, (1, 2, 0)) if obs.ndim == 3 and obs.shape[0] in (1, 3) else obs
        frames.append(frame.copy())
        if done:
            break

    return frames, total_return, step + 1


def save_video(frames, path, fps=10):
    """Save frames as MP4 video.

    Args:
        frames: List of HWC uint8 RGB arrays.
        path: Output file path (e.g. 'episode.mp4').
        fps: Frames per second.
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("pip install imageio[ffmpeg] to save videos")

    imageio.mimsave(path, frames, fps=fps)


def display_video_in_notebook(path):
    """Display an MP4 video inline in a Jupyter notebook.

    Args:
        path: Path to the MP4 file.
    """
    from IPython.display import Video, display
    display(Video(path, embed=True, html_attributes="loop autoplay muted"))


# ─── Engine Comparison ───────────────────────────────────────────────

def compare_backends(seed=42, n_steps=100, env_id_mujoco="memory_maze:MemoryMaze-9x9-v0",
                     env_id_genesis="memory_maze:MemoryMaze-9x9-Genesis-v0"):
    """Run the same action sequence on both MuJoCo and Genesis backends.

    Args:
        seed: Random seed for reproducible actions.
        n_steps: Number of steps to compare.
        env_id_mujoco: MuJoCo environment ID.
        env_id_genesis: Genesis environment ID.

    Returns:
        Dict with keys:
            'mujoco_frames': list of HWC frames from MuJoCo
            'genesis_frames': list of HWC frames from Genesis
            'mujoco_rewards': list of rewards
            'genesis_rewards': list of rewards
            'actions': list of actions taken
            'mujoco_time': total step time for MuJoCo
            'genesis_time': total step time for Genesis
    """
    import gym

    rng = np.random.RandomState(seed)
    actions = [rng.randint(0, 6) for _ in range(n_steps)]

    results = {}

    for tag, env_id in [("mujoco", env_id_mujoco), ("genesis", env_id_genesis)]:
        try:
            env = gym.make(env_id, disable_env_checker=True, seed=seed)
        except Exception as e:
            print(f"Could not create {tag} env ({env_id}): {e}")
            results[f"{tag}_frames"] = []
            results[f"{tag}_rewards"] = []
            results[f"{tag}_time"] = 0.0
            continue

        obs = env.reset()
        frames = []
        rewards = []
        frame = np.transpose(obs, (1, 2, 0)) if obs.ndim == 3 and obs.shape[0] == 3 else obs
        frames.append(frame.copy())

        t0 = time.perf_counter()
        for a in actions:
            obs, reward, done, info = env.step(a)
            rewards.append(reward)
            frame = np.transpose(obs, (1, 2, 0)) if obs.ndim == 3 and obs.shape[0] == 3 else obs
            frames.append(frame.copy())
            if done:
                obs = env.reset()
        elapsed = time.perf_counter() - t0

        env.close()
        results[f"{tag}_frames"] = frames
        results[f"{tag}_rewards"] = rewards
        results[f"{tag}_time"] = elapsed

    results["actions"] = actions
    return results
