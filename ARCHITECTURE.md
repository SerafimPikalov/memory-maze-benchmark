# ARCHITECTURE.md — IMPALA Training on Memory Maze

This document describes the process/thread layout, data flow, and engine separation
in `train_impala.py` (1559 lines). Four training regimes are supported.

---

## 1. Training Regimes

### Regime 1: Unbatched MuJoCo

```
 ┌──────────────────────────────────────────────────────────────┐
 │  Main Process (PID 0)                                       │
 │                                                             │
 │   model.share_memory_()        ┌──────────────────────┐    │
 │   buffers.share_memory_()      │ batch_and_learn       │    │
 │          │                     │  thread 0 (daemon=F)  │    │
 │          │     free_queue ←──→ │  get_batch()          │    │
 │          │     full_queue ←──→ │  learn()              │    │
 │          │                     └──────────────────────┘    │
 │          │                     ┌──────────────────────┐    │
 │          │                     │ batch_and_learn       │    │
 │          │                     │  thread 1             │    │
 │          │                     └──────────────────────┘    │
 │          │                                                  │
 │   monitor_loop() ← runs on main thread                     │
 └──────┬───────┬───────┬──────────────────────────────────────┘
        │       │       │          mp.SimpleQueue (IPC)
 ┌──────┴──┐ ┌──┴───┐ ┌─┴─────┐
 │ Actor 0 │ │ A. 1 │ │ A. N  │   N separate OS processes
 │ (proc)  │ │      │ │       │   fork (Linux) / spawn (macOS)
 │         │ │      │ │       │
 │ act()   │ │act() │ │ act() │   Each creates its own:
 │ MuJoCo  │ │      │ │       │   - gym env (MuJoCo renderer)
 │ env     │ │      │ │       │   - TorchBeast Environment
 └─────────┘ └──────┘ └───────┘   - model copy (shared memory)
```

Command: `python train_impala.py --num_actors 8`

### Regime 2: Unbatched Genesis

Identical process layout to Regime 1. The only difference is `gym.make()` receives
a Genesis env ID (`MemoryMaze-9x9-Genesis-v0` instead of `MemoryMaze-9x9-v0`).
Training code is completely engine-agnostic — the env is selected by string ID.

Command: `python train_impala.py --backend genesis --num_actors 8`

### Regime 3: Batched Genesis (single-process)

```
 ┌──────────────────────────────────────────────────────────────┐
 │  Single Process                                             │
 │                                                             │
 │  ┌─────────────────────────────────────┐                    │
 │  │ Main Thread                         │                    │
 │  │                                     │                    │
 │  │  gs.init()                          │                    │
 │  │  act_batched()                      │                    │
 │  │    BatchGenesisMemoryMazeEnv        │                    │
 │  │    (N envs in one Genesis scene)    │                    │
 │  │                                     │                    │
 │  │  model inference (shared memory)    │                    │
 │  │  env.step(actions)  ← GPU render    │                    │
 │  └───────────┬─────────────────────────┘                    │
 │              │ queue.SimpleQueue (in-process)                │
 │  ┌───────────┴─────────────────────────┐                    │
 │  │ Learner Threads (daemon=True)       │                    │
 │  │  batch_and_learn thread 0           │                    │
 │  │  batch_and_learn thread 1           │                    │
 │  └─────────────────────────────────────┘                    │
 │  ┌─────────────────────────────────────┐                    │
 │  │ Monitor Thread (daemon=True)        │                    │
 │  │  monitor_loop()                     │                    │
 │  └─────────────────────────────────────┘                    │
 └──────────────────────────────────────────────────────────────┘
```

Command: `python train_impala.py --backend genesis --batched --num_actors 8`

Genesis/Taichi requires `gs.init()` + scene build + EGL rendering all on the main
thread (FieldsBuilder is global state), so the batched actor runs on main and
learners run as daemon threads. `gs.init()` is called at line 1452 in `train()`,
after PyTorch optimizer setup (to avoid corrupting setuptools/distutils imports
that Triton needs).

### Regime 4: Multi-process Batched Genesis

```
 ┌──────────────────────────────────────────────────────────────┐
 │  Main Process (PID 0)                                       │
 │                                                             │
 │   model.share_memory_()        ┌──────────────────────┐    │
 │   buffers.share_memory_()      │ batch_and_learn       │    │
 │          │                     │  thread 0 (daemon=T)  │    │
 │          │     free_queue ←──→ │  get_batch()          │    │
 │          │     full_queue ←──→ │  learn()              │    │
 │          │                     └──────────────────────┘    │
 │          │                     ┌──────────────────────┐    │
 │          │                     │ batch_and_learn       │    │
 │          │                     │  thread 1             │    │
 │          │                     └──────────────────────┘    │
 │          │                                                  │
 │   monitor_loop_multi() ← runs on main thread               │
 │     (includes dead actor detection)                         │
 └──────┬───────┬───────┬──────────────────────────────────────┘
        │       │       │    mp.SimpleQueue (spawn context IPC)
 ┌──────┴──┐ ┌──┴───┐ ┌─┴─────┐
 │ Actor 0 │ │ A. 1 │ │ A. K  │   K spawned processes
 │ (proc)  │ │      │ │       │   mp.get_context("spawn")
 │         │ │      │ │       │
 │ gs.init │ │      │ │       │   Each process independently:
 │ act_    │ │      │ │       │   - calls gs.init() in
 │ batched │ │      │ │       │     BatchGenesisMemoryMazeEnv
 │ M envs  │ │      │ │       │   - builds its own Genesis scene
 │         │ │      │ │       │   - runs act_batched() with M envs
 └─────────┘ └──────┘ └───────┘   - fills M buffer slots per unroll
                                  Total envs = K × M = num_actors
```

Command: `python train_impala.py --backend genesis --batched --n_batched_actors 4 --num_actors 128`

This splits `num_actors` environments across K separate OS processes, each running
its own Genesis scene with M = num_actors / K environments. Unlike Regime 3, the
parent process does NOT call `gs.init()` — each child initializes Genesis
independently. This avoids the Taichi global-state limitation (one scene per
process) while still using batched physics/rendering within each process.

Key differences from Regime 3:
- Uses `mp.get_context("spawn")` for clean Taichi/CUDA state in children
- Each child sets `QD_OFFLINE_CACHE_FILE_PATH` to avoid kernel compilation races
- `monitor_loop_multi()` includes dead actor detection and per-actor SPS tracking
- `step_counter` is `mp.Value("i")` (cross-process) instead of a plain list
- `num_actors` must be evenly divisible by `n_batched_actors`

### CPU vs GPU Summary

| Component           | Regime 1 (MuJoCo)       | Regime 2 (Genesis unbatched) | Regime 3 (Genesis batched 1-proc) | Regime 4 (Genesis batched N-proc)  |
|---------------------|-------------------------|------------------------------|-----------------------------------|------------------------------------|
| Physics             | CPU (MuJoCo C)          | CPU (Genesis/Taichi)         | CPU or GPU (Taichi kernel)        | CPU or GPU (Taichi kernel)         |
| Rendering           | CPU (GLFW/OSMesa)       | CPU (Rasterizer)             | GPU (BatchRenderer)               | GPU (BatchRenderer per process)    |
| Model inference     | CPU (actor processes)   | CPU (actor processes)        | GPU if CUDA available             | GPU if CUDA (shared-memory model)  |
| Learner backward    | GPU if CUDA, else CPU   | GPU if CUDA, else CPU        | GPU if CUDA, else CPU             | GPU if CUDA, else CPU              |
| IPC mechanism       | mp.SimpleQueue (OS)     | mp.SimpleQueue (OS)          | queue.SimpleQueue (thread)        | mp.SimpleQueue (spawn context)     |
| Parallelism         | N processes             | N processes                  | 1 process, threads                | K processes + threads              |

---

## 2. Data Flow & Borders

### Wrapper Stack (Unbatched)

```
 ┌─────────────────────────────────────────────────────────────┐
 │  act() loop                                                 │
 │                                                             │
 │  ┌───────────────────────────────────────────────────────┐  │
 │  │ TorchBeast Environment                                │  │
 │  │   environment.py                                      │  │
 │  │                                                       │  │
 │  │   .initial() → dict of (1,1,...) tensors              │  │
 │  │   .step(action_tensor) → dict of (1,1,...) tensors    │  │
 │  │                                                       │  │
 │  │   Adds: reward, done, episode_return, episode_step,   │  │
 │  │         last_action tracking. Auto-resets on done.     │  │
 │  │                                                       │  │
 │  │  ┌─────────────────────────────────────────────────┐  │  │
 │  │  │ MemoryMazeWrapper (gym.Wrapper)                 │  │  │
 │  │  │   train_impala.py:195                           │  │  │
 │  │  │                                                 │  │  │
 │  │  │   HWC (64,64,3) uint8 → CHW (3,64,64) uint8    │  │  │
 │  │  │                                                 │  │  │
 │  │  │  ┌───────────────────────────────────────────┐  │  │  │
 │  │  │  │ gym.Env  (engine-specific)                │  │  │  │
 │  │  │  │                                           │  │  │  │
 │  │  │  │  MuJoCo path:                             │  │  │  │
 │  │  │  │    GymWrapper → dm_control MemoryMaze     │  │  │  │
 │  │  │  │                                           │  │  │  │
 │  │  │  │  Genesis path:                            │  │  │  │
 │  │  │  │    GenesisMemoryMazeEnv (native gym.Env)  │  │  │  │
 │  │  │  │                                           │  │  │  │
 │  │  │  │  .reset() → obs (H,W,3) uint8 ndarray    │  │  │  │
 │  │  │  │  .step(int) → obs, float, bool, dict      │  │  │  │
 │  │  │  └───────────────────────────────────────────┘  │  │  │
 │  │  └─────────────────────────────────────────────────┘  │  │
 │  └───────────────────────────────────────────────────────┘  │
 └─────────────────────────────────────────────────────────────┘
```

### Wrapper Stack (Batched — Regimes 3 & 4)

```
 ┌─────────────────────────────────────────────────────────────┐
 │  act_batched() loop                                         │
 │                                                             │
 │  No TorchBeast Environment — act_batched manages tensors    │
 │  and episode tracking directly.                             │
 │                                                             │
 │  ┌───────────────────────────────────────────────────────┐  │
 │  │ BatchGenesisMemoryMazeEnv                             │  │
 │  │   genesis_backend.py                                  │  │
 │  │                                                       │  │
 │  │   NOT a gym.Env — custom vectorized interface         │  │
 │  │                                                       │  │
 │  │   .reset() → obs (N,H,W,3) uint8 ndarray             │  │
 │  │   .step(actions) → obs, rewards, dones, infos         │  │
 │  │     actions: (N,) int64 ndarray                       │  │
 │  │     obs:     (N,H,W,3) uint8 ndarray                  │  │
 │  │     rewards: (N,) float32 ndarray                     │  │
 │  │     dones:   (N,) bool ndarray                        │  │
 │  │     infos:   list[dict]                               │  │
 │  └───────────────────────────────────────────────────────┘  │
 └─────────────────────────────────────────────────────────────┘
```

### Contracts at Each Border

**Border 1: act() ↔ TorchBeast Environment**

```
env.initial() → dict:
    frame:          Tensor (1, 1, 3, 64, 64) uint8
    reward:         Tensor (1, 1)             float32
    done:           Tensor (1, 1)             uint8
    episode_return: Tensor (1, 1)             float32
    episode_step:   Tensor (1, 1)             int32
    last_action:    Tensor (1, 1)             int64

env.step(action: Tensor (1,1) int64) → same dict shape
```

**Border 2: TorchBeast Environment ↔ gym env**

```
gym_env.reset()       → ndarray (3, 64, 64) uint8   [after MemoryMazeWrapper]
gym_env.step(int)     → (ndarray (3,64,64), float, bool, dict)
```

**Border 3: act_batched() ↔ BatchGenesisMemoryMazeEnv**

```
batch_env.reset()             → ndarray (N, 64, 64, 3)  uint8  [HWC]
batch_env.step(ndarray (N,))  → (obs, rewards, dones, infos)
                                  obs:     (N, 64, 64, 3) uint8
                                  rewards: (N,)            float32
                                  dones:   (N,)            bool
```

act_batched() then transposes HWC→CHW and wraps into `(1, N, ...)` tensors itself.

### Shared-Memory Buffer Mechanism

```
create_buffers()                     Actors                    Learner
 │                                    │                          │
 │  for each of num_buffers slots:    │                          │
 │    torch.empty(T+1, ...).          │                          │
 │          share_memory_()           │                          │
 │         │                          │                          │
 │         ▼                          │                          │
 │  buffers: {key: [Tensor, ...]}  ───┤──────────────────────────┤
 │  initial_agent_state_buffers    ───┤──────────────────────────┤
 │                                    │                          │
 │  free_queue.put(0..num_buffers)    │                          │
 │                                    │                          │
 │                              idx = free_queue.get()           │
 │                              write unroll into buffers[idx]   │
 │                              full_queue.put(idx)              │
 │                                    │                          │
 │                                    │   indices = [full_queue.get()
 │                                    │              for _ in batch_size]
 │                                    │   batch = stack(buffers[indices])
 │                                    │   learn(batch)
 │                                    │   free_queue.put(indices)
```

Buffer shapes per slot (T = unroll_length, default 100):

| Key             | Shape           | Dtype   |
|-----------------|-----------------|---------|
| frame           | (T+1, 3, 64, 64) | uint8   |
| reward          | (T+1,)          | float32 |
| done            | (T+1,)          | bool    |
| episode_return  | (T+1,)          | float32 |
| episode_step    | (T+1,)          | int32   |
| policy_logits   | (T+1, 6)       | float32 |
| baseline        | (T+1,)         | float32 |
| last_action     | (T+1,)         | int64   |
| action          | (T+1,)         | int64   |

After `get_batch()` stacks B slots: each tensor becomes `(T+1, B, ...)`.

---

## 3. Network Architecture

`MemoryMazeNet` (line 321) — IMPALA-style ResNet + LSTM:

```
RGB (3,64,64)
  → ConvBlock(3→16)   64→32 spatial
  → ConvBlock(16→32)  32→16 spatial
  → ConvBlock(32→32)  16→8 spatial
  → ReLU → Flatten (32×8×8 = 2048)
  → FC(2048→256) → ReLU
  → concat(features[256], one_hot_action[6], reward[1]) = 263
  → LSTM(263→256, 1 layer)
  → policy head: Linear(256→6)   [6 discrete actions]
  → value head:  Linear(256→1)
```

Each `ConvBlock`: Conv3x3 → MaxPool(3, stride=2) → 2× ResBlock(ReLU→Conv3x3→ReLU→Conv3x3 + skip).

---

## 4. Engine Separation

### Unbatched (Regimes 1 & 2): Clean separation

The engine boundary lives entirely inside `gym.make(env_id)`:

```
train_impala.py
  └─ create_env(flags)           # line 217
       └─ gym.make(env_id)       # selects engine by string ID
            │
            ├─ "MemoryMaze-9x9-v0"         → MuJoCo (dm_control + GymWrapper)
            └─ "MemoryMaze-9x9-Genesis-v0" → GenesisMemoryMazeEnv
```

Training code never imports MuJoCo or Genesis. Engine selection is purely by
gym registration ID. `act()`, `get_batch()`, `learn()` are all engine-agnostic.

### Batched (Regimes 3 & 4): Engine leaks into training code

The batched path breaks the clean separation. `train_impala.py` directly imports
Genesis in several places:

1. **`act_batched()` line 634**: `from memory_maze.genesis_backend import BatchGenesisMemoryMazeEnv`
2. **Regime 3 — `train()` line 1443**: `import genesis as gs` + `gs.init()` on main thread
3. **Regime 4 — `train()` lines 1306-1331**: parent spawns child processes; each child calls
   `gs.init()` inside `BatchGenesisMemoryMazeEnv.__init__()`. Parent does NOT call `gs.init()`.
4. **`test()` line 1510**: `import genesis as gs` + `gs.init()` for evaluation

This is because:
- `BatchGenesisMemoryMazeEnv` is not a `gym.Env` — it has a custom vectorized API
- Genesis requires `gs.init()` before scene construction, and this must happen on
  the main thread after PyTorch optimizer setup (Regime 3) or independently per
  child process (Regime 4)

A future task would abstract the batched path behind a `gym.vector.VectorEnv`-like
interface so training code stays engine-agnostic for all regimes.
