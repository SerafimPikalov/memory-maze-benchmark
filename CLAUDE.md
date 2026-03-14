# CLAUDE.md

## Project

Memory Maze benchmark — training and evaluating RL agents (IMPALA, DreamerV2) on the Memory Maze 3D navigation task with MuJoCo and Genesis physics backends.

## Build & Run

```bash
# Install
pip install -r requirements.txt
pip install "memory-maze[genesis] @ git+https://github.com/SerafimPikalov/memory-maze.git@genesis"

# Smoke test
make smoke-test

# Train IMPALA (MuJoCo)
python train_impala.py --num_actors 8 --total_steps 10_000_000

# Train IMPALA (Genesis batched, 10x faster physics)
python train_impala.py --backend genesis --batched --physics_timestep 0.05 --total_steps 10_000_000

# Train DreamerV2
python train_dreamer.py --num_envs 8 --total_steps 100_000_000

# Evaluate
python train_impala.py --mode test --xpid <experiment_id>
```

## Key Files

| File | Description |
|------|-------------|
| `train_impala.py` | IMPALA V-trace training (3 regimes: MuJoCo, Genesis single, Genesis batched) |
| `train_dreamer.py` | DreamerV2 world-model training |
| `benchmark_physics.py` | Physics preset benchmark |
| `ARCHITECTURE.md` | Process/thread diagrams |
| `torchbeast/` | Vendored TorchBeast modules (Apache 2.0) |

## Custom Agents

Three domain-expert agents in `.claude/agents/`:
- **training-expert** — IMPALA/DreamerV2 architecture, training dynamics, performance
- **genesis-expert** — Genesis physics engine, BatchRenderer, batched simulation
- **maze-expert** — Memory Maze environment, dm_control/MuJoCo stack, gym API
