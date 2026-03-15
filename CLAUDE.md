# CLAUDE.md

## Project

Memory Maze benchmark — training and evaluating RL agents (IMPALA V-trace) on the Memory Maze 3D navigation task with MuJoCo and Genesis physics backends. Includes 5 guided Jupyter notebooks, one-command RunPod GPU deployment, and 4 custom Claude Code AI agents for domain-specific assistance.

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

# Evaluate
python train_impala.py --mode test --xpid <experiment_id>
```

## Key Files

| File | Description |
|------|-------------|
| `train_impala.py` | IMPALA V-trace training (3 regimes: MuJoCo, Genesis single, Genesis batched) |
| `benchmark_physics.py` | Physics preset benchmark |
| `ARCHITECTURE.md` | Process/thread diagrams |
| `torchbeast/` | Vendored TorchBeast modules (Apache 2.0) |
| `runpod/pod_manager.py` | One-command GPU pod deployment |

## Notebooks

5 guided Jupyter notebooks in `notebooks/` — the main way to explore the project interactively:

| # | Notebook | What it does |
|---|----------|-------------|
| 1 | `01_environment_tour` | Create environments, inspect obs/actions, render frames |
| 2 | `02_train_and_plot` | Launch training, plot learning curves |
| 3 | `03_evaluate_and_record` | Load checkpoint, evaluate, record video |
| 4 | `04_engine_comparison` | MuJoCo vs Genesis side-by-side |
| 5 | `05_model_playground` | Inspect model internals, visualize features |

Run with: `cd notebooks && jupyter lab`

## RunPod GPU Deployment

```bash
pip install runpod
export RUNPOD_API_KEY="your-key"
python runpod/pod_manager.py gpus                                          # list GPUs
python runpod/pod_manager.py create --workload smoke_test --backend genesis # create pod
python runpod/pod_manager.py list                                          # check status
python runpod/pod_manager.py terminate <pod_id>                            # cleanup
```

## Custom Agents

Four domain-expert agents in `.claude/agents/`:
- **training-expert** — IMPALA architecture, training dynamics, performance
- **genesis-expert** — Genesis physics engine, BatchRenderer, batched simulation
- **maze-expert** — Memory Maze environment, dm_control/MuJoCo stack, gym API
- **runpod-manager** — GPU pod lifecycle: create, monitor, cost tracking, spot instances
