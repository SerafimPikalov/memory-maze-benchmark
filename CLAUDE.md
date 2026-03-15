# CLAUDE.md

## Project

Memory Maze benchmark — training and evaluating RL agents (IMPALA V-trace) on the Memory Maze 3D navigation task with MuJoCo and Genesis physics backends. Includes 5 guided Jupyter notebooks, one-command RunPod GPU deployment, and 4 custom Claude Code AI agents for domain-specific assistance.

## Getting Started

When a user asks to "set up", "try this project", "get started", or "run tests", follow the path that matches their environment:

### Path A: Local machine (no GPU)
```bash
pip install -r requirements.txt
make test-fast              # import checks + env creation + model forward pass (~10s)
cd notebooks && jupyter lab # explore interactively
```

### Path B: Local machine with NVIDIA GPU (Docker)
```bash
docker build -f docker/Dockerfile -t mmaze .
docker run --gpus all mmaze python smoke_test.py     # validate CUDA, EGL, backends (~60s)
docker run --gpus all -it mmaze                      # interactive shell
docker run --gpus all -p 8888:8888 mmaze jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### Path C: Cloud GPU (RunPod)
Use the **runpod-manager** agent — it will ask what the user wants to do, gather credentials, and create the pod. Invoke with: "set up a runpod" or "create a GPU pod".

### Docker image rebuild
The Docker Hub image may be stale. If bugs appear that are already fixed in the repo, rebuild:
```bash
./docker/deploy.sh                                    # build locally
DOCKER_REPO=youruser/image ./docker/deploy.sh          # build + push to your Docker Hub
```

### When unsure which path
Ask the user: "Do you have a local NVIDIA GPU, or should we use cloud (RunPod)? Or just explore locally without GPU?"

## Build & Run

```bash
# Install
pip install -r requirements.txt

# Run tests
make test           # smoke + environment + training unit tests (~30s)
make test-all       # includes slow 200-step integration test (~60s)

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
| `benchmark_physics.py` | MuJoCo physics preset benchmark (timestep, iterations) |
| `benchmark_backends.py` | MuJoCo vs Genesis comparison (SPS at different parallelism levels) |
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
- **runpod-manager** — GPU pod lifecycle: create, monitor, cost tracking, spot instances. **Always use this agent for cloud GPU setup** — it gathers requirements (credentials, intent, testing preference) before creating pods.
