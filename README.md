# Memory Maze Benchmark: MuJoCo vs Genesis

Training and benchmarking RL agents on [Memory Maze](https://arxiv.org/abs/2210.13383) — a 3D maze environment for evaluating long-term memory in reinforcement learning. Supports both the original MuJoCo backend and a [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) GPU-accelerated backend.

## Key Findings

- **Physics is the bottleneck** (74-89% of step time), not rendering (~3ms regardless of backend)
- **Genesis `dt=0.05` gives 10x physics speedup** (460ms -> 49ms per step) with stable walker dynamics
- **IMPALA + LSTM** achieves mean return 9-12 at 23M steps (vs paper's 17-18)
- **IMPALA (V-trace) training** with four regimes: MuJoCo, Genesis single-env, Genesis batched (single-process), and Genesis batched (multi-process)

See [docs/bottleneck_analysis.md](docs/bottleneck_analysis.md) and [docs/physics_timestep_optimization.md](docs/physics_timestep_optimization.md) for the full data.

## Prerequisites

- Python >= 3.10
- MuJoCo backend: works on macOS (glfw) and Linux (EGL + NVIDIA GPU)
- Genesis backend: Linux + NVIDIA GPU (CUDA 12.x), >= 16 GB VRAM for batched mode
- Genesis depends on a [fork of memory-maze](https://github.com/SerafimPikalov/memory-maze/tree/genesis) that adds the Genesis backend. The fork is not merged upstream because the Genesis backend is experimental and requires additional dependencies (`genesis-world`, `gs-madrona`).

## Quick Start

```bash
# Install core dependencies
pip install -r requirements.txt

# Smoke test
make smoke-test

# Train IMPALA on 9x9 maze (MuJoCo backend)
python train_impala.py --num_actors 8 --total_steps 10_000_000

# Train with Genesis batched mode (GPU, ~10x faster physics)
python train_impala.py --backend genesis --batched --physics_timestep 0.05 --total_steps 10_000_000
```

## Training

### IMPALA (V-trace)

Four training regimes:

```bash
# 1. MuJoCo (default) — N separate actor processes
python train_impala.py --num_actors 8 --total_steps 10_000_000

# 2. Genesis single-env — N separate actor processes, Genesis physics
python train_impala.py --backend genesis --num_actors 8 --total_steps 10_000_000

# 3. Genesis batched (single-process) — main thread actor, GPU physics + BatchRenderer
python train_impala.py --backend genesis --batched --physics_timestep 0.05 --total_steps 10_000_000

# 4. Genesis batched (multi-process) — K actor processes, each with N/K batched envs
python train_impala.py --backend genesis --batched --n_batched_actors 4 --physics_timestep 0.05 --total_steps 10_000_000

# Evaluate trained agent
python train_impala.py --mode test --xpid <experiment_id>

# With W&B logging
python train_impala.py --wandb --wandb_project memory-maze --total_steps 10_000_000
```

## Architecture

IMPALA uses a ResNet encoder (3 blocks: 64->32->16->8 spatial) + LSTM(256) for recurrent memory, with 6 discrete actions and V-trace importance sampling.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed process/thread diagrams of all four training regimes.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Environment Tour](notebooks/01_environment_tour.ipynb) | Create environments, inspect observations and actions, render frames |
| 2 | [Train and Plot](notebooks/02_train_and_plot.ipynb) | Launch IMPALA training, plot learning curves |
| 3 | [Evaluate and Record](notebooks/03_evaluate_and_record.ipynb) | Load checkpoint, run evaluation, record video |
| 4 | [Engine Comparison](notebooks/04_engine_comparison.ipynb) | MuJoCo vs Genesis side-by-side |
| 5 | [Model Playground](notebooks/05_model_playground.ipynb) | Load trained model, inspect internals, visualize features |

## Docker (GPU)

Build from source recommended — the Docker Hub image (`serapikalov/memorymaze-train`) may be behind the repo.

```bash
# Build from source (recommended)
docker build -f docker/Dockerfile -t mmaze .

# Smoke test — validates CUDA, EGL, MuJoCo, Genesis (~60s)
docker run --gpus all mmaze python smoke_test.py

# Train IMPALA with MuJoCo (1M steps, ~30 min)
docker run --gpus all mmaze python train_impala.py --num_actors 8 --total_steps 1_000_000

# Train with Genesis batched mode (fastest)
docker run --gpus all mmaze python train_impala.py --backend genesis --batched --physics_timestep 0.05

# Interactive shell
docker run --gpus all -it mmaze

# Jupyter notebooks
docker run --gpus all -p 8888:8888 mmaze jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

See [docker/README.md](docker/README.md) for cloud deployment (RunPod, Lambda, etc.).

## RunPod GPU Pods

Spin up a GPU pod for training in one command:

```bash
pip install runpod
export RUNPOD_API_KEY="your-key"

# List available GPUs with pricing
python runpod/pod_manager.py gpus

# Create a pod for smoke testing
python runpod/pod_manager.py create --workload smoke_test --backend genesis

# Create a pod for full training (spot instance, cheapest GPU)
python runpod/pod_manager.py create --workload long_train --agent impala --backend genesis

# Monitor costs
python runpod/pod_manager.py cost

# Clean up when done
python runpod/pod_manager.py terminate <pod_id>
```

See `python runpod/pod_manager.py --help` for all commands.

## Repository Structure

```
.
├── train_impala.py          # IMPALA V-trace training (4 regimes)
├── benchmark_physics.py     # Physics preset benchmark
├── ARCHITECTURE.md          # Process/thread diagrams
├── torchbeast/              # Vendored pure-Python V-trace modules (Apache 2.0)
├── notebooks/               # 5 guided Jupyter notebooks
├── runpod/                  # GPU pod lifecycle manager (create, monitor, cost, cleanup)
├── docker/                  # GPU deployment (Dockerfile, smoke test, training launcher)
├── docs/                    # Research findings and analysis
│   ├── bottleneck_analysis.md
│   ├── physics_timestep_optimization.md
│   ├── engine_comparison.md
│   └── genesis_training_report.md
└── tests/                   # Smoke tests
```

## Known Limitations

- **Lighting model differs**: MuJoCo uses a camera-following headlight; Genesis uses fixed directional lights with high ambient. This affects visual appearance but training still converges.
- **Training results gap**: Our best Genesis IMPALA run reached mean return 9-12 at 23M steps (MuJoCo backend, default hyperparameters) vs the paper's 17-18 at 100M steps. The gap is expected — we trained for fewer steps and used a single seed. The paper's results required 100M+ steps with tuned hyperparameters.
- **Genesis/Taichi GPU memory**: Genesis does not fully free GPU memory when environments are destroyed. Long sessions with many env create/destroy cycles may require a process restart.
- **gs-madrona source build**: The BatchRenderer requires `gs-madrona` built from source with a uint8 clamp fix. The stock PyPI package has a color corruption bug.

## Related

- [memory-maze](https://github.com/SerafimPikalov/memory-maze/tree/genesis) — Memory Maze fork with Genesis backend (this repo depends on it)
- [Memory Maze paper](https://arxiv.org/abs/2210.13383) — "Evaluating Long-Term Memory in 3D Mazes" (Pasukonis et al., 2022)
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) — Universal physics engine for robotics and embodied AI

## License

Apache 2.0 — see [LICENSE](LICENSE). Training scripts and notebooks are original work. TorchBeast modules are vendored from [Facebook Research](https://github.com/facebookresearch/torchbeast) under Apache 2.0.
