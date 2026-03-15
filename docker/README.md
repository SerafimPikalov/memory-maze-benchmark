# Docker Deployment

GPU Docker image for training Memory Maze agents. Supports two training
configurations: IMPALA x {MuJoCo, Genesis}.

The image is based on `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
and includes headless EGL/Vulkan rendering, Genesis with the Madrona
BatchRenderer, and all Python dependencies pre-installed.

## Image Contents

| Layer | What it provides |
|-------|------------------|
| Base image | Ubuntu 22.04, Python 3.11, PyTorch 2.4 + CUDA 12.4, SSH, JupyterLab |
| EGL/Vulkan libs | `libegl1`, `libvulkan1`, NVIDIA vendor JSON for headless GPU rendering |
| Python packages | `genesis-world`, `dm_control`, `labmaze`, `gs-madrona`, `wandb`, etc. |
| gs-madrona patch | uint8 clamp fix for HLSL shader (see Known Issues) |
| Application code | Training scripts, TorchBeast modules, Memory Maze (editable install) |

## 1. Local GPU

Build the image from the repository root (the build context must include
`train_impala.py`, `torchbeast/`, `memory-maze/`, etc.):

```bash
# On Apple Silicon (cross-compile to amd64):
docker build --platform linux/amd64 -f docker/Dockerfile -t memorymaze-train .

# On Linux x86_64:
docker build -f docker/Dockerfile -t memorymaze-train .
```

Run training:

```bash
# IMPALA on MuJoCo (default CMD)
docker run --gpus all memorymaze-train

# Interactive shell
docker run --gpus all -it memorymaze-train bash

# Run smoke test
docker run --gpus all memorymaze-train python smoke_test.py

# Use the env-var-driven launcher
docker run --gpus all \
    -e AGENT=impala \
    -e BACKEND=genesis \
    -e EXTRA_FLAGS="--batched --physics_timestep 0.05" \
    -e WANDB_API_KEY=your_key_here \
    memorymaze-train \
    bash run_training.sh
```

Mount a host directory to persist logs across container restarts:

```bash
docker run --gpus all \
    -v /path/to/logs:/workspace/logs \
    memorymaze-train
```

## 2. RunPod

### Create a Template

In the RunPod web console, go to **Templates > New Template** and configure:

| Field | Value |
|-------|-------|
| Container Image | `your-dockerhub-user/memorymaze-train:latest` |
| Container Disk | 20 GB |
| Volume Disk | 50 GB |
| Volume Mount Path | `/workspace` |
| Expose HTTP Ports | `8888,6006` |
| Expose TCP Ports | `22` |

### Environment Variables

Set these in the template or at pod creation:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT` | `impala` | `impala` |
| `BACKEND` | `mujoco` | `mujoco` or `genesis` |
| `MAZE_SIZE` | `9x9` | `9x9`, `11x11`, `13x13`, `15x15` |
| `TOTAL_STEPS` | `100000000` | Total training steps |
| `NUM_ACTORS` | `32` | IMPALA actor count |
| `N_BATCHED_ACTORS` | _(unset)_ | Batched actor processes (Genesis batched mode) |
| `SEED` | _(unset)_ | Reproducibility seed |
| `WANDB_API_KEY` | _(unset)_ | Enables W&B logging |
| `WANDB_PROJECT` | `memorymaze` | W&B project name |
| `WANDB_ENTITY` | _(unset)_ | W&B team/entity |
| `EXTRA_FLAGS` | _(unset)_ | Additional flags for the training script |

### Start a Pod

1. Create a pod from your template, selecting a GPU (A40 48GB recommended for
   Genesis batched mode; L4 24GB sufficient for MuJoCo).
2. Once the pod is running, SSH in or open the web terminal.
3. Run the smoke test first:

```bash
cd /app && python smoke_test.py
```

4. Launch training via the env-var launcher:

```bash
cd /app && bash run_training.sh
```

Or run directly:

```bash
cd /app && python train_impala.py \
    --backend genesis --batched --physics_timestep 0.05 \
    --num_actors 32 --total_steps 100000000 \
    --savedir /workspace/logs --wandb
```

### RunPod Hook Scripts

If you want training to start automatically when the pod boots, create
`/pre_start.sh` and `/post_start.sh` in your Dockerfile or volume. RunPod's
base image `/start.sh` runs them in order during initialization. For example:

```bash
# /post_start.sh -- auto-start training
#!/bin/bash
cd /app && nohup bash run_training.sh > /workspace/logs/training.log 2>&1 &
```

### Network Volume

Logs, checkpoints, and Taichi JIT cache are written to `/workspace/`. Attach a
network volume at pod creation to persist this data across pod restarts and
spot preemptions. The volume must be in the same datacenter as the pod.

### Spot Instances

Spot pods are 50-60% cheaper but can be interrupted with 5 seconds notice.
The training scripts checkpoint periodically to `--savedir`. After a spot
preemption, start a new pod with the same network volume and training will
resume from the latest checkpoint automatically (if using `--savedir /workspace/logs`).

## 3. Other Cloud GPUs

The image works on any cloud provider that supports NVIDIA GPU containers
(AWS, GCP, Azure, Lambda Labs, CoreWeave, etc.). Requirements:

- **NVIDIA GPU** with compute capability >= 7.0 (Volta or newer)
- **NVIDIA Container Toolkit** (`nvidia-docker2` or `nvidia-container-toolkit`)
- **Docker** or compatible runtime (Podman, Singularity with `--nv`)

Generic launch:

```bash
# Pull and run
docker pull your-dockerhub-user/memorymaze-train:latest
docker run --gpus all \
    -v /persistent/storage:/workspace \
    -e AGENT=impala \
    -e BACKEND=genesis \
    -e EXTRA_FLAGS="--batched --physics_timestep 0.05" \
    memorymaze-train \
    bash run_training.sh
```

For cloud instances without Docker, install dependencies manually following
the Dockerfile as a reference. The critical pieces are:

1. NVIDIA EGL vendor JSON at `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
2. Vulkan ICD at `/usr/share/vulkan/icd.d/nvidia_icd.json`
3. `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` environment variables
4. The gs-madrona HLSL shader patch (see below)

## Environment Variables Reference

These environment variables control rendering behavior and must be set
correctly for headless GPU environments:

| Variable | Value | Purpose |
|----------|-------|---------|
| `MUJOCO_GL` | `egl` | MuJoCo uses NVIDIA EGL for headless rendering |
| `PYOPENGL_PLATFORM` | `egl` | PyOpenGL uses EGL backend |
| `NVIDIA_DRIVER_CAPABILITIES` | `all` | Expose compute + graphics + utility to container |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Make all GPUs visible |
| `GENESIS_SKIP_TK_INIT` | `1` | Prevent Tk/Tcl initialization (no display) |
| `MPLBACKEND` | `Agg` | Matplotlib non-interactive backend |
| `TI_CACHE_PATH` | `/workspace/taichi_cache` | Persist Taichi JIT cache |

## Smoke Test

Always run the smoke test after deploying to a new environment:

```bash
python smoke_test.py                    # test everything
python smoke_test.py --backend mujoco   # MuJoCo only
python smoke_test.py --backend genesis  # Genesis only
```

The test verifies: CUDA availability, EGL configuration, BatchRenderer import,
and environment creation + stepping for each backend.

## Known Issues

### gs-madrona uint8 overflow patch

The stock `gs-madrona==0.0.7.post2` package has a bug in the HLSL shader
`draw_deferred_rgb.hlsl`: the `linearToSRGB8()` function does not clamp values
before casting to uint8, so values > 1.0 wrap modulo 256. This makes yellow
surfaces appear green. The Dockerfile patches this automatically by inserting a
`clamp(srgb, 0.0f, 1.0f)` call before the uint8 cast. If you install
gs-madrona outside of Docker, apply the same patch manually.

### First Genesis run is slow (JIT compilation)

Genesis uses Taichi as its compute backend. The first time a Genesis environment
runs, Taichi JIT-compiles GPU kernels, which takes 2-5 minutes. Subsequent runs
use the cached kernels from `TI_CACHE_PATH`. To avoid this delay on every pod
start, persist `/workspace/taichi_cache` on a network volume.

### NVIDIA Vulkan ICD required for BatchRenderer

The Madrona BatchRenderer (used in Genesis batched mode) requires Vulkan. If
the NVIDIA Vulkan ICD descriptor is missing at
`/usr/share/vulkan/icd.d/nvidia_icd.json`, `vkCreateInstance` will fail. The
Dockerfile creates this file automatically. If deploying without Docker, create
it manually:

```json
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3"
    }
}
```

### No /dev/dri on some cloud platforms

Some cloud platforms (including RunPod) do not expose `/dev/dri` (DRM render
nodes). This prevents GLX and some EGL paths from working. The NVIDIA EGL
surfaceless extension (`EGL_EXT_platform_device`) bypasses this requirement.
The `10_nvidia.json` vendor file directs libglvnd to use NVIDIA's EGL
implementation, which supports surfaceless rendering.

### Silent CPU rendering fallback

If NVIDIA EGL is misconfigured, MuJoCo and Genesis silently fall back to Mesa
software rendering. There is no error -- rendering just becomes 10-50x slower.
Verify GPU rendering is active by checking that the smoke test passes and that
training achieves expected steps-per-second rates.
