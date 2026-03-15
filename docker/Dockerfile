# Memory Maze Training -- GPU Docker Image
# Supports 2 training configs: IMPALA x {MuJoCo, Genesis}
#
# Build from the repository root (on Apple Silicon, must target amd64):
#   docker build --platform linux/amd64 -f docker/Dockerfile -t memorymaze-train .
#
# Build from the repository root (on Linux x86_64):
#   docker build -f docker/Dockerfile -t memorymaze-train .
#
# Run:
#   docker run --gpus all -e AGENT=impala -e BACKEND=mujoco memorymaze-train

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# -- Headless rendering environment ------------------------------------
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

# -- EGL + Vulkan system libraries -------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgles2 \
    libegl1-mesa-dev \
    libgl1 \
    libglvnd0 \
    libglx0 \
    libxrender1 \
    xvfb \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# -- NVIDIA EGL vendor config ------------------------------------------
# Without this JSON, MuJoCo/Genesis silently fall back to Mesa CPU
# rendering (10-50x slower, no errors). This tells libglvnd to use
# NVIDIA's EGL implementation.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# -- NVIDIA Vulkan ICD (required by Madrona BatchRenderer) -------------
# BatchRenderer uses Vulkan for GPU rasterization. Without this ICD
# descriptor, vkCreateInstance fails and Genesis batched mode cannot
# render frames.
RUN mkdir -p /usr/share/vulkan/icd.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}' \
    > /usr/share/vulkan/icd.d/nvidia_icd.json

# -- Python dependencies -----------------------------------------------
RUN pip install --no-cache-dir \
    genesis-world \
    "gym>=0.21,<1.0" \
    dm_control \
    labmaze \
    PyOpenGL==3.1.5 \
    gs-madrona==0.0.7.post2 \
    wandb \
    imageio[ffmpeg] \
    matplotlib \
    pandas \
    "ipywidgets>=8.0" \
    seaborn \
    pytest \
    ruff

# -- Patch gs_madrona HLSL shader (uint8 clamp fix) --------------------
# Stock gs-madrona has a uint8 overflow bug in linearToSRGB8(): when
# linearToSRGB() returns >1.0, the value wraps mod 256 (yellow turns
# green). Fix: clamp srgb to [0,1] before uint8 cast. The HLSL shaders
# are JIT-compiled to SPIR-V at runtime, so patching the text file works.
# Note: avoid `import gs_madrona` -- it loads a C extension, slow under QEMU.
RUN GS_DIR=$(pip show gs-madrona 2>/dev/null | grep ^Location | cut -d' ' -f2)/gs_madrona && \
    SHADER="$GS_DIR/src/render/shaders/draw_deferred_rgb.hlsl" && \
    test -f "$SHADER" && \
    python3 -c "p='$SHADER'; t=open(p).read(); assert 'float3ToUint32(srgb)' in t; t=t.replace('    return float3ToUint32(srgb);','    srgb = clamp(srgb, 0.0f, 1.0f);\n    return float3ToUint32(srgb);'); open(p,'w').write(t); print('Patched:',p)" && \
    grep -q "clamp(srgb" "$SHADER" && echo "Verified clamp present" || (echo "PATCH FAILED" && exit 1)

# -- CUDA headers for gs_madrona JIT compilation -----------------------
# gs_madrona compiles device code at runtime using a baked-in relative
# path (./cuda/include). Symlink to the pip-installed CUDA headers.
RUN GS_DIR=$(pip show gs-madrona 2>/dev/null | grep ^Location | cut -d' ' -f2)/gs_madrona && \
    CUDA_INC=$(pip show nvidia-cuda-runtime-cu12 2>/dev/null | grep ^Location | cut -d' ' -f2)/nvidia/cuda_runtime/include && \
    mkdir -p "$GS_DIR/cuda" && \
    ln -sf "$CUDA_INC" "$GS_DIR/cuda/include" && \
    test -f "$GS_DIR/cuda/include/cuda_runtime_api.h" && echo "CUDA headers linked" || echo "Warning: CUDA headers not found (will resolve on native hardware)"

# -- Application code ---------------------------------------------------
WORKDIR /app

# Training and benchmark scripts
COPY train_impala.py benchmark_physics.py benchmark_backends.py ./

# TorchBeast pure-Python core modules (vendored as proper package)
COPY torchbeast/ torchbeast/

# Example notebooks
COPY notebooks/ notebooks/

# Docker support scripts
COPY docker/smoke_test.py docker/run_training.sh ./

# Tests and Makefile (for `make test` inside container)
COPY tests/ tests/
COPY Makefile pyproject.toml ./

# Memory Maze with Genesis backend (from fork)
RUN pip install --no-cache-dir "memory-maze[genesis] @ git+https://github.com/SerafimPikalov/memory-maze.git@genesis"

# -- Genesis environment ------------------------------------------------
# Skip Tk init (no display in headless)
ENV GENESIS_SKIP_TK_INIT=1
ENV MPLBACKEND=Agg
# Persist Taichi JIT cache across container restarts
ENV TI_CACHE_PATH=/workspace/taichi_cache

# -- Default env vars (override at container creation) ------------------
ENV AGENT=impala
ENV BACKEND=mujoco
ENV MAZE_SIZE=9x9
ENV WANDB_SILENT=true

# Image version — agents check this to detect stale images.
ENV MEMORYMAZE_IMAGE_VERSION=3

# /start.sh is the RunPod base image entrypoint (starts SSH + Jupyter).
# For plain Docker: docker run --gpus all -it image bash
# For RunPod: /start.sh runs automatically, keeping the container alive.
CMD ["/start.sh"]
