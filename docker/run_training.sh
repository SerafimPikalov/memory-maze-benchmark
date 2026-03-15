#!/usr/bin/env bash
# Run Memory Maze IMPALA training -- reads configuration from environment variables.
#
# Env vars:
#   BACKEND     -- mujoco | genesis (default: mujoco)
#   BATCHED     -- 1 to enable Genesis batched mode (default: unset)
#   MAZE_SIZE   -- 9x9 | 11x11 | 13x13 | 15x15 (default: 9x9)
#   TOTAL_STEPS -- override total_steps (default: 100000000)
#   NUM_ACTORS  -- IMPALA actors (default: 32)
#   PHYSICS_TIMESTEP -- Genesis physics timestep (default: 0.05 when batched)
#   SEED        -- reproducibility seed (default: none)
#   WANDB_API_KEY -- W&B API key (enables --wandb automatically)
#   WANDB_PROJECT -- W&B project name (default: memorymaze)
#   WANDB_ENTITY  -- W&B entity / team (optional)
#   N_BATCHED_ACTORS -- number of batched actor processes (default: unset)
#   EXTRA_FLAGS -- additional flags passed to the training script
#   SAVEDIR     -- log/checkpoint directory (default: /workspace/logs)
#
# Usage:
#   BACKEND=mujoco ./run_training.sh
#   BACKEND=genesis BATCHED=1 ./run_training.sh

set -euo pipefail

# Raise file descriptor limit — IMPALA with many actors needs >1024
ulimit -n 65536 2>/dev/null || true

# Defaults
BACKEND="${BACKEND:-mujoco}"
BATCHED="${BATCHED:-}"
MAZE_SIZE="${MAZE_SIZE:-9x9}"
TOTAL_STEPS="${TOTAL_STEPS:-100000000}"
NUM_ACTORS="${NUM_ACTORS:-32}"
N_BATCHED_ACTORS="${N_BATCHED_ACTORS:-}"
PHYSICS_TIMESTEP="${PHYSICS_TIMESTEP:-}"
SEED="${SEED:-}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# Resolve paths
APP_DIR="${APP_DIR:-/app}"
SAVEDIR="${SAVEDIR:-/workspace/logs}"

# EGL for headless rendering (Dockerfile sets this too, but be explicit)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Build env ID
ENV_ID="memory_maze:MemoryMaze-${MAZE_SIZE}-v0"

echo "============================================================"
echo "Memory Maze IMPALA Training"
echo "============================================================"
echo "  Backend:     ${BACKEND}"
echo "  Batched:     ${BATCHED:-no}"
echo "  Maze size:   ${MAZE_SIZE}"
echo "  Env ID:      ${ENV_ID}"
echo "  Total steps: ${TOTAL_STEPS}"
echo "  Num actors:  ${NUM_ACTORS}"
echo "  Seed:        ${SEED:-<none>}"
echo "  Save dir:    ${SAVEDIR}"
echo "============================================================"

# Validate
if [[ "$BACKEND" != "mujoco" && "$BACKEND" != "genesis" ]]; then
    echo "ERROR: BACKEND must be 'mujoco' or 'genesis', got '${BACKEND}'"
    exit 1
fi

# Ensure save directory exists
mkdir -p "${SAVEDIR}"

# W&B flags
WANDB_FLAGS=""
if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_FLAGS="--wandb"
    if [ -n "${WANDB_PROJECT:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_project ${WANDB_PROJECT}"
    fi
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_entity ${WANDB_ENTITY}"
    fi
    echo "  W&B:         enabled (project=${WANDB_PROJECT:-memorymaze})"
fi

# Seed flags
SEED_FLAGS=""
if [ -n "${SEED}" ]; then
    SEED_FLAGS="--seed ${SEED}"
fi

# Batched mode flags
BATCHED_FLAGS=""
if [ -n "${BATCHED}" ] && [ "${BATCHED}" = "1" ]; then
    BATCHED_FLAGS="--batched --physics_timestep ${PHYSICS_TIMESTEP:-0.05}"
fi

# Multi-actor batched flags
BATCHED_ACTOR_FLAGS=""
if [ -n "${N_BATCHED_ACTORS}" ]; then
    BATCHED_ACTOR_FLAGS="--n_batched_actors ${N_BATCHED_ACTORS}"
fi

# Build command
CMD="python ${APP_DIR}/train_impala.py \
    --env ${ENV_ID} \
    --backend ${BACKEND} \
    --num_actors ${NUM_ACTORS} \
    --total_steps ${TOTAL_STEPS} \
    --savedir ${SAVEDIR} \
    ${BATCHED_FLAGS} \
    ${BATCHED_ACTOR_FLAGS} \
    ${WANDB_FLAGS} \
    ${SEED_FLAGS} \
    ${EXTRA_FLAGS}"

echo "Running: ${CMD}"
echo "============================================================"

exec ${CMD}
