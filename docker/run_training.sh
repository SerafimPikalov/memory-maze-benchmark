#!/usr/bin/env bash
# Run Memory Maze training -- reads configuration from environment variables.
#
# Env vars:
#   AGENT       -- impala | dreamer (default: impala)
#   BACKEND     -- mujoco | genesis (default: mujoco)
#   MAZE_SIZE   -- 9x9 | 11x11 | 13x13 | 15x15 (default: 9x9)
#   TOTAL_STEPS -- override total_steps (default: 100000000)
#   NUM_ACTORS  -- IMPALA actors (default: 32)
#   NUM_ENVS    -- DreamerV2 envs (default: 8)
#   SEED        -- reproducibility seed (default: none)
#   WANDB_API_KEY -- W&B API key (enables --wandb automatically)
#   WANDB_PROJECT -- W&B project name (default: memorymaze)
#   WANDB_ENTITY  -- W&B entity / team (optional)
#   N_BATCHED_ACTORS -- number of batched actor processes (default: unset)
#   EXTRA_FLAGS -- additional flags passed to the training script
#
# Usage:
#   AGENT=impala BACKEND=mujoco ./run_training.sh
#   AGENT=dreamer BACKEND=genesis MAZE_SIZE=11x11 ./run_training.sh

set -euo pipefail

# Defaults
AGENT="${AGENT:-impala}"
BACKEND="${BACKEND:-mujoco}"
MAZE_SIZE="${MAZE_SIZE:-9x9}"
TOTAL_STEPS="${TOTAL_STEPS:-100000000}"
NUM_ACTORS="${NUM_ACTORS:-32}"
NUM_ENVS="${NUM_ENVS:-8}"
N_BATCHED_ACTORS="${N_BATCHED_ACTORS:-}"
SEED="${SEED:-}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# Resolve paths
APP_DIR="${APP_DIR:-/app}"
SAVEDIR="${SAVEDIR:-/workspace/logs}"

# Build env ID
ENV_ID="memory_maze:MemoryMaze-${MAZE_SIZE}-v0"

echo "============================================================"
echo "Memory Maze Training"
echo "============================================================"
echo "  Agent:       ${AGENT}"
echo "  Backend:     ${BACKEND}"
echo "  Maze size:   ${MAZE_SIZE}"
echo "  Env ID:      ${ENV_ID}"
echo "  Total steps: ${TOTAL_STEPS}"
echo "  Batched actors: ${N_BATCHED_ACTORS:-<default>}"
echo "  Seed:        ${SEED:-<none>}"
echo "  Save dir:    ${SAVEDIR}"
echo "============================================================"

# Validate
if [[ "$AGENT" != "impala" && "$AGENT" != "dreamer" ]]; then
    echo "ERROR: AGENT must be 'impala' or 'dreamer', got '${AGENT}'"
    exit 1
fi

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

# Multi-actor batched flags (IMPALA + Genesis batched only)
BATCHED_ACTOR_FLAGS=""
if [ -n "${N_BATCHED_ACTORS}" ]; then
    BATCHED_ACTOR_FLAGS="--n_batched_actors ${N_BATCHED_ACTORS}"
fi

# Build command
if [ "$AGENT" = "impala" ]; then
    CMD="python ${APP_DIR}/train_impala.py \
        --env ${ENV_ID} \
        --backend ${BACKEND} \
        --num_actors ${NUM_ACTORS} \
        --total_steps ${TOTAL_STEPS} \
        --savedir ${SAVEDIR} \
        ${BATCHED_ACTOR_FLAGS} \
        ${WANDB_FLAGS} \
        ${SEED_FLAGS} \
        ${EXTRA_FLAGS}"
elif [ "$AGENT" = "dreamer" ]; then
    CMD="python ${APP_DIR}/train_dreamer.py \
        --env ${ENV_ID} \
        --backend ${BACKEND} \
        --num_envs ${NUM_ENVS} \
        --total_steps ${TOTAL_STEPS} \
        --savedir ${SAVEDIR} \
        ${WANDB_FLAGS} \
        ${SEED_FLAGS} \
        ${EXTRA_FLAGS}"
fi

# Determine if xvfb-run is needed for headless rendering.
# BatchRenderer (batched Genesis mode) uses Vulkan -- no X11 needed.
# Rasterizer (non-batched Genesis, MuJoCo) uses OpenGL -- needs a display.
NEEDS_XVFB=false
if [ -z "${DISPLAY:-}" ]; then
    if [ "$BACKEND" = "genesis" ]; then
        # Batched mode uses BatchRenderer (Vulkan) -- skip xvfb-run
        # Non-batched uses Rasterizer (OpenGL) -- needs xvfb-run
        if echo "${EXTRA_FLAGS}" | grep -q -- "--batched"; then
            export MUJOCO_GL=egl
            export PYOPENGL_PLATFORM=egl
        else
            NEEDS_XVFB=true
        fi
    elif [ "$BACKEND" = "mujoco" ]; then
        NEEDS_XVFB=true
    fi
fi

if [ "$NEEDS_XVFB" = true ]; then
    CMD="xvfb-run ${CMD}"
fi

echo "Running: ${CMD}"
echo "============================================================"

exec ${CMD}
