.DEFAULT_GOAL := help

# Detect platform for MuJoCo rendering backend
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  export MUJOCO_GL ?= glfw
else
  export MUJOCO_GL ?= egl
endif

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install core training dependencies
	pip install -r requirements.txt

install-notebooks:  ## Install notebook dependencies
	pip install -r requirements-notebooks.txt

smoke-test:  ## Quick sanity check (no GPU needed)
	python -c "import memory_maze; print('memory-maze OK')"
	python -c "from train_impala import MemoryMazeNet; print('IMPALA model OK')"
	python -c "from train_dreamer import DreamerModel; print('DreamerV2 model OK')"

train-smoke-impala:  ## Short IMPALA training run (~2 min)
	python train_impala.py --num_actors 2 --total_steps 1000 --batch_size 2

train-smoke-dreamer:  ## Short DreamerV2 training run (~2 min)
	python train_dreamer.py --num_envs 2 --total_steps 2000 --batch_size 2 --sequence_length 8

train:  ## Train IMPALA on 9x9 maze (default settings)
	python train_impala.py --num_actors 8 --total_steps 10_000_000

train-genesis:  ## Train IMPALA with Genesis batched mode (GPU)
	python train_impala.py --backend genesis --batched --physics_timestep 0.05 --total_steps 10_000_000

benchmark:  ## Run physics preset benchmark
	python benchmark_physics.py --episodes 10

lint:  ## Check code style
	ruff check .
	ruff format --check .

clean:  ## Remove generated artifacts
	rm -rf logs/ checkpoints/ recordings/ __pycache__/ wandb/

.PHONY: help install install-notebooks smoke-test train-smoke-impala train-smoke-dreamer train train-genesis benchmark lint clean
