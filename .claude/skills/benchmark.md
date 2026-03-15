---
name: benchmark
description: "Run MuJoCo vs Genesis backend comparison benchmark"
user_invocable: true
---

Run the backend comparison benchmark and explain the results.

1. Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. If CUDA available:
   - Run: `python benchmark_backends.py --backends mujoco,genesis-batched --actors 1,4,8 --steps 4000`
   - This compares MuJoCo (N forked processes) vs Genesis batched (single process, GPU vectorized)
3. If no CUDA:
   - Run: `python benchmark_backends.py --backends mujoco --actors 1,2,4 --steps 2000`
   - Explain: "Genesis batched requires CUDA. Showing MuJoCo scaling only."
4. Present the results table and explain the speedup

Two benchmark scripts are available:
- `benchmark_backends.py` — MuJoCo vs Genesis SPS comparison at different parallelism levels
- `benchmark_physics.py` — MuJoCo physics preset comparison (timestep, iterations)
