# Physics Timestep Optimization for Genesis Backend

## Summary

Genesis physics step time scales **linearly** with the number of substeps per control step.
Increasing the physics timestep (reducing substeps) provides proportional speedup with stable walker dynamics.

## Background

Memory Maze uses:
- Control timestep: 0.25s (4 Hz agent decision rate, matches the paper)
- Physics timestep: 0.005s (200 Hz, inherited from MuJoCo default)
- Substeps per control step: 0.25 / 0.005 = **50**

Each substep launches Taichi kernels for forward kinematics, collision detection, and velocity integration.
With 225 wall entities the collision solver dominates.

## Profiling Results (RTX A4000, 3 envs)

| physics_dt | substeps | physics (ms) | total (ms) | speedup |
|------------|----------|-------------|-----------|---------|
| 0.005      | 50       | 460         | 466       | 1.0x    |
| 0.01       | 25       | 253         | 258       | 1.8x    |
| 0.025      | 10       | 99          | 104       | 4.5x    |
| 0.05       | 5        | 45          | 49        | 9.5x    |

Per-substep cost is constant at ~9.2ms regardless of dt.
Render time is constant at ~3.5ms regardless of substep count.

## Walker Stability

Tested at multiple timesteps during actual IMPALA training (2026-03-05, L40, 64 envs):

| dt | Substeps | Stable? | Return@64k steps | Notes |
|----|----------|---------|-----------------|-------|
| 0.005 | 50 | Yes | baseline | Default, too slow |
| 0.05 | 5 | **Yes** | 0.4 | Recommended |
| 0.125 | 2 | **Degraded** | 0.2 | 50% lower returns — collision instability |

dt=0.05 is the safe maximum. dt=0.125 causes the collision solver to miss contacts with
only 2 substeps per control step, resulting in worse navigation and halved episode returns.

## Recommendation

Use `physics_timestep=0.05` (5 substeps) for training. This gives ~10x physics speedup
with no behavior change. **Do NOT use dt=0.125** — it degrades learning. The parameter is exposed via:

- `BatchGenesisMemoryMazeEnv(physics_timestep=0.05)`
- `python train_impala.py --backend genesis --batched --physics_timestep 0.05`
- `python benchmark_profiling.py --backend genesis-batched --detailed --physics_dt 0.05`

## Full Overhead Breakdown (dt=0.005 baseline, 3 envs)

```
Phase                Mean      % Total
physics            460.54ms    98.9%
  apply_actions      0.24ms     0.1%
  substeps         460.29ms    98.8%
render               3.78ms     0.8%
camera_update        0.97ms     0.2%
target_checks        0.15ms     0.0%
heading_update       0.14ms     0.0%
action_mapping       0.02ms     0.0%
border_draw          0.04ms     0.0%
auto_reset           0.02ms     0.0%
TOTAL              465.66ms   100.0%
```

Date: 2026-03-05, RunPod RTX A4000 16GB, Genesis 0.4.0
