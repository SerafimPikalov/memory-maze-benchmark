# Comprehensive Bottleneck Analysis

Every known and hypothesized performance bottleneck across all engine × mode × model × platform combinations, with clear separation between measured data and assumptions. Informs optimization priorities (Tasks 18, 19, 25) and hardware purchasing decisions.

**Last updated**: 2026-03-03

---

## 1. Configuration Matrix

| # | Engine | Mode | Renderer | Model | Platform | Primary Bottleneck | Status | Key Number |
|---|--------|------|----------|-------|----------|--------------------|--------|------------|
| 1a | MuJoCo | non-batched | GLFW (CPU OpenGL) | IMPALA | macOS | **~50/50 physics/rendering** | **MEASURED** | 5.90ms/step, ~170 SPS single-env |
| 1b | MuJoCo | non-batched | EGL (GPU OpenGL) | IMPALA | Linux+NVIDIA | **Physics (89% of step)** | **MEASURED** | 30.98ms/step, ~32 SPS single-env |
| 2a | Genesis | non-batched | Rasterizer (CPU OpenGL) | IMPALA | macOS | **Physics (67% of step)** | **MEASURED** | 24.31ms/step, ~41 SPS single-env |
| 2b | Genesis | non-batched | Rasterizer (EGL) | IMPALA | Linux+NVIDIA | **Physics (74% of step)** | **MEASURED** | 32.33ms/step, ~31 SPS single-env |
| 3 | Genesis | batched | BatchRenderer (Vulkan/CUDA) | IMPALA | Linux+NVIDIA | Pipeline bubble (99% of cycle) | **MEASURED** | 640 peak, ~118 effective |
| 4 | MuJoCo | non-batched | EGL | DreamerV2 | Linux+NVIDIA | World model GPU training (hypothesis) | **NOT MEASURED** | no data |
| 5 | Genesis | non-batched | Rasterizer/EGL | DreamerV2 | Linux+NVIDIA | World model GPU training (hypothesis) | **NOT MEASURED** | no data |
| 6 | Genesis | batched | BatchRenderer/CUDA | DreamerV2 | Linux+NVIDIA | Unknown | **NOT MEASURED** | no data |

**Platform note**: macOS uses CPU-only OpenGL (Apple deprecated GPU OpenGL). Linux EGL uses real GPU rasterization. This affects the rendering component of every non-batched configuration.

---

## 2. Per-Configuration Deep Dives

### Config 1a: MuJoCo × non-batched × IMPALA × macOS

**Source**: `benchmark_profiling.py` (monkey-patching `Physics.render` at class level)
**Hardware**: MacBook Pro M1 Pro, 10 cores, 16 GB RAM
**Date**: 2026-03-03 (corrected from 2026-02 initial measurements)

#### Pipeline Diagram

```
Per-step (5.90ms total):

  Physics (all non-render work)          Physics.render() calls
  ├── 2.89ms (49%) ──────────────────┤   ├── 3.01ms (51%) ────────────────────┤
  │  before_step + mj_step×50 +      │   │  mjv_updateScene + mjr_render +    │
  │  dm_control bookkeeping           │   │  mjr_readPixels                    │
  └───────────────────────────────────┘   └────────────────────────────────────┘
```

#### Measured Data (corrected 2026-03-03)

| Component | Time | % of step | Notes |
|-----------|------|-----------|-------|
| Physics (all non-render work) | **2.89ms** | **49%** | Includes mj_step×50, before_step, action application, dm_control overhead |
| Rendering (Physics.render calls) | **3.01ms** | **51%** | All render work including scene traversal, rasterization, readPixels |
| **Total** | **5.90ms** | | **~170 single-env steps/sec** |

Detailed stats: physics 2.89±0.16ms (p50=2.85, p95=3.19), render 3.01±0.50ms (p50=2.95, p95=3.42)

**Discrepancy with old measurements**: Old analysis reported 0.85ms physics (22%). The old approach likely only timed raw `mj_step()` calls, missing `before_step`, action application, and dm_control bookkeeping that the monkey-patch approach captures as physics time.

IMPALA training: 8 actors → **160 SPS** steady state.

#### What Helps / What Doesn't

| Optimization | Impact | Why |
|-------------|--------|-----|
| Physics presets (fewer substeps) | **+++** (up to 2.0×) | Physics is 49% — reducing it has real impact |
| More actors (>8) | **+** (linear to ~8 cores) | Each actor is a separate process, scales with CPU cores |
| Faster GPU | **---** | macOS has no GPU OpenGL; GPU is irrelevant |
| BatchRenderer | **N/A** | macOS not supported (CUDA + Linux required) |
| EGL | **N/A** | Not available on macOS |

---

### Config 1b: MuJoCo × non-batched × IMPALA × Linux+NVIDIA (RunPod)

**Source**: `benchmark_profiling.py` (single-env profiling, 2026-03-03), `tasks/done/09_genesis_batching_fix.md` (multi-actor SPS)
**Hardware**: RunPod RTX A4000, 16 GB VRAM; host CPU unknown (community cloud)
**Date**: 2026-03-03 (profiling), 2026-03-02 (SPS data)

#### Pipeline Diagram

```
Per-step (30.98ms total):

  Physics (all non-render work)                   Physics.render() calls
  ├── 27.58ms (89%) ─────────────────────────┤   ├── 3.41ms (11%) ────────┤
  │  before_step + mj_step×50 +              │   │  EGL GPU rasterization │
  │  dm_control bookkeeping                  │   │  + readPixels          │
  └──────────────────────────────────────────┘   └─────────────────────────┘
```

#### Measured Data

| Component | Time | % of step | Notes |
|-----------|------|-----------|-------|
| Physics (all non-render work) | **27.58ms** | **89%** | 9.5× slower than macOS M1 Pro (2.89ms) |
| Rendering (Physics.render calls) | **3.41ms** | **11%** | Similar to macOS (3.01ms) |
| **Total** | **30.98ms** | | **~32 single-env steps/sec** |

Detailed stats: physics 27.58±0.73ms (p50=27.45, p95=27.80), render 3.41±0.32ms (p50=3.39, p95=3.61)

Multi-actor SPS (from Task 09 data, RTX 2000 Ada):

| Metric | Value | Source |
|--------|-------|--------|
| 8 actors SPS | 160, steady | `tasks/done/09_genesis_batching_fix.md` |
| 12 actors SPS | 80, oscillating 160→0 | Same source (CPU contention) |
| GPU utilization | 0% | Same source |

**Key insight**: Physics is **the** bottleneck on RunPod. The host CPU on community cloud GPUs is dramatically slower than M1 Pro for MuJoCo physics (9.5×). Rendering is almost identical across platforms (~3ms). This explains why multi-actor SPS (160) is similar to macOS despite per-env step being 5× slower — multi-actor parallelism compensates.

#### What Helps / What Doesn't

| Optimization | Impact | Why |
|-------------|--------|-----|
| More/faster CPU cores | **+++** | Physics is 89% of step; CPU speed directly determines throughput |
| Physics presets | **+++** | Physics dominates — reducing substeps has huge impact on RunPod |
| Faster GPU | **---** | GPU is idle at 0%; rendering is only 11% |
| BatchRenderer | **N/A** | MuJoCo has no BatchRenderer path |

---

### Config 2a: Genesis × non-batched × IMPALA × macOS

**Source**: `benchmark_profiling.py` (monkey-patching `scene.step` and `scene.render_egocentric`)
**Hardware**: MacBook Pro M1 Pro (same as Config 1a)
**Date**: 2026-03-03

#### Pipeline Diagram

```
Per-step (24.31ms total):

  scene.step() (Taichi physics)    scene.render_egocentric()    Overhead
  ├── 16.30ms (67%) ────────────┤  ├── 7.67ms (32%) ─────────┤  ├── 0.34ms (1%) ┤
  │  PHYSICS (Taichi CPU)        │  │  Rasterizer + readPixels │  │  target checks │
  └──────────────────────────────┘  └──────────────────────────┘  └────────────────┘
```

#### Measured Data

| Component | Time | % of step | Notes |
|-----------|------|-----------|-------|
| Physics (`scene.step()`) | **16.30ms** | **67%** | Taichi CPU — 5.6× slower than MuJoCo (2.89ms) |
| Rendering (`render_egocentric()`) | **7.67ms** | **32%** | 2.5× slower than MuJoCo rendering (3.01ms) |
| Overhead (target checks, etc.) | 0.34ms | 1% | |
| **Total** | **24.31ms** | | **~41 single-env steps/sec** |

Detailed stats: physics 16.30±0.63ms (p50=16.24, p95=17.48), render 7.67±0.49ms (p50=7.67, p95=8.32)

**H1 REFUTED**: The old claim of "Genesis 2.6× speedup over MuJoCo" is contradicted. Genesis is actually **4.1× slower** than MuJoCo on macOS (24.31ms vs 5.90ms). Genesis physics is 5.6× slower, Genesis rendering is 2.5× slower. The old "64 vs 25 SPS" comparison (from `genesis_batching_problem.md`) likely measured under different conditions or code versions.

#### What Helps / What Doesn't

| Optimization | Impact | Why |
|-------------|--------|-----|
| Physics presets | **++** | Physics is 67% of step — reducing it helps significantly |
| BatchRenderer | **N/A** | Not available on macOS |
| More actors | **+** | Same multi-process scaling as MuJoCo |

---

### Config 2b: Genesis × non-batched × IMPALA × Linux+NVIDIA (RunPod)

**Source**: `benchmark_profiling.py` (single-env profiling, 2026-03-03), `tasks/done/09_genesis_batching_fix.md` (multi-actor SPS)
**Hardware**: RunPod RTX A4000, 16 GB VRAM; host CPU unknown (community cloud)
**Date**: 2026-03-03 (profiling), 2026-03-02 (SPS data)

#### Pipeline Diagram

```
Per-step (32.33ms total):

  scene.step() (Taichi physics)    scene.render_egocentric()    Overhead
  ├── 24.03ms (74%) ────────────┤  ├── 7.97ms (25%) ─────────┤  ├── 0.33ms (1%) ┤
  │  PHYSICS (Taichi CPU)        │  │  Rasterizer + readPixels │  │  target checks │
  └──────────────────────────────┘  └──────────────────────────┘  └────────────────┘
```

#### Measured Data

| Component | Time | % of step | Notes |
|-----------|------|-----------|-------|
| Physics (`scene.step()`) | **24.03ms** | **74%** | 1.5× slower than macOS (16.30ms), but close to MuJoCo RunPod (27.58ms) |
| Rendering (`render_egocentric()`) | **7.97ms** | **25%** | Similar to macOS (7.67ms) |
| Overhead (target checks, etc.) | 0.33ms | 1% | |
| **Total** | **32.33ms** | | **~31 single-env steps/sec** |

Detailed stats: physics 24.03±0.57ms (p50=23.98, p95=24.15), render 7.97±0.14ms (p50=7.96, p95=8.03)

Multi-actor SPS (from Task 09 data, RTX 2000 Ada):

| Metric | Value | Source |
|--------|-------|--------|
| 8 actors SPS | 160, steady | `tasks/done/09_genesis_batching_fix.md` |
| 12 actors SPS | 160, steady | Same source — **50% more headroom than MuJoCo's 80** |
| GPU utilization | 0% | Same source |

**Key insight**: Genesis and MuJoCo are nearly identical on RunPod (32.33ms vs 30.98ms). Genesis physics is actually slightly faster on RunPod (24.03ms vs 27.58ms), but Genesis rendering is slower (7.97ms vs 3.41ms). The 12-actor scalability advantage of Genesis likely comes from lower per-core CPU usage in Taichi vs MuJoCo.

#### What Helps / What Doesn't

| Optimization | Impact | Why |
|-------------|--------|-----|
| More/faster CPU cores | **+++** | Physics is 74% of step |
| Physics presets | **++** | Physics dominates |
| BatchRenderer (→Config 3) | **+++** | Eliminates glReadPixels entirely, moves physics to GPU |

---

### Config 3: Genesis × batched × IMPALA × Linux+NVIDIA (RunPod)

**Source**: `tasks/done/08_genesis_gpu_migration.md`, `tasks/done/09_genesis_batching_fix.md`
**Hardware**: RunPod RTX 2000 Ada, 16 vCPU, ~5.5 GB GPU used
**Date**: 2026-03-01 (Task 08), 2026-03-02 (Task 09)

#### Pipeline Diagram

```
Single batched actor, 32 envs, unroll length 80:

  ┌─── Produce phase (~13.8s) ──────────────────────────────┐  ┌─ Consume (~0.2s) ─┐
  │ step→model→step→model→...  ×80 steps ×32 envs           │  │ learner processes  │
  │                                                          │  │ all batches        │
  └──────────────────────────────────────────────────────────┘  └────────────────────┘
  ↑                                                                                  ↑
  640 SPS peak                                                                0 SPS (starving)

  Effective: ~118 SPS (14s cycle, data produced in bursts)
```

#### Measured Data — TWO DIFFERENT MEASUREMENTS

**Task 08 timing** (2026-03-01, post-BatchRenderer migration):

| Component | Time | % of cycle | Source |
|-----------|------|-----------|--------|
| Model inference | 59ms | 51% | `tasks/done/08_genesis_gpu_migration.md` |
| Step + render | 52ms | 45% | Same |
| Buffer writes | 5ms | 4% | Same |

**Task 09 timing** (2026-03-02, post-mortem):

| Component | Time | % of cycle | Source |
|-----------|------|-----------|--------|
| Step (physics + render) | 165ms | 95.4% | `tasks/done/09_genesis_batching_fix.md` |
| Model inference | 7.6ms | 4.4% | Same |
| Buffer writes | 0.4ms | 0.2% | Same |

#### DISCREPANCY: Task 08 vs Task 09 Timing

These measurements are **wildly different** — model is 51% in one and 4.4% in the other:

| Metric | Task 08 | Task 09 |
|--------|---------|---------|
| Model % | 51% | 4.4% |
| Step % | 45% | 95.4% |
| Model time | 59ms | 7.6ms |
| Step time | 52ms | 165ms |

**Most likely explanation**: Task 08 was measured *before* Task 13 (vectorize batched actor loop). The pre-vectorization Python loop had per-env overhead that inflated "model" time (sequential per-env model calls). After vectorization, model inference dropped from 59ms to 7.6ms (batched forward pass), and the step+render time increased from 52ms to 165ms — possibly because the measurement methodology changed, or because vectorization exposed that step time was always dominant but was previously masked by interleaved Python overhead.

**Status**: [UNRESOLVED] — need to re-instrument and measure again (→ Task 25)

#### Other Measured Data

| Metric | Value | Source |
|--------|-------|--------|
| Peak SPS | 640 | Both Task 08 and Task 09 |
| Effective SPS | ~118 | Task 09 post-mortem |
| GPU memory | ~5.5 GB | Task 08 |
| GPU utilization | 88% | Task 08 (BatchRenderer) |
| Transfer cost (GPU→CPU) | ~0.4ms/step (0.24%) | Task 19 analysis |
| SPS pattern | 640→0→0→0→640 | Task 09 (pipeline bubble) |
| Cycle time | ~14s (13.8s produce + 0.2s consume) | Task 09 |

#### Build Timing (2026-03-05, RTX A4000, n_envs=3)

The batched scene build takes **~4 minutes** total (not a hang):

| Phase | Duration | % of build |
|-------|----------|-----------|
| `_visualizer.build()` | **126.5s** | **52%** — OpenGL mesh setup for 230 entities × N envs |
| Kernel compilation (`_sim.step()`) | **69.0s** | **29%** — First Taichi CUDA compile (cacheable) |
| `RigidSolver.build()` | **19.4s** | **8%** — 227 geoms, 226 collision pairs |
| `_parallelize()` + `_reset()` | 0.4s | <1% |
| **Total build** | **215.5s** | |

After build: step=0.1s, reset=2.0s. GPU memory: 14.4 GB / 16 GB (A4000).

#### Env Scaling (2026-03-05, NVIDIA L40, 48GB, dt=0.05)

Genesis physics scales **linearly** with n_envs — more envs = proportionally longer collection:

| Envs | Batch size | Batch interval | Effective SPS | VRAM |
|------|-----------|----------------|---------------|------|
| 8 | 3200 | ~25s | ~128 | 18.6 GB |
| 32 | 3200 | ~12s | **~256** | 18.6 GB |
| 64 | 6400 | ~25s | ~256 | 18.6 GB |

Key findings: 32 envs is the sweet spot (2x over 8 envs). 64 envs produces 2x more data per rollout but takes 2x longer — no net gain. VRAM usage ~18.6 GB regardless of env count (shared kernels/meshes). RTX 3090 (24 GB, $0.22/hr) sufficient; L40 (48 GB, $0.69/hr) is overkill.

#### Physics Timestep Impact on Returns (2026-03-05, L40, 64 envs)

| dt | Substeps | Effective SPS | Return@64k steps |
|----|----------|---------------|-----------------|
| 0.05 | 5 | ~256 | 0.4 |
| 0.125 | 2 | ~427 | 0.2 (degraded) |

dt=0.125 gives 67% more SPS but **halves returns** — walker collision stability degrades with only 2 substeps. **Use dt=0.05 for training.**

#### Hypotheses [NOT MEASURED]

- **H6**: Within the 165ms "step", how much is Taichi physics vs Madrona rendering? Unknown — this is a key Task 25 measurement
- **H7**: ~~BatchRenderer rendering time scales sublinearly with env count~~ **REFUTED**: Physics scales linearly with n_envs, so total step time scales linearly. No sublinear benefit observed.

#### What Helps / What Doesn't

| Optimization | Impact | Why |
|-------------|--------|-----|
| Multiple batched actors (Task 18) | **+++** | Eliminates pipeline bubble; staggered actors = steady data flow |
| MPS (NVIDIA Multi-Process Service) | **++** | Enables concurrent GPU kernels from multiple processes (Task 18 Phase 1) |
| GPU-only frame pipeline (Task 19) | **-** | Transfer is 0.24% of step time; negligible |
| Physics presets (dt=0.05) | **++** | 10x physics speedup confirmed; dt=0.125 degrades learning |
| More envs (>32) | **0** | Linear physics scaling cancels out more data; 32 is sweet spot |
| Faster GPU | **++** | 88% GPU utilization means GPU matters here (unlike non-batched) |
| Taichi kernel cache | **+** | Saves 69s on rebuild; persist `TI_CACHE_PATH` across restarts |

---

### Config 4: MuJoCo × non-batched × DreamerV2 × Linux+NVIDIA

**Source**: No benchmarks exist
**Hardware**: N/A
**Date**: N/A

#### Pipeline Diagram (Hypothetical)

```
DreamerV2 data collection (8 envs, sequential):

  ┌─── Collect phase ─────────────────────────┐  ┌─── Train phase (GPU-heavy) ──────────────────────┐
  │ 8 envs × N steps, MuJoCo + EGL rendering  │  │ World model: RSSM GRU(2048) + ConvEncoder/Decoder │
  │ Same pipeline as Config 1b                 │  │ Actor-Critic: 15-step imagination rollouts        │
  └────────────────────────────────────────────┘  │ Lambda-returns with slow critic EMA               │
                                                  └──────────────────────────────────────────────────┘
```

#### Architecture Context

- 8 parallel envs (not 128 like IMPALA) — less CPU pressure
- Replay buffer (10M steps) decouples collection from training
- World model training is GPU-heavy: RSSM GRU(2048), ConvEncoder (IMPALA ResNet), ConvDecoder (transposed convolutions), 15-step imagination rollouts
- Each training step processes sequences of length 48 from replay

#### Hypotheses [NOT MEASURED]

- **H8**: GPU-bound on world model training — GRU(2048) + Conv encoder/decoder + 15-step imagination is significantly more GPU work than IMPALA's single forward+backward pass
- **H9**: Fewer envs (8 vs 128) means less CPU pressure, but data collection is proportionally slower
- **H10**: A better GPU matters here (unlike IMPALA where GPU is idle)
- **H11**: Collection is ~8× IMPALA single-env rate, training is continuous from replay — may be compute-starved rather than data-starved

#### What Would Help (Hypothetical)

| Optimization | Expected Impact | Why |
|-------------|----------------|-----|
| Faster GPU | **+++** | World model training is the hypothesized bottleneck |
| More CPU cores | **+** | Only 8 envs, much less CPU pressure than IMPALA's 128 |
| Physics presets | **+** | Same rendering-dominated env step as Config 1b |
| Mixed precision (fp16) | **++** | Would halve GPU compute for world model |

---

### Config 5: Genesis × non-batched × DreamerV2 × Linux+NVIDIA

**Source**: No benchmarks exist
**Hardware**: N/A
**Date**: N/A

Same hypotheses as Config 4 but with Genesis CPU efficiency advantage for the environment step.

#### Differences from Config 4

- Genesis uses less CPU per step (Config 2b shows 50% more headroom at 12 actors)
- If DreamerV2 is GPU-bound on training (H8), the faster env step only helps if collection is the bottleneck
- With only 8 envs, data collection is relatively cheap — the Genesis advantage may be smaller than for IMPALA (where 128 actors magnify per-step savings)

#### What Would Help (Hypothetical)

Same as Config 4, plus:

| Optimization | Expected Impact | Why |
|-------------|----------------|-----|
| BatchRenderer (→Config 6) | **+** to **+++** | Depends on whether data collection or GPU training is the bottleneck |

---

### Config 6: Genesis × batched × DreamerV2 × Linux+NVIDIA

**Source**: No benchmarks exist. Completely unexplored territory.
**Hardware**: N/A
**Date**: N/A

#### Hypothetical Pipeline

```
Option A: Replay buffer decouples collection from training (good)

  Actor process:   [batched step+render] → [store to replay buffer] → repeat
  Learner process: [sample from replay] → [world model + actor-critic training] → repeat

  These run asynchronously — no pipeline bubble if replay is large enough

Option B: Sequential collect-then-train (bad)

  [collect N steps batched] → [train on collected data] → [collect again]

  Same pipeline bubble as Config 3, but potentially worse because DreamerV2
  trains for many gradient steps per collected step
```

#### Hypotheses [NOT MEASURED]

- **H12**: Replay buffer may eliminate the pipeline bubble that plagues Config 3 — collection and training are naturally decoupled
- **H13**: Alternatively, replay buffer may make the bubble *worse* — batched DreamerV2 collects data fast (like Config 3) but then trains for many steps per collected step, leaving the actor idle during extended training phases
- **H14**: GPU contention between BatchRenderer rendering and DreamerV2 training — both want GPU compute. Config 3's IMPALA model is tiny (7.6ms); DreamerV2's world model is massive. Sharing a single GPU between BatchRenderer and world model training may be the new bottleneck.

---

## 3. Testable Hypotheses

Each hypothesis with its source, what would confirm/refute it, and which experiment tests it.

### H1: Genesis 2.6× speedup comes from physics, not rendering — **REFUTED**

- **Source**: `docs/research/genesis_batching_problem.md` — "The 2x speedup we measured comes from faster physics/scene management, not from rendering improvements"
- **Result (2026-03-03)**: **REFUTED**. Genesis is actually 4.1× *slower* than MuJoCo on macOS (24.31ms vs 5.90ms). Genesis physics is 5.6× slower (16.30ms vs 2.89ms), Genesis rendering is 2.5× slower (7.67ms vs 3.01ms). There is no speedup — the old "64 vs 25 SPS" claim cannot be reproduced and likely came from a different code version or measurement methodology.
- **On RunPod**: Genesis and MuJoCo are nearly equivalent (32.33ms vs 30.98ms). Genesis physics is slightly faster (24.03ms vs 27.58ms) but rendering is slower (7.97ms vs 3.41ms).

### H2: Linux EGL rendering is faster than macOS GLFW — **REFUTED**

- **Source**: `docs/rendering_bottleneck.md` — "Even with EGL on Linux, the readPixels GPU→CPU copy still dominates" (but no numbers)
- **Result (2026-03-03)**: **REFUTED** (but for unexpected reasons). MuJoCo rendering is nearly identical: macOS GLFW 3.01ms vs RunPod EGL 3.41ms. EGL is actually *slightly slower*, not faster. The real difference between platforms is physics — 2.89ms (macOS M1 Pro) vs 27.58ms (RunPod host CPU) — a 9.5× gap due to CPU quality, not rendering backend.

### H3: In batched 165ms step, physics dominates (not rendering)

- **Source**: Assumption based on BatchRenderer being "fast" and Taichi simulating 32 envs sequentially
- **Current evidence**: None — the 165ms is opaque
- **Confirm**: Profile batched step on RunPod, separate `scene.step()` from `camera.render()`. If physics >80%, confirmed.
- **Refute**: If rendering is >50%, BatchRenderer may need optimization or the assumption that "rendering is solved" is wrong.
- **Experiment**: Task 25, Config 6 (Genesis BatchRenderer, 32 envs)

### H4: DreamerV2 is GPU-bound on world model training

- **Source**: Architecture analysis — GRU(2048) + ConvEncoder + ConvDecoder + 15-step imagination is heavy
- **Current evidence**: No DreamerV2 benchmarks exist
- **Confirm**: Run DreamerV2 on MuJoCo, profile GPU utilization. If GPU >50% during training phases, confirmed.
- **Refute**: If GPU utilization is low (like IMPALA), DreamerV2 training is also CPU/data-bound.
- **Experiment**: New benchmark (not yet in task list)

### H5: Task 08 vs Task 09 timing discrepancy — which is correct?

- **Source**: Task 08 (model 51%, step 45%) vs Task 09 (step 95.4%, model 4.4%)
- **Current evidence**: Both measured on same hardware, different dates. Likely Task 08 was pre-vectorization.
- **Confirm**: Re-run timing on current code (post-vectorization). If model is ~4% and step is ~95%, Task 09 is correct and Task 08 was stale.
- **Refute**: If model is ~50%, something else changed.
- **Experiment**: Task 25, Config 6 (re-instrument batched actor timing)

### H6: BatchRenderer rendering scales sublinearly with env count

- **Source**: Madrona architecture — renders all envs in one GPU kernel
- **Current evidence**: None (only 32-env measurement exists)
- **Confirm**: Profile BatchRenderer at 1, 8, 16, 32, 64 envs. If per-env rendering time decreases, confirmed.
- **Refute**: If per-env time is constant (linear scaling), Madrona doesn't amortize well at these counts.
- **Experiment**: Task 25, compare Config 5 (1 env) vs Config 6 (32 envs)

### H7: MPS gives >1× speedup for multi-process BatchRenderer

- **Source**: NVIDIA MPS enables concurrent kernel execution from multiple processes
- **Current evidence**: Task 18 showed two processes time-slicing (3+3=6 vs single 6), no MPS tested
- **Confirm**: Enable MPS, run dual BatchRenderer. If combined SPS > single-process SPS, confirmed.
- **Refute**: If GPU memory bandwidth is the bottleneck (not compute), MPS won't help.
- **Experiment**: Task 18, Phase 1

### H8: GPU contention between BatchRenderer and DreamerV2 training

- **Source**: Both want GPU compute on the same device
- **Current evidence**: No DreamerV2 + BatchRenderer combination has been tested
- **Confirm**: Run DreamerV2 batched, monitor GPU SM utilization during collection vs training phases. If both phases use >50%, contention exists.
- **Refute**: If collection and training are temporally separated (replay buffer), no contention.
- **Experiment**: Config 6 benchmark (not yet in task list)

---

## 4. Optimization Impact Matrix

Impact of each optimization across all configurations. Scale: **+++** (transformative), **++** (significant), **+** (modest), **0** (no effect), **-** (minimal), **N/A** (not applicable).

| Optimization | 1a MuJoCo macOS | 1b MuJoCo Linux | 2a Genesis macOS | 2b Genesis Linux | 3 Genesis batched | 4 MuJoCo DreamerV2 | 5 Genesis DreamerV2 | 6 Genesis batched DreamerV2 |
|-------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| More/faster CPU cores | ++ | **+++** | ++ | **+++** | 0 | + | + | 0 |
| Faster GPU | --- | --- | --- | --- | ++ | +++ | +++ | +++ |
| Physics presets | **+++** | **+++** | **++** | **++** | ? | + | + | ? |
| BatchRenderer | N/A | N/A | N/A | +++¹ | baseline | N/A | +++¹ | baseline |
| Multi-actor (Task 18) | N/A | N/A | N/A | N/A | +++ | N/A | N/A | ? |
| GPU-only pipeline (Task 19) | N/A | N/A | N/A | N/A | - | N/A | N/A | +² |
| MPS | N/A | N/A | N/A | N/A | ++ | N/A | N/A | + |
| Mixed precision (fp16) | 0 | 0 | 0 | 0 | 0 | ++ | ++ | ++ |
| More envs per actor | + | + | + | + | + | 0 | 0 | + |

**Notes**:
1. ¹ = Upgrading from non-batched to batched (i.e., switching from Rasterizer to BatchRenderer, transitioning to Config 3 or 6)
2. ² = For DreamerV2 with GPU-resident replay buffer, GPU-only pipeline avoids unnecessary GPU→CPU→GPU round-trip for frames going into replay. More impactful than for IMPALA where frames must cross process boundaries via CPU shared memory.

**Key insight (updated 2026-03-03)**: Physics dominates on all non-batched configs, especially on RunPod:
- **Non-batched on RunPod** (1b, 2b): **CPU speed is the bottleneck** — physics is 89% (MuJoCo) / 74% (Genesis). GPU is idle (0% utilization). Faster CPU cores or physics presets have the highest impact.
- **Non-batched on macOS** (1a, 2a): Physics is 49% (MuJoCo) / 67% (Genesis) — still significant, but more balanced.
- **Batched IMPALA** (3): Multi-actor processes (Task 18)
- **DreamerV2** (4, 5, 6): Faster GPU + mixed precision

---

## 5. Data Sources

Every measured number traced to its source file, platform, hardware, and date.

| Number | Value | Source File | Platform | Hardware | Date |
|--------|-------|------------|----------|----------|------|
| MuJoCo macOS physics | **2.89ms (49%)** | `benchmark_profiling.py` | macOS | M1 Pro | 2026-03-03 |
| MuJoCo macOS rendering | **3.01ms (51%)** | `benchmark_profiling.py` | macOS | M1 Pro | 2026-03-03 |
| MuJoCo macOS total step | **5.90ms** | `benchmark_profiling.py` | macOS | M1 Pro | 2026-03-03 |
| MuJoCo RunPod physics | **27.58ms (89%)** | `benchmark_profiling.py` | Linux EGL | RTX A4000 | 2026-03-03 |
| MuJoCo RunPod rendering | **3.41ms (11%)** | `benchmark_profiling.py` | Linux EGL | RTX A4000 | 2026-03-03 |
| MuJoCo RunPod total step | **30.98ms** | `benchmark_profiling.py` | Linux EGL | RTX A4000 | 2026-03-03 |
| Genesis macOS physics | **16.30ms (67%)** | `benchmark_profiling.py` | macOS | M1 Pro | 2026-03-03 |
| Genesis macOS rendering | **7.67ms (32%)** | `benchmark_profiling.py` | macOS | M1 Pro | 2026-03-03 |
| Genesis macOS total step | **24.31ms** | `benchmark_profiling.py` | macOS | M1 Pro | 2026-03-03 |
| Genesis RunPod physics | **24.03ms (74%)** | `benchmark_profiling.py` | Linux EGL | RTX A4000 | 2026-03-03 |
| Genesis RunPod rendering | **7.97ms (25%)** | `benchmark_profiling.py` | Linux EGL | RTX A4000 | 2026-03-03 |
| Genesis RunPod total step | **32.33ms** | `benchmark_profiling.py` | Linux EGL | RTX A4000 | 2026-03-03 |
| ~~MuJoCo physics time (OLD)~~ | ~~0.85ms (22%)~~ | ~~`docs/rendering_bottleneck.md`~~ | macOS | M1 Pro | 2026-02 |
| ~~MuJoCo rendering time (OLD)~~ | ~~2.95ms (78%)~~ | ~~`docs/rendering_bottleneck.md`~~ | macOS | M1 Pro | 2026-02 |
| Physics presets (4 tiers) | 0.06–0.85ms physics, 3.13–6.11ms total | `docs/rendering_bottleneck.md` | macOS | M1 Pro | 2026-02 |
| IMPALA macOS 8a SPS | 160 steady | `docs/findings.md` | macOS | M1 Pro | 2026-02 |
| MuJoCo RunPod 8a SPS | 160 steady | `tasks/done/09_genesis_batching_fix.md` | Linux | RTX 2000 Ada, 16 vCPU | 2026-03-02 |
| MuJoCo RunPod 12a SPS | 80, oscillating | `tasks/done/09_genesis_batching_fix.md` | Linux | RTX 2000 Ada, 16 vCPU | 2026-03-02 |
| MuJoCo RunPod GPU util | 0%, 130 MiB | `tasks/done/09_genesis_batching_fix.md` | Linux | RTX 2000 Ada, 16 vCPU | 2026-03-02 |
| Learner capacity | ~12,800 SPS | `tasks/done/09_genesis_batching_fix.md` | Linux | RTX 2000 Ada, 16 vCPU | 2026-03-02 |
| Genesis macOS single-env | 64 steps/sec | `docs/research/genesis_batching_problem.md` | macOS | M1 Pro | 2026-02 |
| MuJoCo macOS single-env | 25 steps/sec | `docs/research/genesis_batching_problem.md` | macOS | M1 Pro | 2026-02 |
| Genesis RunPod 8a SPS | 160 steady | `tasks/done/09_genesis_batching_fix.md` | Linux | RTX 2000 Ada, 16 vCPU | 2026-03-02 |
| Genesis RunPod 12a SPS | 160 steady | `tasks/done/09_genesis_batching_fix.md` | Linux | RTX 2000 Ada, 16 vCPU | 2026-03-02 |
| Batched peak SPS | 640 | `tasks/done/08_genesis_gpu_migration.md` | Linux | RTX 2000 Ada | 2026-03-01 |
| Batched effective SPS | ~118 | `tasks/done/09_genesis_batching_fix.md` | Linux | RTX 2000 Ada, 16 vCPU | 2026-03-02 |
| Batched GPU memory | ~5.5 GB | `tasks/done/08_genesis_gpu_migration.md` | Linux | RTX 2000 Ada | 2026-03-01 |
| Batched GPU utilization | 88% | `tasks/done/08_genesis_gpu_migration.md` | Linux | RTX 2000 Ada | 2026-03-01 |
| Task 08 timing: model | 59ms (51%) | `tasks/done/08_genesis_gpu_migration.md` | Linux | RTX 2000 Ada | 2026-03-01 |
| Task 08 timing: step+render | 52ms (45%) | `tasks/done/08_genesis_gpu_migration.md` | Linux | RTX 2000 Ada | 2026-03-01 |
| Task 08 timing: write | 5ms (4%) | `tasks/done/08_genesis_gpu_migration.md` | Linux | RTX 2000 Ada | 2026-03-01 |
| Task 09 timing: step | 165ms (95.4%) | `tasks/done/09_genesis_batching_fix.md` | Linux | RTX 2000 Ada, 16 vCPU | 2026-03-02 |
| Task 09 timing: model | 7.6ms (4.4%) | `tasks/done/09_genesis_batching_fix.md` | Linux | RTX 2000 Ada, 16 vCPU | 2026-03-02 |
| Task 09 timing: write | 0.4ms (0.2%) | `tasks/done/09_genesis_batching_fix.md` | Linux | RTX 2000 Ada, 16 vCPU | 2026-03-02 |
| Transfer cost (GPU→CPU) | ~0.4ms/step (0.24%) | `tasks/19_gpu_only_frame_pipeline.md` | Linux | RTX 2000 Ada | 2026-03-03 |
| Pipeline bubble cycle | ~14s (13.8s produce + 0.2s consume) | `tasks/done/09_genesis_batching_fix.md` | Linux | RTX 2000 Ada, 16 vCPU | 2026-03-02 |
| Multi-process BatchRenderer | 3+3=6 (time-sliced) vs 6 single | `tasks/18_multi_process_batchrenderer.md` | Linux | RTX 2000 Ada | 2026-03-02 |

---

## 6. Cross-Reference: Task 25 Profiling Plan

Task 25 (`tasks/25_physics_vs_rendering_profiling.md`) is designed to fill measurement gaps. **Progress as of 2026-03-03**:

| Config | Previous Status | Current Status | Notes |
|--------|---------------|----------------|-------|
| 1a (MuJoCo macOS) | MEASURED (old, wrong split) | **MEASURED (corrected)** | 49/51 not 22/78 |
| 1b (MuJoCo Linux EGL) | PARTIAL (SPS only) | **MEASURED** | 89/11 physics/render |
| 2a (Genesis macOS) | PARTIAL (SPS only) | **MEASURED** | 67/32 physics/render |
| 2b (Genesis Linux) | PARTIAL (SPS only) | **MEASURED** | 74/25 physics/render |
| 3 (Genesis batched) | MEASURED (SPS + cycle), but 165ms opaque | **PARTIAL** | `BatchedGenesisMazeEnv` not in Docker image |

Hypotheses tested: **H1 REFUTED** (Genesis not faster), **H2 REFUTED** (EGL not faster rendering).

Remaining: H3 (batched 165ms split — needs Docker image fix), H4 (DreamerV2 GPU), H5 (partially confirmed — Genesis scales better at 12 actors due to lower CPU pressure), H6 (BatchRenderer env scaling), H7 (MPS), H8 (GPU contention).
