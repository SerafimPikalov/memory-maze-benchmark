# Engine Comparison Report: Porting Memory Maze to GPU-Accelerated Backends

**Date:** February 27, 2026
**Purpose:** Evaluate Genesis, MuJoCo/MJX, and Madrona as candidate engines for porting the Memory Maze RL benchmark to enable large-scale, GPU-accelerated neuroscience-inspired AI research.

---

## 1. Executive Summary

Memory Maze is a 3D maze navigation benchmark that tests long-term memory in RL agents — a capability central to neuroscience-inspired AI research. Its current implementation on dm_control + MuJoCo is CPU-bound, limiting experiment throughput. We evaluated three GPU-accelerated engines as replacement backends.

| Engine | Verdict | Best For | Porting Effort |
|--------|---------|----------|----------------|
| **MuJoCo/MJX (Warp)** | **Recommended first** | Lowest risk, same physics, GPU rendering via Warp backend | 3-6 weeks |
| **Genesis** | Strong alternative | Python-native workflow, good batch sim, Madrona-based rendering | 2-4 weeks |
| **Madrona (via madrona_mjx)** | Highest ceiling | Maximum throughput when combined with MJX physics | 2-4 weeks (hybrid) |
| **Madrona (native C++)** | Highest effort | Full control, max performance, but requires complete rewrite | 4-8 weeks |

**Bottom line:** The **MJX Warp backend** is the safest starting point — same physics engine, proven batch simulation, and new GPU rendering. **Genesis** is the best Python-first alternative with the most ergonomic developer experience. The **madrona_mjx hybrid** (MJX physics + Madrona batch rendering) offers the highest throughput ceiling if rendering is the bottleneck.

---

## 2. What Memory Maze Needs From an Engine

The reference implementation analysis identified these concrete requirements:

### 2.1 Must-Have Features

| Requirement | Detail | Difficulty to Replicate |
|------------|--------|------------------------|
| **Rolling ball physics** | Sphere (r=0.2m) with hinge joints for roll/steer, high damping (5.0/20.0), slide joints for root movement | Medium — physics fidelity matters |
| **First-person RGB rendering** | 64x64 or 256x256, 80-degree FOV, camera attached to walker at 0.3m height | High — must be GPU-batched for throughput |
| **Contact-based target detection** | MuJoCo `gap` parameter enables proximity activation at distance; checks `physics.data.contact` each substep | Medium — needs proximity equivalent |
| **Procedural maze regeneration** | New maze layout every episode via labmaze; MJCF recompiled each reset | High — biggest porting challenge universally |
| **Textured walls and floors** | DMLab texture packs, per-region wall textures, floor variations | Medium — visual diversity matters for RL transfer |
| **Collision filtering** | Visual-only geoms (contype=0) for textured planes; collision-only geoms for walls | Low-Medium |
| **Deterministic simulation** | Reproducible given same seed | Low — most engines support this |

### 2.2 Scene Complexity

- ~100-300 geoms per maze (walls + floor tiles + targets + walker parts)
- 3-6 target spheres per maze (varies by size)
- Physics: 200Hz internal, 4Hz control = 50 substeps per action
- Episode length: 1,000-4,000 control steps (50K-200K physics steps)
- 6 discrete actions mapped from 2D continuous [-1, 1]

### 2.3 The Core Bottleneck

The reference implementation's performance bottleneck is **per-episode MJCF recompilation** (rebuilding the scene graph for each new maze layout) and **CPU-bound single-environment execution**. Any GPU port must address both:
1. **Batch simulation** — running thousands of environments in parallel
2. **GPU rendering** — generating first-person RGB observations without CPU roundtrips
3. **Efficient maze variation** — avoiding full scene rebuild every episode

### 2.4 Reusable Components (Engine-Independent)

These components can be reused directly regardless of engine choice:
- `labmaze` C++ library for maze layout generation
- Observation wrappers (border drawing, coordinate transforms, minimap overlay — all numpy)
- Reward logic (target cycling, sparse +1.0)
- Action discretization mapping

---

## 3. Engine-by-Engine Analysis

### 3.1 Genesis

**What it is:** Python-first universal physics engine with a Taichi-derived GPU backend (Quadrants), supporting rigid body dynamics, multiple solver types, and three rendering backends including a Madrona-based batch renderer.

#### Architecture Highlights
- `gs.init() -> Scene -> Simulator -> Solvers -> Entities`
- Seven physics solvers; only RigidSolver needed for Memory Maze
- MJCF loading via MuJoCo's own parser (`mujoco.MjModel.from_xml_string`)
- Pydantic-based configuration system
- `scene.build(n_envs=N)` for native batch simulation

#### Strengths

| Strength | Impact |
|----------|--------|
| **Native batch simulation** | `scene.build(n_envs=4096)` creates thousands of parallel environments with GPU-parallelized physics |
| **Madrona-based BatchRenderer** | GPU-batched first-person RGB rendering at scale (exactly what MM needs) |
| **MJCF parsing via MuJoCo** | Robust, well-tested MJCF support; understands the MuJoCo data model |
| **PyTorch-native tensors** | All state on GPU as PyTorch tensors; zero-copy to RL training |
| **Camera attachment API** | `camera.attach(rigid_link, offset_T)` for egocentric views |
| **Per-environment reset** | `scene.reset(envs_idx=[0,3,7])` for async episode management |
| **MuJoCo compatibility mode** | `enable_mujoco_compatibility=True` for physics validation against reference |
| **Python-first workflow** | Lowest barrier to entry; familiar ecosystem |
| **Procedural scene construction** | `gs.morphs.Box`, `gs.morphs.Sphere` for programmatic geometry — mirrors dm_control pattern |

#### Weaknesses

| Weakness | Severity | Mitigation |
|----------|----------|------------|
| **No dynamic scene rebuild after `build()`** | HIGH | Pre-generate maze pool; assign different layouts to parallel envs; reposition targets via `set_pos()` |
| **BatchRenderer is CUDA + Linux x86-64 only** | MEDIUM | Use Rasterizer on other platforms for dev; deploy on Linux for training |
| **No dm_control/labmaze integration** | MEDIUM | Keep labmaze as dependency; write adapter to map layouts to Genesis entities |
| **Single-entity MJCF assumption** | MEDIUM | Use separate `add_entity()` calls for walker, walls, targets |
| **No built-in Gymnasium wrapper** | LOW | Write custom wrapper (straightforward, ~100 lines) |
| **Research-grade maturity** | LOW-MEDIUM | Active development; community growing |
| **Kernel compilation overhead** | LOW | One-time cost at first `build()`; cached subsequently |

#### Porting Approach
1. Keep `labmaze` for maze generation
2. Pre-generate pool of 100-1,000 maze layouts at startup
3. Build scene with one layout per parallel environment
4. Walker: `gs.morphs.Sphere` with free joint + velocity actuators
5. Walls: `gs.morphs.Box` (fixed per layout)
6. Targets: `gs.morphs.Sphere` (reposition via `set_pos()` at episode reset)
7. Attach egocentric camera to walker link
8. Use BatchRenderer on Linux/CUDA, Rasterizer elsewhere
9. Target detection: distance-based check in PyTorch (replaces MuJoCo `gap`)

**Estimated effort: 2-4 weeks** for working port + 1-2 weeks optimization/validation.

---

### 3.2 MuJoCo / MJX

**What it is:** MJX is MuJoCo's JAX-based GPU re-implementation. It offers batch simulation via `jax.vmap` with four backends: JAX (pure JAX physics), Warp (NVIDIA GPU physics + rendering), C, and CPP.

#### Architecture Highlights
- Same physics as MuJoCo — numerically equivalent
- `mjx.put_model(mj_model)` transfers model to GPU as JAX PyTree
- `jax.vmap(mjx.step, in_axes=(None, 0))` for trivial batch parallelism
- Warp backend adds GPU rendering (CUDA only, `warp-lang==1.11.1`)
- Full JAX ecosystem compatibility: `jit`, `vmap`, `grad`, `pmap`

#### Strengths

| Strength | Impact |
|----------|--------|
| **Same physics engine** | Memory Maze already uses MuJoCo; zero physics fidelity concerns |
| **MJCF files work directly** | No model conversion needed |
| **Proven batch simulation** | Tutorial demonstrates 4096-8192 parallel envs; Brax integration for full RL pipelines |
| **Warp GPU rendering** | New (2026) GPU rendering path with batch support: `create_render_context(nworld=N)` |
| **JAX ecosystem** | Compatible with Brax, PureJaxRL, Gymnax; `jax.grad` for differentiable physics |
| **Mature, DeepMind-backed** | Active development by Google DeepMind; production-grade |
| **Policy transfer guaranteed** | Policies trained in MJX run in standard MuJoCo and vice versa |

#### Weaknesses

| Weakness | Severity | Mitigation |
|----------|----------|------------|
| **JAX backend has NO rendering** | CRITICAL | Must use Warp backend for GPU rendering, or CPU fallback (kills perf) |
| **Warp backend is NVIDIA-only** | HIGH | Hard constraint; no AMD/Apple Silicon GPU rendering |
| **Warp rendering is new (2026)** | MEDIUM | May have rough edges; `nworld` fixed at context creation |
| **No dm_control compatibility** | HIGH | Entire composer/task/observable stack must be rewritten as JAX functions |
| **Static model structure** | HIGH | `mjx.Model` cannot change structure between steps; maze regeneration needs workarounds |
| **No built-in environment framework** | MEDIUM | Must follow Brax `PipelineEnv` pattern or write custom JAX env |

#### Rendering Path Options

| Approach | GPU Physics | GPU Rendering | Batched | Platform |
|----------|-------------|---------------|---------|----------|
| MJX JAX + CPU render | Yes | No (CPU fallback) | Physics only | Any |
| **MJX Warp** | Yes | **Yes** | **Both** | **NVIDIA only** |
| MJX JAX + Madrona render | Yes | Yes (external) | Both | NVIDIA + Linux |

#### Porting Approach
1. Extract MJCF XML from dm_control's model builder (serialize the composed model)
2. Pre-generate maze layouts; create one `MjModel` per layout variant
3. Rewrite environment as Brax `PipelineEnv` or standalone JAX env
4. Use Warp backend with `create_render_context(nworld=batch_size)` for GPU rendering
5. Target detection: distance check in JAX (replace MuJoCo `gap`)
6. Handle maze variation via pre-compiled model pool or geometry reconfiguration via `Data` fields

**Estimated effort: 3-6 weeks** (Warp path); 2-4 weeks (JAX-only, CPU rendering — but impractical for vision RL).

---

### 3.3 Madrona

**What it is:** A GPU-accelerated batch simulation game engine from Stanford/CMU (SIGGRAPH 2023) using ECS architecture and CUDA megakernel compilation. Not a standalone simulator — it's a framework for building custom simulators in C++.

#### Architecture Highlights
- Entity Component System (ECS) with compile-time archetype registration
- TaskGraph execution model: DAG of parallel-for operations over ECS data
- Multi-world batch execution: `numWorlds` independent instances on one GPU
- Built-in high-throughput batch renderer (Vulkan-based, SIGGRAPH Asia 2024)
- PyTorch/JAX tensor export via DLPack (zero-copy)
- `madrona_mjx`: external project combining MJX physics + Madrona rendering

#### Strengths

| Strength | Impact |
|----------|--------|
| **Extreme throughput** | Purpose-built for thousands of parallel environments; millions of steps/sec |
| **Best-in-class batch renderer** | SIGGRAPH Asia 2024 paper; generates first-person RGBD at scale across thousands of worlds |
| **Zero-copy GPU tensors** | Direct PyTorch/JAX export via DLPack; no CPU-GPU transfer |
| **JAX XLA custom call integration** | Production-grade; supports PBT, checkpointing |
| **madrona_mjx precedent** | Proven hybrid: MJX physics + Madrona rendering — directly applicable to Memory Maze |
| **CPU/GPU code portability** | Same C++ code runs on both; CPU for debugging, GPU for training |
| **Navmesh support** | BFS/Dijkstra pathfinding built in — useful for oracle/evaluation |

#### Weaknesses

| Weakness | Severity | Mitigation |
|----------|----------|------------|
| **C++ development required** (native path) | HIGH | Fork existing simulator; or use madrona_mjx hybrid |
| **No MJCF loading** | HIGH | Must recreate geometry manually or use MJX for physics |
| **Limited physics** (XPBD only) | HIGH | Spheres, convex hulls, planes only; no articulated bodies or actuators |
| **Research codebase** | MEDIUM | "Missing features / documentation / bugs"; breaking API changes |
| **Linux + NVIDIA GPU required** (GPU backend) | MEDIUM | macOS CPU backend for debugging only |
| **Fixed archetype schema** | LOW | Over-allocate for max targets; manageable |
| **No dm_control integration** | HIGH | All environment logic rewritten |

#### Three Porting Paths

| Path | Physics | Rendering | Effort | Throughput |
|------|---------|-----------|--------|------------|
| **A: Full native C++** | Madrona XPBD | Madrona batch | 4-8 weeks | Maximum |
| **B: madrona_mjx hybrid** | MJX (JAX) | Madrona batch | 2-4 weeks | High (rendering-accelerated) |
| **C: Rendering-only** | Python/JAX | Madrona batch | 1-2 weeks | Rendering gain only |

**Recommended path: B (madrona_mjx hybrid)** — preserves MuJoCo physics fidelity, gains Madrona's rendering throughput, avoids full C++ rewrite.

---

## 4. Comparative Matrix

### 4.1 Feature Comparison

| Feature | Memory Maze Needs | Genesis | MJX (Warp) | Madrona (native) | madrona_mjx |
|---------|-------------------|---------|------------|-------------------|-------------|
| **Batch simulation** | Thousands of parallel envs | `n_envs=N` | `jax.vmap` | `numWorlds=N` | MJX vmap + Madrona worlds |
| **GPU physics** | Rolling ball + contacts | Quadrants GPU | JAX/Warp GPU | XPBD GPU | MJX GPU |
| **Physics fidelity** | Match MuJoCo behavior | MuJoCo compat mode | Identical (same engine) | Low (game-grade) | Identical (MJX) |
| **GPU batch rendering** | First-person RGB at scale | Madrona BatchRenderer | Warp renderer | Madrona batch renderer | Madrona batch renderer |
| **MJCF support** | Load existing models | Via MuJoCo parser | Native | None | Via MJX |
| **Camera attachment** | Egocentric view | `camera.attach()` | Camera in model | `RenderCamera` ECS | Via Madrona ECS |
| **Texture support** | Wall/floor textures | Yes (materials) | Yes (Warp) | Yes (materials) | Yes (Madrona) |
| **Per-env reset** | Async episode management | `reset(envs_idx=)` | JAX conditional | Per-world reset | Per-world reset |
| **Dynamic scene rebuild** | Maze regeneration per episode | No (pre-generate pool) | No (static model) | No (fixed archetypes) | No (static model) |
| **Python API** | Developer ergonomics | Native Python | JAX Python | nanobind wrapper | MJX Python + Madrona |
| **Gymnasium wrapper** | RL framework compat | Custom (easy) | Custom (Brax pattern) | Custom (C++/Python) | Custom |
| **Platform** | Dev + training | CUDA+Linux (batch), all (raster) | NVIDIA (Warp), all (JAX) | Linux+NVIDIA (GPU) | Linux+NVIDIA |
| **Differentiable physics** | Optional (future) | Yes | Yes (JAX grad) | No | Yes (MJX) |

### 4.2 Developer Experience

| Dimension | Genesis | MJX (Warp) | Madrona (native) | madrona_mjx |
|-----------|---------|------------|-------------------|-------------|
| **Primary language** | Python | Python/JAX | C++ | Python/JAX + C++ build |
| **Learning curve** | Low | Medium (JAX) | High (ECS + CUDA) | Medium-High |
| **Debugging** | Python debugger, all platforms | JAX debugging tools | CPU backend + viewer | Mixed |
| **Documentation** | Growing, examples | Good (MuJoCo docs) | Limited (research code) | Minimal |
| **Community** | Growing | Large (MuJoCo + JAX) | Small (academic) | Very small |
| **Iteration speed** | Fast (Python) | Medium (JIT compilation) | Slow (C++ rebuild) | Medium |

---

## 5. Performance Projections

### 5.1 Reference Baseline

The current Memory Maze implementation (CPU MuJoCo + dm_control):
- **~50-100 env steps/sec** per CPU core (including rendering)
- Typical setup: 8-16 parallel processes on a workstation
- **Total: ~400-1,600 env steps/sec**

### 5.2 Projected GPU Throughput

| Engine | Estimated Env Steps/Sec | Speedup vs Reference | Confidence | Notes |
|--------|------------------------|---------------------|------------|-------|
| **Genesis (BatchRenderer)** | 10,000-50,000 | 10-50x | Medium | Depends on maze complexity, rendering resolution |
| **MJX Warp** | 10,000-100,000 | 10-100x | Medium | Physics well-proven; rendering path newer |
| **Madrona (native)** | 50,000-500,000+ | 50-500x | Medium-High | Purpose-built for this; Overcooked trains in ~2 min |
| **madrona_mjx** | 20,000-200,000 | 20-200x | Medium | MJX physics throughput + Madrona rendering |

**Key caveat:** These are estimates based on comparable benchmarks, not direct measurements. Actual throughput depends on:
- GPU hardware (A100/H100 vs consumer GPUs)
- Rendering resolution (64x64 vs 256x256 — 16x pixel difference)
- Number of parallel environments fitting in GPU memory
- Overhead of maze geometry transfer between systems (for hybrid approaches)

### 5.3 Where the Time Goes

In vision-based RL on Memory Maze, the per-step cost breaks down roughly as:

| Component | CPU MuJoCo | GPU (any engine) |
|-----------|-----------|-------------------|
| Physics step (50 substeps) | ~40% | <5% (GPU-parallel) |
| RGB rendering (64x64) | ~50% | ~20-40% (batch-rendered) |
| Observation transfer | ~5% | <1% (zero-copy GPU) |
| Python overhead | ~5% | ~5-20% (varies) |
| **Total bottleneck** | **Physics + rendering** | **Rendering** |

This confirms that **GPU batch rendering is the critical capability** — and all three candidate engines provide it (Genesis via Madrona BatchRenderer, MJX via Warp, Madrona natively).

---

## 6. The Universal Challenge: Dynamic Maze Regeneration

Every candidate engine shares the same fundamental challenge: **Memory Maze regenerates its maze layout every episode**, but GPU-optimized engines require static scene structure for compiled/batched execution.

### 6.1 Why It Matters

- Memory Maze calls `initialize_episode_mjcf()` before each episode
- This rebuilds all wall geometry, floor textures, and target positions
- dm_control then recompiles the MJCF into a new MuJoCo model
- In GPU engines, scene structure (entity count, archetypes, compiled kernels) is fixed after initialization

### 6.2 Solution Strategies

| Strategy | Description | Trade-off |
|----------|-------------|-----------|
| **Pre-generated maze pool** | Generate N maze layouts at startup; assign one per parallel env; cycle through pool | Finite variety (but 1,000+ layouts is ample for training) |
| **Superset geometry** | Build scene with max possible walls; show/hide via position or alpha | Memory overhead; rendering cost for invisible geometry |
| **Entity repositioning** | Keep fixed number of wall entities; reposition between episodes | Limited to mazes with similar wall count; complex bookkeeping |
| **Multiple scenes** | Build separate scenes per layout; swap between them | Memory-intensive; GPU memory limits applicability |
| **Model re-creation** | Rebuild the GPU model each episode | Breaks JIT compilation; negates batch speedup |

**Recommended strategy: Pre-generated maze pool.** Generate 500-1,000 layouts at startup using `labmaze`. Each parallel environment gets a different layout from the pool. Targets are repositioned within layouts using `set_pos()` / data field updates. This preserves >99% of the randomization benefit while enabling static scene compilation.

For a 9x9 maze with 4,096 parallel environments: 4,096 environments cycling through 1,000 layouts gives each layout ~4 uses per "epoch" — sufficient randomization for most RL training regimes.

---

## 7. Porting Roadmap

### 7.1 Recommended Sequence

```
Phase 1 (Week 1-2): Foundation
├── Choose primary engine (recommendation: Genesis or MJX Warp)
├── Implement maze pool generation using labmaze
├── Build minimal scene: ball + flat ground + walls + camera
├── Validate: ball rolls, camera renders, walls collide
│
Phase 2 (Week 2-4): Core Environment
├── Implement full maze construction from labmaze layouts
├── Add target spheres with proximity detection
├── Implement reward logic and episode management
├── Implement observation pipeline (RGB + border + optional extras)
├── Write Gymnasium/dm_env wrapper
│
Phase 3 (Week 4-5): Batch & Performance
├── Enable batch simulation (n_envs=1024+)
├── Enable GPU batch rendering
├── Profile and optimize bottlenecks
├── Verify determinism
│
Phase 4 (Week 5-6): Validation
├── Compare agent trajectories: reference vs ported implementation
├── Train baseline RL agent (e.g., IMPALA) on ported env
├── Compare learning curves and final performance
├── Document any behavioral differences
```

### 7.2 Which Engine First?

**For the fastest path to a working GPU-accelerated Memory Maze:**

| Priority | Engine | Rationale |
|----------|--------|-----------|
| **1st** | **Genesis** | Lowest barrier: Python-native, good batch sim, Madrona BatchRenderer, MJCF understanding. Best developer ergonomics for rapid iteration. |
| **2nd** | **MJX Warp** | Same physics guarantees fidelity. Warp rendering solves the GPU bottleneck. More JAX boilerplate but proven ecosystem. |
| **3rd** | **madrona_mjx** | Highest throughput ceiling, but requires understanding two systems (MJX + Madrona). Best for production-scale training after prototyping. |
| **4th** | **Madrona native** | Only if throughput requirements exceed what hybrid approaches can deliver. Full C++ rewrite is rarely justified for a benchmark. |

**Pragmatic recommendation:** Start with **Genesis** for fastest time-to-working-prototype. If rendering throughput is insufficient, integrate the Madrona BatchRenderer (which Genesis already wraps). If physics fidelity concerns arise, pivot to **MJX Warp** as the fallback with guaranteed MuJoCo equivalence.

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Probability | Impact | Affected Engines | Mitigation |
|------|-------------|--------|-----------------|------------|
| **GPU rendering quality differs from MuJoCo** | High | Medium | All | Visual differences are expected; retrain agents rather than transfer pretrained models |
| **Maze pool insufficient for training diversity** | Low | Medium | All | 1,000+ layouts is ample; increase pool size if needed |
| **Physics behavior diverges from reference** | Medium | High | Genesis, Madrona native | Use MuJoCo compat mode (Genesis) or MJX (guaranteed fidelity) |
| **Warp backend instability** | Medium | Medium | MJX Warp | New code (2026); have fallback to CPU rendering for validation |
| **BatchRenderer platform lock-in** | High | Low | Genesis, Madrona | CUDA+Linux is standard for RL training infrastructure |
| **Memory limits with large batch sizes** | Medium | Medium | All | Profile GPU memory; reduce batch size or maze pool as needed |
| **labmaze incompatibility with batch setup** | Low | Low | All | labmaze runs on CPU at startup; generates layouts before GPU scene build |

### 8.2 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Scope creep beyond benchmark porting** | Medium | High | Define MVP: image-only observation, 9x9 maze, single agent. Extend later. |
| **Engine API breaking changes** | Medium (Genesis, Madrona) | Medium | Pin versions; Genesis and Madrona are research code |
| **Insufficient documentation** | High (Madrona) | Medium | Read source code; study example projects; community Slack/Discord |
| **Performance doesn't justify effort** | Low | High | Profile reference implementation first; confirm rendering is the bottleneck |

### 8.3 Research Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **RL results don't reproduce on new engine** | Medium | High | Side-by-side comparison; accept visual differences if behavior matches |
| **Community doesn't adopt ported version** | Medium | Medium | Open-source release; publish comparison results; maintain reference compatibility |

---

## 9. Recommendation

### Primary Recommendation: Genesis

**Start with Genesis** as the primary porting target for the following reasons:

1. **Fastest path to a working prototype.** Python-native API means rapid iteration. No JAX boilerplate, no C++ compilation cycles. The `gs.morphs.Box/Sphere` primitives directly mirror how dm_control constructs the maze.

2. **Battle-tested batch simulation.** `scene.build(n_envs=4096)` is a first-class feature with per-environment reset. The locomotion examples demonstrate exactly this pattern at scale.

3. **Madrona BatchRenderer built in.** Genesis already wraps Madrona's GPU batch renderer — you get the rendering throughput of Madrona without writing C++.

4. **MuJoCo compatibility mode.** `enable_mujoco_compatibility=True` provides a path to validate physics behavior against the reference implementation.

5. **PyTorch-native tensors.** Zero-copy integration with standard RL training frameworks (PPO, IMPALA, R2D2) that the Memory Maze paper evaluates.

### Secondary Recommendation: MJX Warp

**Pivot to MJX Warp if physics fidelity is critical:**

- If trained policies must transfer between the ported environment and the original MuJoCo version, MJX is the only option that guarantees identical physics.
- The Warp rendering path solves the GPU rendering bottleneck.
- The JAX ecosystem (Brax, PureJaxRL) provides strong RL training infrastructure.
- Trade-off: more boilerplate, JAX learning curve, newer rendering code.

### Future Optimization: madrona_mjx

**For production-scale training at maximum throughput,** investigate the `madrona_mjx` hybrid after establishing a working prototype. This combines MJX's physics fidelity with Madrona's purpose-built batch renderer — the best of both worlds, at the cost of integrating two systems.

### What Not To Do

- **Don't start with native Madrona C++** unless you have a dedicated C++ developer and throughput requirements that hybrid approaches can't meet. The 4-8 week effort for a full rewrite is rarely justified for a benchmark environment.
- **Don't use MJX JAX backend without GPU rendering** — CPU rendering fallback eliminates most of the speedup for vision-based RL.
- **Don't try to replicate dm_control's composer framework** — build a simpler, purpose-built environment class. The composer abstraction adds complexity that isn't needed when targeting a single environment.

---

## 10. Appendix: Individual Analysis Reports

Detailed per-engine analyses are available in:
- `analysis_genesis.md` — Genesis engine deep dive
- `analysis_mujoco_mjx.md` — MuJoCo/MJX deep dive
- `analysis_madrona.md` — Madrona engine deep dive
- `analysis_memory_maze_reference.md` — Memory Maze reference implementation analysis

---

*Report compiled February 27, 2026. Based on source code analysis of Genesis, MuJoCo 3.5.1/MJX, Madrona, and Memory Maze repositories.*
