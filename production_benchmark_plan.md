# Production Benchmark Plan: Full-Scale Exclusive Runs & Prerequisite Tasks

## 1. Overview & Objectives

This document defines the plan for the full-scale, exclusive benchmark of the `mini_app` across all ML interface variants (Direct SmartSim, Direct AIx, CMI AIx, CMI SmartSim, and CMI PhyDLL).

### Core Goals
1. **Accurate Performance & Overhead Comparison**:
   - Compare Direct vs. CMI overhead for SmartSim and AIx under identical hardware allocations.
   - Compare 1-time terrain preloading vs. recurring terrain transfer in Direct SmartSim.
   - Evaluate CMI SmartSim parallel puts (`c0`) and sequential chain variants (`c1`, `c2`, `c3`).
   - Evaluate AIx Collective vs. Pipelined (P2P credit exchange) transport.
   - Evaluate PhyDLL transport layouts (`packed` vs. `uniform_chunks`) and clients (C++ vs. Python).
2. **Standardized Geometry & Workload**:
   - Grid: `3840x2160` (WaterCNN model, `chunk_size=12`, total 8,294,400 cells / grid elements).
   - Profiles: Both CPU-inference and 4-GPU-inference profiles tested at the same `3840x2160` resolution for direct comparability.
   - Execution: 3 runs per case, 102 steps per run (step 2 warm-up, steps 4–102 produce 50 steady-state ML samples per run; 150 steady samples total).
3. **Execution Environment**:
   - Slurm Account: `thes2181`
   - Node Allocation: `--exclusive` nodes with `--mem=0`.

---

## 2. Target Hardware & Topology Specifications

### Hardware (CLAIX-2023)
- **CPU Nodes (`c23mm`)**: Dual Intel Xeon Platinum 8468 (Sapphire Rapids), 96 physical cores (1 socket × 48 cores × 2), 512 GB DDR5 RAM, InfiniBand NDR200.
- **GPU Nodes (`c23g`)**: Dual Intel Xeon Platinum 8468 (96 cores), 4× NVIDIA H100 SXM5 GPUs (94 GB HBM3 per GPU, 50 MB L2 cache per GPU, 900 GB/s NVLink 4 interconnect).

### Topology Mapping

| Provider / Mode | Profile | Solver Component (`c23mm`) | Inference Component (`c23mm` / `c23g`) | Total MPI Communicator |
|---|:---:|---|---|---|
| **Direct / CMI SmartSim** | CPU | 1 exclusive node, 96 ranks × 1 CPU core | 1 exclusive node, 1 SmartSim DB × 96 CPU cores | 96 solver ranks (`12x8` grid) |
| **Direct / CMI SmartSim** | GPU | 1 exclusive node, 96 ranks × 1 CPU core | 1 exclusive node, 1 SmartSim DB × 96 CPU cores + 4 GPUs | 96 solver ranks (`12x8` grid) |
| **Direct / CMI AIx** | CPU | 1 exclusive node, 96 ranks × 1 CPU core | *(In-process host inference; no separate node)* | 96 solver ranks (`12x8` grid) |
| **Direct / CMI AIx** | GPU | 1 exclusive node, 96 ranks × 1 CPU core | 1 exclusive node, 4 AIx controller ranks × 24 CPU cores + 4 GPUs | 100 ranks (`10x10` grid: 96 CPU + 4 GPU) |
| **CMI PhyDLL** | CPU | 1 exclusive node, 96 ranks × 1 CPU core | 1 exclusive node, 1 DL client × 96 CPU cores | 96 solver ranks (`12x8` grid) + 1 DL rank |
| **CMI PhyDLL** | GPU | 1 exclusive node, 96 ranks × 1 CPU core | 1 exclusive node, 4 DL clients × 24 CPU cores + 4 GPUs | 96 solver ranks (`12x8` grid) + 4 DL ranks |

---

## 3. Provider Batch Size Semantics & Scaling Behavior

Understanding what `batch_size` does for each provider is essential:

1. **Direct SmartSim (`--ml-batch-size`)**:
   - Semantics: Client-side chunking cap per solver rank.
   - For `3840x2160` with 96 ranks, each rank has $86,400$ cells.
   - Values $\ge 86,400$ cause 1 request per rank per step.
   - Values $< 86,400$ (e.g. 43,200) split each rank's step into multiple sequential chunks/keys.
   - *RedisAI Dynamic Batching*: RedisAI has server-side model-store parameters (`batch_size`, `min_batch_size`, `min_batch_timeout`). It aggregates concurrent client requests targeting the **same model key** and same tensor shape into a single batch on dimension 0. Rank-unique input/output tensor keys are supported and necessary to avoid write races while allowing server-side coalescing.

2. **CMI SmartSim (`provider.batch_size`, `min_batch_size`, `min_batch_timeout`)**:
   - In CMI's SmartSim provider, these three settings map directly to SmartRedis's model registration.
   - In the current `terrain_solver.cpp`, `provider.batch_size` was hardcoded to 0; our calibration benchmark uses the pure CMI configuration where `batch_size` and `min_batch_size` can be set to test controlled aggregation.

3. **Direct AIx & CMI AIx (`--ml-batch-size` / `provider.batchsize`)**:
   - Semantics: The GPU controller's internal PyTorch forward-pass chunk size after gathering the workgroup samples.
   - With 96 solver ranks and 4 GPU controllers, each controller gathers $24 \times 86,400 = 2,073,600$ samples.
   - AIx chunks this $2.07\text{M}$ tensor into slices of `batchsize` for PyTorch forward execution.

4. **PhyDLL (`config_phydll.toml` `provider.batch_size`)**:
   - Semantics: The DL client's internal PyTorch forward-pass chunk size over its aggregated batch from connected PHY ranks.
   - Wire communication sends the complete frame; the DL client transfers the aggregate input to GPU, then uses `input_tensor.slice(0, offset, chunk_size)` to control forward-pass activation VRAM, concatenates outputs, and copies back to host.

---

## 4. Prerequisite Tasks & 1-GPU Batch/Memory Calibration Suite

Before running the full 84-job exclusive production suite, we execute the standalone batch/memory calibration suite in `batch_memory_benchmark/`.

### 1-GPU Calibration Harness Design (`c23g` allocation: 24 cores, 1 GPU)
- **Workload**: Fixed total work of 2,000,000 samples ($[2000000, 18]$ float32 input) per step.
- **Ranks**:
  - SmartSim & PhyDLL: 20 fake solver ranks (100,000 samples each) + 1 DB/DL rank (4 cores).
  - AIx: 20 total ranks (19 CPU ranks + 1 GPU controller rank, 100,000 samples each).
- **Tested Forward Batch Sizes**: $B \in \{100\,000,\; 500\,000,\; 2\,000\,000\}$.
- **Telemetry Sidecar**: `nvidia-smi` sampling GPU VRAM at 50 ms resolution (`gpu_telemetry_<run_id>.csv`).
- **Regression Fitting & Cache Boundaries**:
  - Linear regression: $\text{Peak VRAM}(B) = V_0 + \alpha \cdot B$.
  - Extrapolate maximum capacity for 94 GB HBM3 ($B_{\max} = (94\text{ GB} - V_0) / \alpha$).
  - Calculate safe 80 GB operational headroom limit.
  - Evaluate L2 cache boundary (50 MB L2) and through-cache latency scaling.

---

## 5. Full Production Benchmark Suite Specification

Once prerequisites are complete:
- **Configurations**: 14 modes per profile × 2 profiles (CPU & GPU) × 3 replicate runs = **84 exclusive production jobs**.
- **Case Matrix**:
  1. Direct SmartSim (Preload terrain, split I/O)
  2. Direct SmartSim (Force terrain upload, split I/O)
  3. Direct AIx (Collective)
  4. Direct AIx (Pipelined)
  5. CMI AIx (Collective)
  6. CMI AIx (Pipelined)
  7. CMI SmartSim (`c0`: parallel puts)
  8. CMI SmartSim (`c1`: sequential chain 1)
  9. CMI SmartSim (`c2`: sequential chain 2)
  10. CMI SmartSim (`c3`: sequential chain 3)
  11. CMI PhyDLL C++ (Packed layout)
  12. CMI PhyDLL C++ (Uniform chunks layout)
  13. CMI PhyDLL Python (Packed layout)
  14. CMI PhyDLL Python (Uniform chunks layout)
- **Run Length**: 102 steps (50 warm ML measurement steps per run; 150 steady samples per mode).
