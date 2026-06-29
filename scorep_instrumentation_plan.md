# Score-P Manual Instrumentation Plan for mini_app Coupling Evaluation

## Goal

Detailed per-phase performance characterization of mini_app coupling across all five backends
(AIX, PhyDLL C++ DL, PhyDLL Python DL, SmartSim-CMI, direct-SmartSim-from-solver) using
**Score-P 8.4 manual instrumentation**, profiling mode (CUBE4), with **nvml-in-process +
nvidia-smi sidecar** for GPU memory, **PAPI infiniband + net counters** for per-region IB
byte traffic, and **Score-P Python adapter** for `phydll_dl_client.py`.

## Decisions (locked in)

| # | Decision | Resolution |
|---|----------|------------|
| D1 | Backend scope | **CMI (AIX, PhyDLL C++ DL, PhyDLL Python DL, SmartSim) + direct SmartSim solver path.** All five variants instrumented. |
| D2 | Score-P module | **`Score-P/8.4`** (gompi/2022a, GCC 11.3 + OpenMPI 4.1.4). No `--cuda` adapter (unavailable on this toolchain; Score-P/8.4 built without libcuda). CUDA 12.4 toolchain remains in use for nvml + nvidia-smi. |
| D3 | Output mode | **Profile only (CUBE4)** for now. One trace run later if needed after `scorep-score` buffer sizing. |
| D4 | Adapters enabled | `--nocompiler --user --mpp=mpi --io=none --memory=malloc --thread=none --nocuda` |
| D5 | GPU timing/memory | **nvmlDeviceGetMemoryInfo() in-process** for AIx server rank + PhyDLL C++ DL (we own these). **nvidia-smi dmon sidecar** for the Redis process (external, we don't own it). |
| D6 | IB traffic | **PAPI infiniband + net counters** (`SCOREP_METRIC_PAPI`). Exact event tokens TBD via `papi_avail -a` on a compute node. |
| D7 | Python DL timing | **Score-P Python adapter** (`pip install scorep` in smartsim_cuda-12 venv, launch via `scorep python`). |
| D8 | Toolchain | **Keep production gompi/2022a** (GCC 11.3 + OpenMPI 4.1.4 + CUDA 12.4 + cuDNN + HDF5 1.12.2). No migration. |
| D9 | Build gating | **Env-gated** by `USE_SCOREP=1`. Default (unset) behaviour unchanged. |
| D10 | Model for testing | **`MODEL_NAME_ENV=perfect_model`** for Score-P runs (shorter runtime, devel-node-friendly). |
| D11 | Implementation | **Step-by-step with validation at each step.** |

## Cluster capability (verified)

```
Score-P/8.4              → CUDA-less, PMIx 4.1.2-aligned, built against gompi/2022a
PAPI/7.0.0               → infiniband (98 native events), net (320 counters), perf_event, appio, coretemp
CubeLib/4.9.1             → cube4 / CubeGUI for reading .cubex profiles
Vampir/10.5.0             → timeline viewer (future trace run)
VampirServer/10.5.0       → server-side OTF2 reader
```

Score-P adapters present in the installed lib: compiler, memory (cxx, libc), MPI, posix_io.
**No CUDA adapter** in this Score-P build. GPU measurement via nvml/sidecar instead.

## Toolchain (production env, unchanged)

```
module load gompi/2022a          # pulls GCC 11.3 + OpenMPI 4.1.4 + cuDNN + imkl
module load Score-P/8.4
module load PAPI/7.0.0
# CUDA/12.4.0 already in the toolchain for nvml + nvidia-smi
```

## Score-P build environment (env-gated by `USE_SCOREP=1`)

```bash
SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--nocompiler --user --mpp=mpi --io=none --memory=malloc --thread=none --nocuda"
SCOREP_ENABLE_PROFILING=true
SCOREP_ENABLE_TRACING=false
SCOREP_TOTAL_MEMORY=16000K          # bump after scorep-score sizing
SCOREP_EXPERIMENT_DIRECTORY="scorep_${backend}_${run_idx}"
SCOREP_METRIC_PAPI='infiniband:::port_xmit_bytes,infiniband:::port_rcv_bytes,net:::rx_bytes,net:::tx_bytes'
```

Compiler: swap `CXX` → `scorep-mpicxx` in CMake configure.
Python DL: launch via `scorep python phydll_dl_client.py`.
CMake extra: `-DWITH_SCOREP=ON -DWITH_AIX=ON -DFORCE_AIX_REBUILD=ON` (first time only to rebuild AIx prebuilt).

---

## Instrumentation Surface Map

### 1. Pre-flight — AIxeleratorService fork

The current `extern/AIxeleratorService/INSTALL-SCOREP/` was built **with `WITH_SCOREP=OFF`**
(confirmed: `CMakeCache.txt` shows `WITH_SCOREP:BOOL=OFF`). Rebuild required.

**Existing regions (Tom/Fabian, already in source, gated by `#ifdef SCOREP`):**

| File | Line range | Region |
|------|------------|--------|
| `src/aixeleratorService/aixeleratorService.cpp` | `:369–371` | `inference` (function-level) |
|  | `:376–383` | `gatherInputData` (around `MPI_Gatherv`) |
|  | `:389–397` | `deviceInference` |
|  | `:401–409` | `hostInference` |
|  | `:414–421` | `scatterOutputData` (around `MPI_Scatterv`) |
| `src/inferenceStrategy/torchInference/torchInference.cpp` | `:90–179` | `torchInference` (function-level) |
|  | `:133–139` | `torchInference::forward` (per-chunk; GPU branch) |
|  | `:164–171` | `torchInference::forward` (CPU branch) |

**New regions to add (our fork):**

| File | Line | Region name | Notes |
|------|------|-------------|-------|
| `torchInference.cpp` | `:121` | `h2d_copy` | `input_gpu_ = input_batch_.to(kCUDA)` per chunk |
| `torchInference.cpp` | `:142` | `d2h_copy` | `output_.slice(...) = output_gpu_.to(kCPU)` per chunk |
| `torchInference.cpp` | `:148` | `cpu_chunk_slice` | CPU-branch `input_batch_.slice(...)` |
| `torchInference.cpp` | `:173` | `cpu_chunk_assign` | CPU-branch `output_.slice(...) = model_out` |
| `aixeleratorService.cpp` | around `:369` | `gpu_mem_used_bytes` (metric) | `nvmlDeviceGetMemoryInfo()` before/after inference |

Add a small `nvml.hpp` helper in the `aixeleratorService` util, link `-lnvidia-ml`.
The nvml metric is a `SCOREP_USER_METRIC_UINT64` emitted at function exit from `inference()`.

### 2. CPP-ML-Interface library

Add top-level `option(WITH_SCOREP OFF)` and `find_package(Scorep)` in `CPP-ML-Interface/CMakeLists.txt`,
reusing the existing `extern/AIxeleratorService/cmake/FindScorep.cmake`. Call:

```cmake
scorep_instrument_target(cpp_ml_interface_library USER ON COMPILER OFF MEMORY ON MPP MPI)
scorep_instrument_target(cpp_ml_interface_headers INTERFACE USER ON COMPILER OFF)
```

All new regions guarded by `#ifdef USE_SCOREP` (the define added automatically by the Score-P wrapper).

#### 2a. Top dispatch — `include/ml_coupling.hpp`

| Line | Region name |
|------|-------------|
| `:302` | `cppml_prepare_input` |
| `:303–304` | `cppml_static_inference` |
| `:305` | `cppml_finalize_output` |

#### 2b. PhyDLL provider — `include/provider/ml_coupling_provider_phydll.hpp`

| Line(s) | Region name | Notes |
|---------|-------------|-------|
| `:413–437` | `phydll_prepack` | float→double cast; `prepare_data_buffer()` |
| `:113–114` | `phydll_send` | `phydll_set_field` + `phydll_send()` |
| `:116–133` | `phydll_recv` | `phydll_recv()` + `phydll_get_field` loop |
| `:439–461` | `phydll_unpack` | double→float cast; `unpack_output_buffer()` |

Byte-count user metrics for PhyDLL:
- `bytes_sent_logical` = `sum(input_sizes) * sizeof(float)` (solver-side payload)
- `bytes_sent_actual` = `sum(input_sizes) * sizeof(double)` (PhyDLL wire overhead)
- `bytes_recv_logical` = `sum(output_sizes) * sizeof(float)`
- `bytes_recv_actual` = `sum(output_sizes) * sizeof(double)`

These directly answer the "logical 2560 floats vs transferred 2560 doubles + overhead" question.

#### 2c. SmartSim provider — `include/provider/ml_coupling_provider_smartsim.hpp`

| Line range | Region name | Notes |
|------------|-------------|-------|
| `:457–493` | `smartsim_put_tensor` | per-tensor `put_tensor` call at `:487` |
| `:508–526` | `smartsim_run_model` | `run_model[_multigpu]` at `:512/514/518/520` |
| `:529–566` | `smartsim_unpack_tensor` | `unpack_tensor` at `:560` |

Byte-count user metrics: `input_bytes` = tensor `numel() * element_size()` (computed at `:397,404`),
`output_bytes` = likewise.

#### 2d. AIX provider — `include/provider/ml_coupling_provider_aixelerator.hpp`

| Line | Region name |
|------|-------------|
| `:127` | `aix_inference` | wraps `service->inference()` — sub-phases come from the instrumented AIxeleratorService `.so` |

#### 2e. Provider-base merge (flex fallback) — `include/provider/ml_coupling_provider.hpp`

| Line range | Region name | Notes |
|------------|-------------|-------|
| `:216–297` | `merge_stack` | `stack_data()` — the per-batch copy for flex-ordered stacking |
| `:299–427` | `merge_list` | `list_data()` — per-batch copy for flex-keyed list |

These are the fallback-logic merge copies the user asked about.

#### 2f. Data layer — `include/data/ml_coupling_data.hpp`

| Line range | Region name | Notes |
|------------|-------------|-------|
| `:140–143` | `data_from_flat_copy` | allocation + `std::copy` |
| `:328–342` | `data_as_flat_vector` | non-contiguous flatten (per-element un-r这个-index + `at()`) |
| `:344–348` | `data_flatten` | `as_flat_vector` + `from_flat_copy` |
| `:357–428` | `data_unflatten` | recursive reshape from flat back to nested |

#### 2g. Application layer — `include/application/ml_coupling_application_flow_extrapolator.hpp`

| Line(s) | Region name | Notes |
|---------|-------------|-------|
| `:186–196` | `flowex_make_input_buffer` | allocation `[nFields*cubes, seqLen, cubeSize]` |
| `:72,88–98` | `flowex_extract_cubes` | per-field cube packing into model input |
| `:199–208` | `flowex_make_output_buffer` | allocation `[nFields*cubes, forecast_window, cubeSize]` |
| `:141–146` | `flowex_reconstruct_output` | cube→grid scatter |

### 3. mini_app solver — `solver_cpp/terrain_solver.cpp`

Add `solver_cpp/scorep_regions.hpp` with all `SCOREP_USER_REGION_DEFINE` handles, guarded by
`#ifdef USE_SCOREP`. Wrap the existing chrono-timed intervals with `SCOREP_USER_REGION_BEGIN/END`.

| Region name | Path | Line range | Notes |
|-------------|------|------------|-------|
| `solver_setup` | both | `:3052→3461` | config + model-load + pre-upload |
| `solver_step` | both | `:3534→3590` | per-step wall (already chrono via `step_ms`) |
| `halo_exchange` | both | `:3529` | around `exchange_halo_1cell` |
| `ml_step_wall` | CMI | `:1652→1732` | already chrono |
| `ml_step_wall` | SS | `:1773→1991` | already chrono |
| `input_prep` | CMI | `:1665→1692` | `fill_flat_input_chunk` / `build_nested_view` |
| `input_prep` | SS | `:1799→1819` | `build_nested_view_for_field_chunk` + `build_flat_input_chunk` copy |
| `ml_send` | SS | `put_tensor` `:1820/1828/1846` per call | also preload `:3441` |
| `ml_inference` | SS | `run_model` `:1872/1874/1886/1888`, retry `:1858–1919` | |
| `ml_inference` | CMI | `ml_coupling.ml_step()` at `:1698` | opaque — sub-phases come from CMI library profiling |
| `ml_recv` | SS | `unpack_tensor` `:1932` | not `get_tensor`, no poll loop |
| `ml_unpack_postprocess` | CMI | `:1705→1710` | `std::copy` output → `tile_output` |
| `ml_unpack_postprocess` | SS | `:1959→1964` | `tile_output` → `next` |

Byte-count user metrics in solver: feed from existing `add_ml_traffic()` (`:522–526`). Add:
- `SCOREP_USER_METRIC_UINT64("ml_bytes_input", input_bytes)` around each `put_tensor` /
  `ml_step()` call.
- `SCOREP_USER_METRIC_UINT64("ml_bytes_output", output_bytes)` around each `unpack_tensor` /
  post-ml_step copy.

**Keep all existing chrono + `/proc` + `getrusage` + `ML_TRAFFIC` + `NET_USAGE` lines** — they
remain the human-readable cross-check, independent of Score-P.

CMake: in `solver_cpp/CMakeLists.txt`, when `USE_SCOREP=ON`:
```cmake
set(CMAKE_CXX_COMPILER scorep-mpicxx)
target_compile_definitions(${PROJECT_NAME} PRIVATE USE_SCOREP)
```

### 4. PhyDLL C++ DL side

Files: `dl_clients/dl_client.cpp`, `dl_clients/phydll_dl_runtime.cpp`, `dl_clients/phydll_dl_runtime.hpp`.

| File | Line(s) | Region name | Notes |
|------|---------|-------------|-------|
| `phydll_dl_runtime.cpp` | `:138–161` | `dl_recv` | `phydll_irecv` + `phydll_wait_irecv` + `phydll_get_field` loop |
| `phydll_dl_runtime.cpp` | `:110–136` | `dl_send` | `phydll_set_field` loop + `phydll_send` |
| `dl_client.cpp` | `:341–343` | `dl_h2d` | `torch::from_blob(...).clone().to(torch_device)` |
| `dl_client.cpp` | `:348–352` | `dl_torch_forward` | chunked `model.forward({chunk_tensor})` at `:351` |
| `dl_client.cpp` | `:353–365` | `dl_d2h_scatter` | output → host, `static_cast<double>` scatter per batch |
| `dl_client.cpp` | `:384` | `dl_send_output` | `runtime.send_output(output)` |

Build: compile with `scorep-mpicxx --user --mpp=mpi --memory=malloc --nocuda`.
MPI byte counts free via `--mpp=mpi`.

### 5. PhyDLL Python DL side

File: `dl_clients/phydll_dl_client.py`.

Using the `scorep` pip package (install into `smartsim_cuda-12` venv):

```python
import scorep.user_regions
with scorep.user_region("py_recv"):
    fields = dll.recv()                                  # :188

with scorep.user_region("py_inference"):
    # pack batch :235–255
    # H2D :253–255
    for chunk_idx in range(0, batch_size, max_chunk_size):
        model(chunk_tensor)                              # :266
    output_tensor = torch.cat(outputs)                    # :267
    output_tensor = output_tensor.cpu().contiguous().numpy().flatten()  # :269

with scorep.user_region("py_send"):
    # scatter back :272–277
    dll.send({"DL-OUT": output})                         # :289
```

Launch: `scorep python phydll_dl_client.py` (produces a separate scorep directory per MPMD component;
Vampir can load both archives on a shared timeline).

### 6. GPU memory sidecar (SmartSim Redis process)

In `proper_slurm_job.sh`, for SmartSim runs only:

```bash
if [[ -n "${USE_SCOREP:-}" ]]; then
    nvidia-smi dmon -s mu -d 1 -o TD > "${SCOREP_EXPERIMENT_DIRECTORY}/redis_gpu.log" &
    REDIS_GPU_MONITOR_PID=$!
    trap "kill ${REDIS_GPU_MONITOR_PID} 2>/dev/null" EXIT
fi
```

Parse the log in `analysis.ipynb` — independent of Score-P, per-step GPU memory layer.

### 7. Build pipeline wiring

Scripts to modify:

| Script | Change |
|--------|--------|
| `slurm_build.sh` | Gate: if `USE_SCOREP=1`, load `Score-P/8.4` + `PAPI/7.0.0`, swap to `scorep-mpicxx`, set `SCOREP_WRAPPER_INSTRUMENTER_FLAGS` |
| `proper_slurm_job.sh` | Export `SCOREP_ENABLE_PROFILING`, `SCOREP_METRIC_PAPI`, pass `-DWITH_SCOREP=ON` to cmake, wrap Python DL launch with `scorep python`, sidecar `nvidia-smi dmon` for SmartSim |
| `basic_backend_test.sh` | When `USE_SCOREP=1`, set `MODEL_NAME_ENV=perfect_model` |
| `CPP-ML-Interface/build.sh` | Gate: if `USE_SCOREP=1`, add `-DWITH_SCOREP=ON` to cmake call |
| `build_phydll.sh` | Gate: if `USE_SCOREP=1`, compile with `scorep-mpicxx` instead of `mpicc` |
| `dl_clients/CMakeLists.txt` | Gate: if `WITH_SCOREP=ON` / `-DUSE_SCOREP`, use `scorep_instrument_target` |

---

## Pre-flight steps (executed first)

1. **Remove stale build dirs:**
   ```bash
   rm -rf solver_cpp/{build_mpi,stable,build_cpp_module}
   ```
   These are confirmed-stale experimental OpenMPI 4.1.6 / Clang-18 builds from April-June. The
   production build dir is `solver_cpp/build/` (the default `COMPILE_OUTPUT_PATH`).

2. **Wipe and rebuild AIxeleratorService prebuilt:**
   ```bash
   cd extern/AIxeleratorService
   rm -rf INSTALL-SCOREP/
   mkdir INSTALL-SCOREP
   ```
   Then configure with `-DWITH_SCOREP=ON -DWITH_TORCH=ON -DCMAKE_INSTALL_PREFIX=INSTALL-SCOREP`
   using gompi/2022a + Score-P/8.4 module environment. This activates all existing `#ifdef SCOREP`
   regions in the source.

3. **Verify `find_package(Scorep)` works** under gompi/2022a + Score-P/8.4 by running a minimal
   cmake probe against `extern/AIxeleratorService/cmake/FindScorep.cmake`.

---

## Devel-node smoke test plan

**Allocation parameters** (per user instruction — devel partition, no `--exclusive`, 96 tasks for
full-node effect, 1h max, 238 GB mem, default account):

```bash
#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --ntasks=96
#SBATCH --mem=238000
#SBATCH --account=${SLURM_DEFAULT_ACCOUNT}
```

**What to validate in order:**

| # | Smoke | What to confirm |
|---|-------|-----------------|
| 1 | AIx variant rebuild + run | `WITH_SCOREP=ON` + `FORCE_AIX_REBUILD=ON` → libAIxeleratorService.so contains scorep symbols; `scorep_aix_smoke/profile.cubex` opens in `cube4` and shows `inference(gatherInputData, deviceInference/hostInference, scatterOutputData, torchInference::forward, h2d_copy, d2h_copy)` with nonzero visit counts. |
| 2 | CMI library build | After adding all CMI regions, re-smoke AIX + CMI-SmartSim + CMI-PhyDLL-C++. Profile shows `cppml_prepare_input`, `cppml_static_inference` (nesting `aix_inference`/`phydll_prepack`/...), `cppml_finalize_output` nested inside `ml_step_wall`. |
| 3 | Solver build | After adding solver regions, all five backend variants show `solver_step` ⟶ `ml_step_wall` ⟶ `input_prep` ⟶ `ml_send`/`ml_inference`/`ml_recv` ⟶ `ml_unpack_postprocess`. |
| 4 | PhyDLL DL side | C++ DL produces a separate scorep dir with `dl_recv`/`dl_torch_forward`/`dl_send_output` regions. Python DL (via `scorep python`) produces a third scorep dir with `py_recv`/`py_inference`/`py_send`. Timestamps align. |
| 5 | PAPI + nvidia-smi | `SCOREP_METRIC_PAPI` produces non-zero IB/net byte counters in the profile. `nvidia-smi dmon` log has per-step GPU memory rows. |

---

## Output analysis (`analysis.ipynb` — follow-up notebook edits)

After all instrumentation is stable, extend the notebook to:

1. Read CUBE4 profiles via `pycube` / `cube4 --export -o output_file` for each backend.
2. Extract per-phase wall times from the region tree for use in bar charts.
3. Plot per-backend breakdown bars (input_prep, send, inference, recv, unpack).
4. Plot bytes-logical-f32 vs bytes-actual-f64 per backend (the PhyDLL overhead quantification).
5. Plot PAPI IB/net per-region byte counters as a grouped bar alongside the analytic ML_TRAFFIC.
6. Parse `nvidia-smi dmon` logs for Redis GPU memory vs step time.

---

## Implementation sequence (step-by-step with validation)

### Step 1 — Pre-flight clean + AIx fork rebuild + devel smoke validate
- Remove stale build dirs.
- Wipe INSTALL-SCOREP, rebuild AIxeleratorService with `WITH_SCOREP=ON`.
- Add new `#ifdef SCOREP` regions to `torchInference.cpp` + `aixeleratorService.cpp`.
- Submit devel smoke for AIX variant: confirm `scorep_aix_*/profile.cubex` region tree.
- **Gate:** profile opens in cube4 with expected region hierarchy + nonzero visit counts.

### Step 2 — CMI library instrumentation + build wiring
- Add top-level `WITH_SCOREP` option, `find_package(Scorep)`, `scorep_instrument_target`.
- Add all `#ifdef USE_SCOREP` regions and user-metric byte counters in provider/data/application headers.
- Rebuild, re-smoke AIX + CMI-SmartSim + CMI-PhyDLL-C++.
- **Gate:** profile shows `cppml_*` top dispatch nesting per-provider sub-regions.

### Step 3 — Solver instrumentation + direct-SmartSim path
- Add `solver_cpp/scorep_regions.hpp` + regions in `terrain_solver.cpp`.
- CMake `USE_SCOREP` wiring in `solver_cpp/CMakeLists.txt`.
- Rebuild, re-smoke all 5 backend variants.
- **Gate:** profile shows `solver_step` ⟶ full nested hierarchy for every variant.

### Step 4 — PhyDLL DL side (C++ + Python)
- Region-wrap `dl_client.cpp` + `phydll_dl_runtime.cpp`.
- Install `scorep` pip package; wrap `phydll_dl_client.py` with `scorep.user_region`.
- Build the C++ DL under `scorep-mpicxx`; launch Python DL via `scorep python`.
- Devel smoke the PhyDLL C++ and Python variants.
- **Gate:** separate scorep dirs per MPMD component, regions visible, timestamps plausible.

### Step 5 — PAPI + nvidia-smi + notebook
- Enumerate PAPI IB/net event tokens on a compute node (`papi_avail -a`).
- Wire nvidia-smi sidecar into `proper_slurm_job.sh`.
- Full multi-variant Score-P run.
- Extend notebook with per-phase bars + byte-count logical-vs-actual + PAPI IB + GPU memory.

---

## Relevant files

| Path | Role |
|------|------|
| `solver_cpp/terrain_solver.cpp` | mini_app MPI solver; has existing chrono/RSS/IB/ML_TRAFFIC timing |
| `solver_cpp/CMakeLists.txt` | solver build; add USE_SCOREP wiring |
| `proper_slurm_job.sh` | primary job script; env-gate Score-P + sidecar launch |
| `slurm_build.sh` | build-only script; env-gate module load + compiler swap |
| `basic_backend_test.sh` | backend-test launcher; scopes backends, sets MODEL_NAME_ENV |
| `CPP-ML-Interface/include/ml_coupling.hpp` | top-level `step()` dispatch; wrap `:302–305` |
| `CPP-ML-Interface/include/provider/ml_coupling_provider.hpp` | provider base; `stack_data`/`list_data` merge regions |
| `CPP-ML-Interface/include/provider/ml_coupling_provider_aixelerator.hpp` | AIX provider; wrap `service->inference()` |
| `CPP-ML-Interface/include/provider/ml_coupling_provider_phydll.hpp` | PhyDLL provider; wrap prepack/send/recv/unpack + byte metrics |
| `CPP-ML-Interface/include/provider/ml_coupling_provider_smartsim.hpp` | SmartSim provider; wrap put/run/unpack |
| `CPP-ML-Interface/include/data/ml_coupling_data.hpp` | data layer; wrap flatten/copy hotspots |
| `CPP-ML-Interface/include/application/ml_coupling_application_flow_extrapolator.hpp` | cubing/ghost; wrap input/output buffer + cube packing |
| `CPP-ML-Interface/CMakeLists.txt` | CMI library build; add WITH_SCOREP option |
| `CPP-ML-Interface/extern/AIxeleratorService/src/aixeleratorService/aixeleratorService.cpp` | AIx orchestrator; already has regions, add nvml user metric |
| `CPP-ML-Interface/extern/AIxeleratorService/src/inferenceStrategy/torchInference/torchInference.cpp` | AIx torch inference; already has forward region, add H2D/D2H/CPU-slice regions |
| `CPP-ML-Interface/extern/AIxeleratorService/cmake/FindScorep.cmake` | Score-P CMake find module; already in tree |
| `CPP-ML-Interface/dl_clients/dl_client.cpp` | C++ DL driver; wrap H2D/forward/D2H/scatter/send |
| `CPP-ML-Interface/dl_clients/phydll_dl_runtime.cpp` | C++ DL runtime; wrap recv_fields/send_output |
| `CPP-ML-Interface/dl_clients/phydll_dl_client.py` | Python DL client; wrap recv/inference/send with scorep.user_region |
| `CPP-ML-Interface/dl_clients/CMakeLists.txt` | DL client build; add scorep_instrument_target |
| `CPP-ML-Interface/build.sh` | CMI build script; env-gate scorep cmake args |
| `CPP-ML-Interface/build_phydll.sh` | PhyDLL build; env-gate compiler swap |
| `analysis.ipynb` | parse scorep profiles + sidecar logs for bar charts |
| `home ~/Master-Thesis/notes/insights.md:499-545` | conceptual Score-P reference document |
| `home /hpcwork/ro092286/MMCP_2026_.../install-MAIA.sh` | MMCP Score-P build reference (SCOREP_WRAPPER_INSTRUMENTER_FLAGS) |
| `home /hpcwork/ro092286/MMCP_2026_.../scorep-*/scorep.cfg` | MMCP Score-P runtime config reference |

---

## Risks and mitigations

| # | Risk | Mitigation |
|---|------|------------|
| R1 | `find_package(Scorep)` fails under gompi/2022a | Test in Step 0.3 before proceeding. Use full path to `scorep-config` if needed via `SCOREP_ROOT_DIR`. |
| R2 | AIxeleratorService forward/libtorch ABI mismatch under scorep-mpicxx | Rebuild the AIx prebuilt under the Score-P-prefixed compiler to ensure ABI consistency between instrumented AIx `.so` and CMI `.so`. |
| R3 | Python `scorep` pip install conflicts with smartsim_cuda-12 venv | Smoke install on a devel node; use `pip install --user` if venv write fails. |
| R4 | `nvidia-smi dmon` output parsing breaks if Redis GPU node differs from solver node | Launch sidecar via `srun --nodelist=<db_node>` in the SmartSim DB group. |
| R5 | PAPI event tokens differ per node architecture | Enumerate on the actual compute-node type used (not login). The 96-task devel allocation serves as the enumeration point. |
| R6 | Existing stale build_mpi artifacts cause confusion | Already confirmed: `build_mpi/` is unused by production. Removed in Step 0.1. |
| R7 | Score-P buffer overflow for larger runs | `scorescorep-score` the devel profile, then set `SCOREP_TOTAL_MEMORY` accordingly for proper-size runs. |
