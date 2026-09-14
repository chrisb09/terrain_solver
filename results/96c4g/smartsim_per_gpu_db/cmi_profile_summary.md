# smartsim_per_gpu_db (96c4g)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 109.943 | 85.247 | 134.640 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 109.941 | 85.233 | 134.639 | 0.004 | 1.0 | 96/96 |
| `app_provider_inference` | 109.657 | 85.024 | 134.342 | 0.001 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 109.656 | 85.022 | 134.341 | 0.179 | 1.0 | 96/96 |
| `MPI_Allreduce` | 72.556 | 96.433 | 112.179 | 72.556 | 44.6 | 96/96 |
| `smartsim_run_model` | 52.786 | 38.038 | 77.413 | 52.786 | 1.0 | 96/96 |
| `smartsim_put_tensor` | 46.557 | 37.502 | 65.630 | 46.557 | 2.0 | 96/96 |
| `smartsim_unpack_tensor` | 10.133 | 9.286 | 17.601 | 10.133 | 1.0 | 96/96 |
| `MPI_Bcast` | 2.167 | 0.016 | 2.581 | 2.167 | 2.4 | 96/96 |
| `MPI_Waitall` | 0.415 | 0.472 | 2.129 | 0.415 | 2.2 | 96/96 |
| `solver_step_compute` | 0.385 | 0.178 | 0.632 | 0.385 | 1.1 | 96/96 |
| `app_prepare_input` | 0.249 | 0.180 | 0.735 | 0.249 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.148 | 0.214 | 0.214 | 0.148 | 0.1 | 96/96 |
| `MPI_Isend` | 0.036 | 0.030 | 0.044 | 0.036 | 8.8 | 96/96 |
| `app_finalize_output` | 0.031 | 0.025 | 0.054 | 0.031 | 1.0 | 96/96 |
| `MPI_Irecv` | 0.015 | 0.014 | 0.017 | 0.015 | 8.8 | 96/96 |
| `MPI_Gather` | 0.001 | 0.019 | 0.019 | 0.001 | 0.1 | 96/96 |
| `smartsim_chunk_plan` | 0.001 | 0.001 | 0.002 | 0.001 | 1.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 9612.608 | 1.0 | 96/96 |
| `solver_setup` | 5308.356 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 365.421 | 1.0 | 96/96 |
| `solver_teardown` | 28.261 | 1.0 | 96/96 |
