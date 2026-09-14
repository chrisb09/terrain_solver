# smartsim_c0 (96c4g)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 96.846 | 94.376 | 104.778 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 96.845 | 94.357 | 104.777 | 0.003 | 1.0 | 96/96 |
| `app_provider_inference` | 96.565 | 94.150 | 104.488 | 0.002 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 96.563 | 94.148 | 104.487 | 0.860 | 1.0 | 96/96 |
| `smartsim_put_tensor` | 72.717 | 64.581 | 85.653 | 72.717 | 2.0 | 96/96 |
| `MPI_Allreduce` | 35.980 | 40.338 | 48.399 | 35.980 | 44.6 | 96/96 |
| `smartsim_run_model` | 20.136 | 25.035 | 27.925 | 20.136 | 1.0 | 96/96 |
| `MPI_Bcast` | 3.056 | 0.017 | 3.737 | 3.056 | 2.4 | 96/96 |
| `smartsim_unpack_tensor` | 2.850 | 3.593 | 7.053 | 2.850 | 1.0 | 96/96 |
| `MPI_Waitall` | 0.402 | 0.770 | 1.739 | 0.402 | 2.2 | 96/96 |
| `solver_step_compute` | 0.395 | 0.264 | 0.664 | 0.395 | 1.1 | 96/96 |
| `app_prepare_input` | 0.246 | 0.176 | 0.429 | 0.246 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.135 | 0.193 | 0.193 | 0.135 | 0.1 | 96/96 |
| `MPI_Isend` | 0.036 | 0.029 | 0.049 | 0.036 | 8.8 | 96/96 |
| `app_finalize_output` | 0.031 | 0.028 | 0.050 | 0.031 | 1.0 | 96/96 |
| `MPI_Irecv` | 0.015 | 0.014 | 0.017 | 0.015 | 8.8 | 96/96 |
| `MPI_Gather` | 0.001 | 0.033 | 0.033 | 0.001 | 0.1 | 96/96 |
| `smartsim_chunk_plan` | 0.001 | 0.001 | 0.002 | 0.001 | 1.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 14405.771 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 5327.528 | 1.0 | 96/96 |
| `solver_setup` | 4475.089 | 1.0 | 96/96 |
| `solver_teardown` | 3.476 | 1.0 | 96/96 |
