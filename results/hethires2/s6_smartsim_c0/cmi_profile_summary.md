# s6_smartsim_c0 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 271.934 | 273.023 | 277.056 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 271.932 | 273.002 | 277.054 | 0.005 | 1.0 | 96/96 |
| `app_provider_inference` | 270.215 | 271.339 | 275.376 | 0.003 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 270.211 | 271.335 | 275.373 | 0.105 | 1.0 | 96/96 |
| `smartsim_put_tensor` | 232.079 | 233.208 | 238.580 | 232.079 | 2.0 | 96/96 |
| `MPI_Allreduce` | 56.165 | 55.909 | 60.115 | 56.165 | 36.9 | 96/96 |
| `smartsim_run_model` | 34.388 | 33.232 | 40.172 | 34.388 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 3.638 | 4.753 | 6.996 | 3.638 | 1.0 | 96/96 |
| `app_prepare_input` | 1.590 | 1.535 | 1.666 | 1.590 | 1.0 | 96/96 |
| `MPI_Bcast` | 1.318 | 0.020 | 1.949 | 1.318 | 2.0 | 96/96 |
| `solver_step_compute` | 1.249 | 0.968 | 1.527 | 1.249 | 1.0 | 96/96 |
| `MPI_Waitall` | 0.477 | 0.780 | 1.543 | 0.477 | 2.0 | 96/96 |
| `app_finalize_output` | 0.122 | 0.123 | 0.172 | 0.122 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.092 | 0.076 | 0.130 | 0.092 | 0.0 | 96/96 |
| `MPI_Isend` | 0.027 | 0.026 | 0.030 | 0.027 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.019 | 0.019 | 0.022 | 0.019 | 8.1 | 96/96 |
| `smartsim_chunk_plan` | 0.003 | 0.003 | 0.038 | 0.003 | 1.0 | 96/96 |
| `MPI_Gather` | 0.000 | 0.002 | 0.002 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 38825.829 | 1.0 | 96/96 |
| `solver_setup` | 2719.430 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 543.673 | 1.0 | 96/96 |
| `solver_teardown` | 66.759 | 1.0 | 96/96 |
