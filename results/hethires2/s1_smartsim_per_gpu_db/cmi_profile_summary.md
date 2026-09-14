# s1_smartsim_per_gpu_db (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 379.355 | 360.531 | 394.004 | 0.008 | 1.0 | 4/4 |
| `solver_ml_provider_call` | 379.347 | 360.506 | 394.001 | 0.005 | 1.0 | 4/4 |
| `app_provider_inference` | 364.014 | 345.123 | 378.683 | 0.002 | 1.0 | 4/4 |
| `smartsim_library_static_step` | 364.012 | 345.121 | 378.682 | 0.157 | 1.0 | 4/4 |
| `smartsim_put_tensor` | 280.651 | 271.446 | 289.875 | 280.651 | 2.0 | 4/4 |
| `smartsim_run_model` | 76.665 | 67.090 | 81.640 | 76.665 | 1.0 | 4/4 |
| `solver_step_compute` | 22.539 | 23.126 | 23.126 | 22.539 | 1.0 | 4/4 |
| `MPI_Allreduce` | 16.389 | 34.661 | 34.661 | 16.389 | 36.9 | 4/4 |
| `app_prepare_input` | 12.302 | 12.356 | 12.356 | 12.302 | 1.0 | 4/4 |
| `smartsim_unpack_tensor` | 6.539 | 6.408 | 7.058 | 6.539 | 1.0 | 4/4 |
| `app_finalize_output` | 3.027 | 3.022 | 3.210 | 3.027 | 1.0 | 4/4 |
| `MPI_Bcast` | 0.479 | 0.006 | 0.652 | 0.479 | 2.0 | 4/4 |
| `MPI_Gatherv` | 0.071 | 0.045 | 0.091 | 0.071 | 0.0 | 4/4 |
| `MPI_Waitall` | 0.068 | 0.066 | 0.071 | 0.068 | 2.0 | 4/4 |
| `MPI_Isend` | 0.018 | 0.018 | 0.019 | 0.018 | 8.1 | 4/4 |
| `MPI_Irecv` | 0.018 | 0.018 | 0.018 | 0.018 | 8.1 | 4/4 |
| `smartsim_chunk_plan` | 0.001 | 0.001 | 0.001 | 0.001 | 1.0 | 4/4 |
| `MPI_Gather` | 0.000 | 0.000 | 0.001 | 0.000 | 0.0 | 4/4 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 55704.575 | 1.0 | 4/4 |
| `solver_setup` | 7257.980 | 1.0 | 4/4 |
| `solver_step_ml_warmup` | 5897.877 | 1.0 | 4/4 |
| `solver_teardown` | 159.014 | 1.0 | 4/4 |
