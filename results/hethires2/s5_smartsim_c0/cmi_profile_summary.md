# s5_smartsim_c0 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 216.381 | 228.314 | 273.360 | 0.002 | 1.0 | 48/48 |
| `solver_ml_provider_call` | 216.379 | 228.300 | 273.358 | 0.005 | 1.0 | 48/48 |
| `app_provider_inference` | 213.050 | 225.408 | 269.712 | 0.003 | 1.0 | 48/48 |
| `smartsim_library_static_step` | 213.047 | 225.405 | 269.710 | 0.100 | 1.0 | 48/48 |
| `smartsim_put_tensor` | 166.224 | 191.254 | 233.788 | 166.224 | 2.0 | 48/48 |
| `MPI_Allreduce` | 76.044 | 64.790 | 136.992 | 76.044 | 36.9 | 48/48 |
| `smartsim_run_model` | 37.914 | 27.596 | 50.126 | 37.914 | 1.0 | 48/48 |
| `smartsim_unpack_tensor` | 8.807 | 6.299 | 19.470 | 8.807 | 1.0 | 48/48 |
| `app_prepare_input` | 3.088 | 2.647 | 3.508 | 3.088 | 1.0 | 48/48 |
| `solver_step_compute` | 2.513 | 1.738 | 3.387 | 2.513 | 1.0 | 48/48 |
| `MPI_Bcast` | 1.047 | 0.046 | 1.530 | 1.047 | 2.0 | 48/48 |
| `MPI_Waitall` | 0.291 | 0.344 | 0.762 | 0.291 | 2.0 | 48/48 |
| `app_finalize_output` | 0.235 | 0.238 | 0.275 | 0.235 | 1.0 | 48/48 |
| `MPI_Gatherv` | 0.097 | 0.135 | 0.179 | 0.097 | 0.0 | 48/48 |
| `MPI_Irecv` | 0.019 | 0.020 | 0.037 | 0.019 | 8.1 | 48/48 |
| `MPI_Isend` | 0.018 | 0.026 | 0.042 | 0.018 | 8.1 | 48/48 |
| `smartsim_chunk_plan` | 0.002 | 0.003 | 0.004 | 0.002 | 1.0 | 48/48 |
| `MPI_Gather` | 0.000 | 0.002 | 0.002 | 0.000 | 0.0 | 48/48 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 35665.894 | 1.0 | 48/48 |
| `solver_setup` | 4716.120 | 1.0 | 48/48 |
| `solver_step_ml_warmup` | 527.619 | 1.0 | 48/48 |
| `solver_teardown` | 34.116 | 1.0 | 48/48 |
