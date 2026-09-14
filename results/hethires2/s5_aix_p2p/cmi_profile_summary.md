# s5_aix_p2p (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 139.458 | 144.072 | 146.140 | 0.004 | 1.0 | 48/48 |
| `solver_ml_provider_call` | 139.454 | 143.973 | 146.138 | 0.003 | 1.0 | 48/48 |
| `app_provider_inference` | 131.274 | 137.048 | 138.980 | 0.001 | 1.0 | 48/48 |
| `aix_library_static_step` | 131.273 | 137.047 | 138.978 | 0.001 | 1.0 | 48/48 |
| `aix_inference` | 131.272 | 137.046 | 138.977 | 0.014 | 1.0 | 48/48 |
| `inference` | 131.258 | 137.028 | 138.958 | 0.007 | 1.0 | 48/48 |
| `p2p_pipelined_exchange` | 131.247 | 137.013 | 138.943 | 0.458 | 1.0 | 48/48 |
| `MPI_Recv` | 115.960 | 76.761 | 134.076 | 115.960 | 2.9 | 47/48 |
| `MPI_Allreduce` | 17.075 | 13.100 | 23.084 | 17.075 | 36.9 | 48/48 |
| `MPI_Send` | 12.400 | 60.223 | 60.223 | 12.400 | 1.0 | 47/48 |
| `app_prepare_input` | 7.874 | 6.682 | 9.445 | 7.874 | 1.0 | 48/48 |
| `MPI_Testsome` | 2.363 | 0.000 | 113.410 | 2.363 | 5.3 | 1/48 |
| `solver_step_compute` | 2.352 | 1.701 | 3.160 | 2.352 | 1.0 | 48/48 |
| `MPI_Bcast` | 0.894 | 0.030 | 1.055 | 0.894 | 2.0 | 48/48 |
| `app_finalize_output` | 0.304 | 0.239 | 0.757 | 0.304 | 1.0 | 48/48 |
| `MPI_Waitall` | 0.188 | 0.228 | 0.333 | 0.188 | 2.0 | 48/48 |
| `MPI_Gatherv` | 0.107 | 0.163 | 0.210 | 0.107 | 0.0 | 48/48 |
| `MPI_Waitall` | 0.056 | 0.000 | 2.683 | 0.056 | 0.1 | 1/48 |
| `MPI_Irecv` | 0.014 | 0.017 | 0.018 | 0.014 | 8.1 | 48/48 |
| `MPI_Isend` | 0.012 | 0.019 | 0.025 | 0.012 | 8.1 | 48/48 |
| `MPI_Isend` | 0.008 | 0.000 | 0.402 | 0.008 | 2.9 | 1/48 |
| `aix_collective_setup` | 0.003 | 0.004 | 0.005 | 0.003 | 1.0 | 48/48 |
| `MPI_Irecv` | 0.003 | 0.000 | 0.130 | 0.003 | 1.0 | 1/48 |
| `MPI_Gather` | 0.001 | 0.003 | 0.003 | 0.001 | 0.0 | 48/48 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 18955.167 | 1.0 | 48/48 |
| `solver_setup` | 1870.278 | 1.0 | 48/48 |
| `solver_step_ml_warmup` | 430.481 | 1.0 | 48/48 |
| `solver_teardown` | 63.401 | 1.0 | 48/48 |
