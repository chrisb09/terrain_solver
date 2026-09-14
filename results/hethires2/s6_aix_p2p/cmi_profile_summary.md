# s6_aix_p2p (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 69.049 | 70.601 | 70.697 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 69.047 | 70.514 | 70.696 | 0.004 | 1.0 | 96/96 |
| `app_provider_inference` | 60.977 | 63.639 | 65.327 | 0.001 | 1.0 | 96/96 |
| `aix_library_static_step` | 60.975 | 63.638 | 65.325 | 0.001 | 1.0 | 96/96 |
| `aix_inference` | 60.974 | 63.637 | 65.325 | 0.014 | 1.0 | 96/96 |
| `inference` | 60.961 | 63.628 | 65.313 | 0.008 | 1.0 | 96/96 |
| `p2p_pipelined_exchange` | 60.949 | 63.613 | 65.303 | 0.839 | 1.0 | 96/96 |
| `MPI_Recv` | 42.882 | 0.000 | 60.905 | 42.882 | 2.9 | 92/96 |
| `MPI_Allreduce` | 20.088 | 16.902 | 21.154 | 20.088 | 36.9 | 96/96 |
| `MPI_Send` | 15.389 | 0.000 | 22.609 | 15.389 | 1.0 | 92/96 |
| `app_prepare_input` | 7.912 | 6.674 | 11.230 | 7.912 | 1.0 | 96/96 |
| `MPI_Testsome` | 1.826 | 44.947 | 44.947 | 1.826 | 272.9 | 4/96 |
| `MPI_Bcast` | 1.575 | 0.018 | 1.779 | 1.575 | 2.0 | 96/96 |
| `solver_step_compute` | 1.261 | 1.485 | 1.947 | 1.261 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.187 | 0.121 | 0.248 | 0.187 | 0.0 | 96/96 |
| `app_finalize_output` | 0.154 | 0.141 | 0.220 | 0.154 | 1.0 | 96/96 |
| `MPI_Waitall` | 0.141 | 0.362 | 0.603 | 0.141 | 2.0 | 96/96 |
| `MPI_Isend` | 0.024 | 0.027 | 0.126 | 0.024 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.014 | 0.018 | 0.089 | 0.014 | 8.1 | 96/96 |
| `MPI_Waitall` | 0.006 | 0.127 | 0.217 | 0.006 | 0.1 | 4/96 |
| `MPI_Isend` | 0.004 | 0.099 | 0.103 | 0.004 | 2.9 | 4/96 |
| `aix_collective_setup` | 0.003 | 0.004 | 0.005 | 0.003 | 1.0 | 96/96 |
| `MPI_Gather` | 0.002 | 0.044 | 0.044 | 0.002 | 0.0 | 96/96 |
| `MPI_Irecv` | 0.001 | 0.035 | 0.038 | 0.001 | 1.0 | 4/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 69298.899 | 1.0 | 96/96 |
| `solver_setup` | 57167.483 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 466.616 | 1.0 | 96/96 |
| `solver_teardown` | 68.311 | 1.0 | 96/96 |
