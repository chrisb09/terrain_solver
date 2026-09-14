# s7_aix_p2p (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `MPI_Allreduce` | 61.229 | 51.654 | 66.426 | 61.229 | 36.9 | 192/192 |
| `solver_step_ml_steady` | 58.884 | 60.003 | 63.602 | 0.003 | 1.0 | 192/192 |
| `solver_ml_provider_call` | 58.881 | 59.726 | 63.601 | 0.002 | 1.0 | 192/192 |
| `app_provider_inference` | 57.191 | 58.971 | 62.349 | 0.001 | 1.0 | 192/192 |
| `aix_library_static_step` | 57.190 | 58.970 | 62.348 | 0.000 | 1.0 | 192/192 |
| `aix_inference` | 57.190 | 58.970 | 62.348 | 0.006 | 1.0 | 192/192 |
| `inference` | 57.184 | 58.965 | 62.344 | 0.005 | 1.0 | 192/192 |
| `p2p_pipelined_exchange` | 57.177 | 58.958 | 62.338 | 0.279 | 1.0 | 192/192 |
| `MPI_Recv` | 53.281 | 56.831 | 60.857 | 53.281 | 2.9 | 188/192 |
| `MPI_Bcast` | 7.344 | 0.003 | 16.970 | 7.344 | 2.0 | 192/192 |
| `MPI_Send` | 2.656 | 2.113 | 7.217 | 2.656 | 1.0 | 188/192 |
| `app_prepare_input` | 1.620 | 0.692 | 3.306 | 1.620 | 1.0 | 192/192 |
| `MPI_Testsome` | 0.926 | 0.000 | 47.101 | 0.926 | 13.1 | 4/192 |
| `solver_step_compute` | 0.575 | 0.452 | 1.004 | 0.575 | 1.0 | 192/192 |
| `MPI_Waitall` | 0.278 | 1.950 | 7.201 | 0.278 | 2.0 | 192/192 |
| `MPI_Gatherv` | 0.141 | 0.315 | 0.360 | 0.141 | 0.0 | 192/192 |
| `app_finalize_output` | 0.069 | 0.060 | 0.113 | 0.069 | 1.0 | 192/192 |
| `MPI_Waitall` | 0.023 | 0.000 | 1.114 | 0.023 | 0.1 | 4/192 |
| `MPI_Irecv` | 0.012 | 0.018 | 0.020 | 0.012 | 8.1 | 192/192 |
| `MPI_Isend` | 0.010 | 0.000 | 0.505 | 0.010 | 2.9 | 4/192 |
| `MPI_Isend` | 0.010 | 0.040 | 0.045 | 0.010 | 8.1 | 192/192 |
| `aix_collective_setup` | 0.002 | 0.002 | 0.003 | 0.002 | 1.0 | 192/192 |
| `MPI_Irecv` | 0.001 | 0.000 | 0.067 | 0.001 | 1.0 | 4/192 |
| `MPI_Gather` | 0.000 | 0.004 | 0.004 | 0.000 | 0.0 | 192/192 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 76391.984 | 1.0 | 192/192 |
| `solver_setup` | 57749.958 | 1.0 | 192/192 |
| `solver_step_ml_warmup` | 486.364 | 1.0 | 192/192 |
| `solver_teardown` | 58.527 | 1.0 | 192/192 |
