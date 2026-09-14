# s6_aix_p2p_fullcredits (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 64.602 | 66.576 | 66.576 | 0.003 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 64.599 | 66.436 | 66.436 | 0.004 | 1.0 | 96/96 |
| `app_provider_inference` | 56.378 | 59.511 | 60.607 | 0.001 | 1.0 | 96/96 |
| `aix_library_static_step` | 56.377 | 59.509 | 60.606 | 0.001 | 1.0 | 96/96 |
| `aix_inference` | 56.376 | 59.508 | 60.605 | 0.014 | 1.0 | 96/96 |
| `inference` | 56.362 | 59.498 | 60.593 | 0.008 | 1.0 | 96/96 |
| `p2p_pipelined_exchange` | 56.351 | 59.485 | 60.584 | 0.708 | 1.0 | 96/96 |
| `MPI_Send` | 42.385 | 0.000 | 51.599 | 42.385 | 1.0 | 92/96 |
| `MPI_Allreduce` | 22.402 | 18.650 | 24.253 | 22.402 | 36.9 | 96/96 |
| `MPI_Recv` | 11.477 | 0.000 | 29.456 | 11.477 | 2.9 | 92/96 |
| `app_prepare_input` | 8.056 | 6.765 | 11.021 | 8.056 | 1.0 | 96/96 |
| `MPI_Testsome` | 1.769 | 43.348 | 43.348 | 1.769 | 12.5 | 4/96 |
| `MPI_Bcast` | 1.284 | 0.062 | 1.535 | 1.284 | 2.0 | 96/96 |
| `solver_step_compute` | 1.280 | 1.838 | 1.848 | 1.280 | 1.0 | 96/96 |
| `app_finalize_output` | 0.162 | 0.156 | 0.216 | 0.162 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.154 | 0.126 | 0.216 | 0.154 | 0.0 | 96/96 |
| `MPI_Waitall` | 0.137 | 0.297 | 0.519 | 0.137 | 2.0 | 96/96 |
| `MPI_Isend` | 0.023 | 0.120 | 0.120 | 0.023 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.015 | 0.018 | 0.122 | 0.015 | 8.1 | 96/96 |
| `MPI_Waitall` | 0.006 | 0.136 | 0.142 | 0.006 | 0.1 | 4/96 |
| `MPI_Isend` | 0.004 | 0.107 | 0.107 | 0.004 | 2.9 | 4/96 |
| `aix_collective_setup` | 0.003 | 0.003 | 0.005 | 0.003 | 1.0 | 96/96 |
| `MPI_Irecv` | 0.001 | 0.037 | 0.037 | 0.001 | 1.0 | 4/96 |
| `MPI_Gather` | 0.000 | 0.002 | 0.002 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 68570.691 | 1.0 | 96/96 |
| `solver_setup` | 56803.589 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 471.966 | 1.0 | 96/96 |
| `solver_teardown` | 61.475 | 1.0 | 96/96 |
