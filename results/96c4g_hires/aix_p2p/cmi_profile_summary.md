# aix_p2p (96c4g_hires)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 66.182 | 66.960 | 67.666 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 66.180 | 66.840 | 67.665 | 0.003 | 1.0 | 96/96 |
| `app_provider_inference` | 58.562 | 62.419 | 63.519 | 0.001 | 1.0 | 96/96 |
| `aix_library_static_step` | 58.561 | 62.417 | 63.517 | 0.001 | 1.0 | 96/96 |
| `aix_inference` | 58.560 | 62.416 | 63.517 | 0.011 | 1.0 | 96/96 |
| `inference` | 58.549 | 62.407 | 63.507 | 0.008 | 1.0 | 96/96 |
| `p2p_pipelined_exchange` | 58.538 | 62.391 | 63.492 | 0.844 | 1.0 | 96/96 |
| `MPI_Recv` | 40.795 | 0.000 | 59.080 | 40.795 | 2.9 | 92/96 |
| `MPI_Send` | 15.125 | 0.000 | 25.930 | 15.125 | 1.0 | 92/96 |
| `MPI_Allreduce` | 13.309 | 12.012 | 14.687 | 13.309 | 36.9 | 96/96 |
| `app_prepare_input` | 7.458 | 4.292 | 11.179 | 7.458 | 1.0 | 96/96 |
| `MPI_Testsome` | 1.762 | 42.990 | 42.990 | 1.762 | 276.6 | 4/96 |
| `MPI_Bcast` | 1.475 | 0.071 | 1.682 | 1.475 | 2.0 | 96/96 |
| `solver_step_compute` | 1.270 | 1.251 | 2.066 | 1.270 | 1.0 | 96/96 |
| `app_finalize_output` | 0.157 | 0.125 | 0.222 | 0.157 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.145 | 0.116 | 0.203 | 0.145 | 0.0 | 96/96 |
| `MPI_Waitall` | 0.141 | 0.158 | 0.337 | 0.141 | 2.0 | 96/96 |
| `MPI_Isend` | 0.026 | 0.029 | 0.125 | 0.026 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.015 | 0.017 | 0.068 | 0.015 | 8.1 | 96/96 |
| `MPI_Waitall` | 0.006 | 0.177 | 0.177 | 0.006 | 0.1 | 4/96 |
| `MPI_Isend` | 0.004 | 0.102 | 0.107 | 0.004 | 2.9 | 4/96 |
| `aix_collective_setup` | 0.003 | 0.004 | 0.020 | 0.003 | 1.0 | 96/96 |
| `MPI_Irecv` | 0.002 | 0.044 | 0.051 | 0.002 | 1.0 | 4/96 |
| `MPI_Gather` | 0.000 | 0.003 | 0.003 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 68732.906 | 1.0 | 96/96 |
| `solver_setup` | 57638.067 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 643.216 | 1.0 | 96/96 |
| `solver_teardown` | 75.927 | 1.0 | 96/96 |
