# aix_p2p_fullcredits (96c4g_hires)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 62.497 | 64.084 | 64.084 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 62.495 | 63.977 | 63.977 | 0.003 | 1.0 | 96/96 |
| `app_provider_inference` | 54.828 | 59.396 | 59.396 | 0.001 | 1.0 | 96/96 |
| `aix_library_static_step` | 54.827 | 59.394 | 59.394 | 0.001 | 1.0 | 96/96 |
| `aix_inference` | 54.826 | 59.394 | 59.394 | 0.011 | 1.0 | 96/96 |
| `inference` | 54.815 | 59.384 | 59.384 | 0.006 | 1.0 | 96/96 |
| `p2p_pipelined_exchange` | 54.805 | 59.369 | 59.369 | 0.686 | 1.0 | 96/96 |
| `MPI_Send` | 38.742 | 0.000 | 50.998 | 38.742 | 1.0 | 92/96 |
| `MPI_Recv` | 13.604 | 0.000 | 37.887 | 13.604 | 2.9 | 92/96 |
| `MPI_Allreduce` | 12.820 | 9.660 | 14.544 | 12.820 | 36.9 | 96/96 |
| `app_prepare_input` | 7.514 | 4.454 | 11.184 | 7.514 | 1.0 | 96/96 |
| `MPI_Bcast` | 1.795 | 0.123 | 1.990 | 1.795 | 2.0 | 96/96 |
| `MPI_Testsome` | 1.761 | 43.309 | 43.309 | 1.761 | 12.7 | 4/96 |
| `solver_step_compute` | 1.269 | 1.647 | 1.867 | 1.269 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.154 | 0.128 | 0.218 | 0.154 | 0.0 | 96/96 |
| `app_finalize_output` | 0.150 | 0.123 | 0.214 | 0.150 | 1.0 | 96/96 |
| `MPI_Waitall` | 0.111 | 0.115 | 0.352 | 0.111 | 2.0 | 96/96 |
| `MPI_Isend` | 0.026 | 0.077 | 0.080 | 0.026 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.014 | 0.074 | 0.074 | 0.014 | 8.1 | 96/96 |
| `MPI_Waitall` | 0.007 | 0.171 | 0.171 | 0.007 | 0.1 | 4/96 |
| `MPI_Isend` | 0.005 | 0.112 | 0.115 | 0.005 | 2.9 | 4/96 |
| `aix_collective_setup` | 0.003 | 0.004 | 0.042 | 0.003 | 1.0 | 96/96 |
| `MPI_Irecv` | 0.002 | 0.044 | 0.046 | 0.002 | 1.0 | 4/96 |
| `MPI_Gather` | 0.000 | 0.004 | 0.004 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 69991.044 | 1.0 | 96/96 |
| `solver_setup` | 57831.303 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 754.403 | 1.0 | 96/96 |
| `solver_teardown` | 124.069 | 1.0 | 96/96 |
