# s7_aix_p2p_fullcredits (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `MPI_Allreduce` | 59.014 | 47.015 | 64.291 | 59.014 | 36.9 | 192/192 |
| `solver_step_ml_steady` | 54.486 | 56.376 | 58.352 | 0.003 | 1.0 | 192/192 |
| `solver_ml_provider_call` | 54.483 | 56.059 | 58.350 | 0.002 | 1.0 | 192/192 |
| `app_provider_inference` | 52.987 | 54.978 | 57.166 | 0.001 | 1.0 | 192/192 |
| `aix_library_static_step` | 52.986 | 54.977 | 57.166 | 0.001 | 1.0 | 192/192 |
| `aix_inference` | 52.985 | 54.823 | 57.165 | 0.007 | 1.0 | 192/192 |
| `inference` | 52.978 | 54.817 | 57.159 | 0.005 | 1.0 | 192/192 |
| `p2p_pipelined_exchange` | 52.972 | 54.809 | 57.153 | 0.252 | 1.0 | 192/192 |
| `MPI_Recv` | 39.688 | 31.909 | 53.679 | 39.688 | 2.9 | 188/192 |
| `MPI_Send` | 12.118 | 22.878 | 33.087 | 12.118 | 1.0 | 188/192 |
| `MPI_Bcast` | 8.818 | 0.002 | 12.814 | 8.818 | 2.0 | 192/192 |
| `app_prepare_input` | 1.429 | 0.948 | 2.750 | 1.429 | 1.0 | 192/192 |
| `MPI_Testsome` | 0.882 | 0.000 | 44.191 | 0.882 | 13.0 | 4/192 |
| `solver_step_compute` | 0.587 | 0.726 | 1.082 | 0.587 | 1.0 | 192/192 |
| `MPI_Waitall` | 0.275 | 4.471 | 7.903 | 0.275 | 2.0 | 192/192 |
| `MPI_Gatherv` | 0.144 | 0.298 | 0.345 | 0.144 | 0.0 | 192/192 |
| `app_finalize_output` | 0.065 | 0.063 | 0.104 | 0.065 | 1.0 | 192/192 |
| `MPI_Waitall` | 0.021 | 0.000 | 1.288 | 0.021 | 0.1 | 4/192 |
| `MPI_Irecv` | 0.012 | 0.017 | 0.045 | 0.012 | 8.1 | 192/192 |
| `MPI_Isend` | 0.010 | 0.021 | 0.216 | 0.010 | 8.1 | 192/192 |
| `MPI_Isend` | 0.009 | 0.000 | 0.430 | 0.009 | 2.9 | 4/192 |
| `aix_collective_setup` | 0.002 | 0.002 | 0.003 | 0.002 | 1.0 | 192/192 |
| `MPI_Irecv` | 0.001 | 0.000 | 0.061 | 0.001 | 1.0 | 4/192 |
| `MPI_Gather` | 0.000 | 0.004 | 0.004 | 0.000 | 0.0 | 192/192 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 75065.066 | 1.0 | 192/192 |
| `solver_setup` | 57030.364 | 1.0 | 192/192 |
| `solver_step_ml_warmup` | 388.646 | 1.0 | 192/192 |
| `solver_teardown` | 62.706 | 1.0 | 192/192 |
