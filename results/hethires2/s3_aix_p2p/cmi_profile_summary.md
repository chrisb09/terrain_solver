# s3_aix_p2p (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 135.221 | 144.220 | 144.220 | 0.006 | 1.0 | 24/24 |
| `solver_ml_provider_call` | 135.215 | 144.107 | 144.107 | 0.004 | 1.0 | 24/24 |
| `app_provider_inference` | 118.105 | 127.838 | 127.838 | 0.003 | 1.0 | 24/24 |
| `aix_library_static_step` | 118.102 | 127.836 | 127.836 | 0.001 | 1.0 | 24/24 |
| `aix_inference` | 118.101 | 127.835 | 127.835 | 0.020 | 1.0 | 24/24 |
| `inference` | 118.081 | 127.818 | 127.818 | 0.009 | 1.0 | 24/24 |
| `p2p_pipelined_exchange` | 118.067 | 127.803 | 127.803 | 1.081 | 1.0 | 24/24 |
| `MPI_Recv` | 82.798 | 0.000 | 106.153 | 82.798 | 2.9 | 23/24 |
| `MPI_Send` | 29.928 | 0.000 | 39.421 | 29.928 | 1.0 | 23/24 |
| `app_prepare_input` | 16.353 | 15.810 | 17.027 | 16.353 | 1.0 | 24/24 |
| `MPI_Allreduce` | 10.832 | 2.171 | 14.518 | 10.832 | 36.9 | 24/24 |
| `solver_step_compute` | 5.640 | 5.301 | 5.994 | 5.640 | 1.0 | 24/24 |
| `MPI_Testsome` | 4.239 | 101.738 | 101.738 | 4.239 | 47.0 | 1/24 |
| `app_finalize_output` | 0.752 | 0.454 | 1.338 | 0.752 | 1.0 | 24/24 |
| `MPI_Bcast` | 0.726 | 0.012 | 0.782 | 0.726 | 2.0 | 24/24 |
| `MPI_Waitall` | 0.109 | 0.078 | 0.182 | 0.109 | 2.0 | 24/24 |
| `MPI_Gatherv` | 0.079 | 0.059 | 0.108 | 0.079 | 0.0 | 24/24 |
| `MPI_Isend` | 0.020 | 0.020 | 0.025 | 0.020 | 8.1 | 24/24 |
| `MPI_Waitall` | 0.016 | 0.377 | 0.377 | 0.016 | 0.1 | 1/24 |
| `MPI_Irecv` | 0.012 | 0.014 | 0.014 | 0.012 | 8.1 | 24/24 |
| `aix_collective_setup` | 0.004 | 0.004 | 0.006 | 0.004 | 1.0 | 24/24 |
| `MPI_Isend` | 0.003 | 0.075 | 0.075 | 0.003 | 2.9 | 1/24 |
| `MPI_Gather` | 0.002 | 0.001 | 0.004 | 0.002 | 0.0 | 24/24 |
| `MPI_Irecv` | 0.001 | 0.029 | 0.029 | 0.001 | 1.0 | 1/24 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 18254.483 | 1.0 | 24/24 |
| `solver_setup` | 1816.200 | 1.0 | 24/24 |
| `solver_step_ml_warmup` | 364.091 | 1.0 | 24/24 |
| `solver_teardown` | 49.393 | 1.0 | 24/24 |
