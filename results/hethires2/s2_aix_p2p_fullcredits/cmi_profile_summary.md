# s2_aix_p2p_fullcredits (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 172.744 | 172.389 | 173.345 | 0.016 | 1.0 | 5/5 |
| `solver_ml_provider_call` | 172.728 | 172.315 | 173.343 | 0.003 | 1.0 | 5/5 |
| `app_provider_inference` | 144.346 | 143.386 | 145.431 | 0.001 | 1.0 | 5/5 |
| `aix_library_static_step` | 144.345 | 143.385 | 145.430 | 0.001 | 1.0 | 5/5 |
| `aix_inference` | 144.345 | 143.384 | 145.430 | 0.008 | 1.0 | 5/5 |
| `inference` | 144.337 | 143.376 | 145.423 | 0.006 | 1.0 | 5/5 |
| `p2p_pipelined_exchange` | 144.328 | 143.368 | 145.414 | 10.798 | 1.0 | 5/5 |
| `MPI_Send` | 80.006 | 94.342 | 110.626 | 80.006 | 0.8 | 4/5 |
| `MPI_Recv` | 35.220 | 49.001 | 52.315 | 35.220 | 2.4 | 4/5 |
| `app_prepare_input` | 25.983 | 26.316 | 26.322 | 25.983 | 1.0 | 5/5 |
| `MPI_Testsome` | 17.532 | 0.000 | 87.659 | 17.532 | 121.6 | 1/5 |
| `solver_step_compute` | 17.235 | 14.448 | 18.601 | 17.235 | 1.0 | 5/5 |
| `MPI_Allreduce` | 4.531 | 7.772 | 7.772 | 4.531 | 36.9 | 5/5 |
| `app_finalize_output` | 2.397 | 2.610 | 2.610 | 2.397 | 1.0 | 5/5 |
| `MPI_Waitall` | 0.757 | 0.000 | 3.787 | 0.757 | 0.6 | 1/5 |
| `MPI_Bcast` | 0.538 | 0.006 | 0.698 | 0.538 | 2.0 | 5/5 |
| `MPI_Waitall` | 0.133 | 0.184 | 0.184 | 0.133 | 2.0 | 5/5 |
| `MPI_Gatherv` | 0.072 | 0.058 | 0.092 | 0.072 | 0.0 | 5/5 |
| `MPI_Isend` | 0.020 | 0.039 | 0.039 | 0.020 | 8.1 | 5/5 |
| `MPI_Irecv` | 0.015 | 0.015 | 0.019 | 0.015 | 8.1 | 5/5 |
| `MPI_Isend` | 0.014 | 0.000 | 0.071 | 0.014 | 2.4 | 1/5 |
| `aix_collective_setup` | 0.002 | 0.002 | 0.002 | 0.002 | 1.0 | 5/5 |
| `MPI_Irecv` | 0.001 | 0.000 | 0.006 | 0.001 | 0.8 | 1/5 |
| `MPI_Gather` | 0.001 | 0.001 | 0.003 | 0.001 | 0.0 | 5/5 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 23582.460 | 1.0 | 5/5 |
| `solver_setup` | 1046.366 | 1.0 | 5/5 |
| `solver_step_ml_warmup` | 405.162 | 1.0 | 5/5 |
| `solver_teardown` | 44.967 | 1.0 | 5/5 |
