# s2_aix_p2p (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 175.577 | 175.347 | 176.078 | 0.015 | 1.0 | 5/5 |
| `solver_ml_provider_call` | 175.562 | 175.277 | 176.077 | 0.003 | 1.0 | 5/5 |
| `app_provider_inference` | 147.246 | 146.491 | 148.282 | 0.001 | 1.0 | 5/5 |
| `aix_library_static_step` | 147.245 | 146.490 | 148.281 | 0.000 | 1.0 | 5/5 |
| `aix_inference` | 147.244 | 146.490 | 148.280 | 0.007 | 1.0 | 5/5 |
| `inference` | 147.237 | 146.483 | 148.274 | 0.006 | 1.0 | 5/5 |
| `p2p_pipelined_exchange` | 147.228 | 146.475 | 148.264 | 8.793 | 1.0 | 5/5 |
| `MPI_Recv` | 82.814 | 95.997 | 108.397 | 82.814 | 2.4 | 4/5 |
| `MPI_Send` | 34.744 | 50.456 | 50.456 | 34.744 | 0.8 | 4/5 |
| `app_prepare_input` | 26.003 | 26.460 | 26.460 | 26.003 | 1.0 | 5/5 |
| `MPI_Testsome` | 20.146 | 0.000 | 100.730 | 20.146 | 3971.6 | 1/5 |
| `solver_step_compute` | 17.179 | 14.373 | 18.422 | 17.179 | 1.0 | 5/5 |
| `MPI_Allreduce` | 4.465 | 7.601 | 7.601 | 4.465 | 36.9 | 5/5 |
| `app_finalize_output` | 2.311 | 2.324 | 2.474 | 2.311 | 1.0 | 5/5 |
| `MPI_Waitall` | 0.716 | 0.000 | 3.580 | 0.716 | 0.6 | 1/5 |
| `MPI_Bcast` | 0.540 | 0.006 | 0.690 | 0.540 | 2.0 | 5/5 |
| `MPI_Waitall` | 0.139 | 0.194 | 0.194 | 0.139 | 2.0 | 5/5 |
| `MPI_Gatherv` | 0.073 | 0.058 | 0.092 | 0.073 | 0.0 | 5/5 |
| `MPI_Isend` | 0.020 | 0.037 | 0.037 | 0.020 | 8.1 | 5/5 |
| `MPI_Irecv` | 0.016 | 0.015 | 0.021 | 0.016 | 8.1 | 5/5 |
| `MPI_Isend` | 0.014 | 0.000 | 0.072 | 0.014 | 2.4 | 1/5 |
| `aix_collective_setup` | 0.002 | 0.002 | 0.002 | 0.002 | 1.0 | 5/5 |
| `MPI_Irecv` | 0.001 | 0.000 | 0.006 | 0.001 | 0.8 | 1/5 |
| `MPI_Gather` | 0.001 | 0.000 | 0.003 | 0.001 | 0.0 | 5/5 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 35272.440 | 1.0 | 5/5 |
| `solver_step_ml_warmup` | 6243.238 | 1.0 | 5/5 |
| `solver_setup` | 1057.666 | 1.0 | 5/5 |
| `solver_teardown` | 79.482 | 1.0 | 5/5 |
