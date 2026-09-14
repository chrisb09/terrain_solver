# s3_aix_p2p_fullcredits (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 147.458 | 162.096 | 162.096 | 0.006 | 1.0 | 24/24 |
| `solver_ml_provider_call` | 147.452 | 161.988 | 161.988 | 0.004 | 1.0 | 24/24 |
| `app_provider_inference` | 130.605 | 145.759 | 145.759 | 0.003 | 1.0 | 24/24 |
| `aix_library_static_step` | 130.602 | 145.757 | 145.757 | 0.001 | 1.0 | 24/24 |
| `aix_inference` | 130.601 | 145.756 | 145.756 | 0.020 | 1.0 | 24/24 |
| `inference` | 130.581 | 145.741 | 145.741 | 0.009 | 1.0 | 24/24 |
| `p2p_pipelined_exchange` | 130.568 | 145.727 | 145.727 | 1.820 | 1.0 | 24/24 |
| `MPI_Send` | 104.008 | 0.000 | 109.726 | 104.008 | 1.0 | 23/24 |
| `MPI_Recv` | 20.475 | 0.000 | 33.920 | 20.475 | 2.9 | 23/24 |
| `MPI_Allreduce` | 16.502 | 2.074 | 34.862 | 16.502 | 36.9 | 24/24 |
| `app_prepare_input` | 16.317 | 15.807 | 17.014 | 16.317 | 1.0 | 24/24 |
| `solver_step_compute` | 5.669 | 5.336 | 6.014 | 5.669 | 1.0 | 24/24 |
| `MPI_Testsome` | 4.251 | 102.029 | 102.029 | 4.251 | 47.0 | 1/24 |
| `MPI_Bcast` | 0.936 | 0.012 | 0.998 | 0.936 | 2.0 | 24/24 |
| `app_finalize_output` | 0.527 | 0.418 | 0.864 | 0.527 | 1.0 | 24/24 |
| `MPI_Waitall` | 0.117 | 0.078 | 0.272 | 0.117 | 2.0 | 24/24 |
| `MPI_Gatherv` | 0.079 | 0.055 | 0.105 | 0.079 | 0.0 | 24/24 |
| `MPI_Isend` | 0.021 | 0.042 | 0.042 | 0.021 | 8.1 | 24/24 |
| `MPI_Irecv` | 0.013 | 0.015 | 0.015 | 0.013 | 8.1 | 24/24 |
| `MPI_Waitall` | 0.009 | 0.223 | 0.223 | 0.009 | 0.1 | 1/24 |
| `aix_collective_setup` | 0.004 | 0.004 | 0.006 | 0.004 | 1.0 | 24/24 |
| `MPI_Isend` | 0.003 | 0.080 | 0.080 | 0.003 | 2.9 | 1/24 |
| `MPI_Gather` | 0.001 | 0.001 | 0.004 | 0.001 | 0.0 | 24/24 |
| `MPI_Irecv` | 0.001 | 0.029 | 0.029 | 0.001 | 1.0 | 1/24 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 20120.542 | 1.0 | 24/24 |
| `solver_setup` | 1841.618 | 1.0 | 24/24 |
| `solver_step_ml_warmup` | 371.479 | 1.0 | 24/24 |
| `solver_teardown` | 41.268 | 1.0 | 24/24 |
