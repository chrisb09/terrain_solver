# s1_aix_p2p_fullcredits (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 150.750 | 158.776 | 158.776 | 0.024 | 1.0 | 4/4 |
| `solver_ml_provider_call` | 150.726 | 158.683 | 158.683 | 0.003 | 1.0 | 4/4 |
| `app_provider_inference` | 112.691 | 121.941 | 121.941 | 0.001 | 1.0 | 4/4 |
| `aix_library_static_step` | 112.689 | 121.940 | 121.940 | 0.000 | 1.0 | 4/4 |
| `aix_inference` | 112.689 | 121.940 | 121.940 | 0.007 | 1.0 | 4/4 |
| `inference` | 112.682 | 121.932 | 121.932 | 0.006 | 1.0 | 4/4 |
| `p2p_pipelined_exchange` | 112.675 | 121.923 | 121.923 | 15.002 | 1.0 | 4/4 |
| `MPI_Send` | 64.276 | 0.000 | 86.873 | 64.276 | 0.8 | 3/4 |
| `app_prepare_input` | 35.508 | 34.000 | 37.130 | 35.508 | 1.0 | 4/4 |
| `solver_step_compute` | 21.522 | 21.399 | 21.748 | 21.522 | 1.0 | 4/4 |
| `MPI_Recv` | 17.909 | 0.000 | 33.716 | 17.909 | 2.2 | 3/4 |
| `MPI_Testsome` | 15.160 | 60.641 | 60.641 | 15.160 | 160.7 | 1/4 |
| `MPI_Allreduce` | 8.659 | 0.676 | 21.295 | 8.659 | 36.9 | 4/4 |
| `app_finalize_output` | 2.524 | 2.739 | 2.739 | 2.524 | 1.0 | 4/4 |
| `MPI_Bcast` | 0.495 | 0.006 | 0.666 | 0.495 | 2.0 | 4/4 |
| `MPI_Waitall` | 0.321 | 1.282 | 1.282 | 0.321 | 0.8 | 1/4 |
| `MPI_Gatherv` | 0.070 | 0.041 | 0.090 | 0.070 | 0.0 | 4/4 |
| `MPI_Waitall` | 0.062 | 0.079 | 0.087 | 0.062 | 2.0 | 4/4 |
| `MPI_Isend` | 0.014 | 0.016 | 0.016 | 0.014 | 8.1 | 4/4 |
| `MPI_Irecv` | 0.014 | 0.016 | 0.016 | 0.014 | 8.1 | 4/4 |
| `MPI_Isend` | 0.005 | 0.021 | 0.021 | 0.005 | 2.2 | 1/4 |
| `aix_collective_setup` | 0.002 | 0.002 | 0.002 | 0.002 | 1.0 | 4/4 |
| `MPI_Irecv` | 0.001 | 0.006 | 0.006 | 0.001 | 0.8 | 1/4 |
| `MPI_Gather` | 0.001 | 0.000 | 0.003 | 0.001 | 0.0 | 4/4 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 33925.175 | 1.0 | 4/4 |
| `solver_step_ml_warmup` | 12429.575 | 1.0 | 4/4 |
| `solver_setup` | 2767.203 | 1.0 | 4/4 |
| `solver_teardown` | 49.412 | 1.0 | 4/4 |
