# s4_aix_p2p (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 146.004 | 145.346 | 147.431 | 0.005 | 1.0 | 25/25 |
| `solver_ml_provider_call` | 145.999 | 145.246 | 147.430 | 0.004 | 1.0 | 25/25 |
| `app_provider_inference` | 131.111 | 130.651 | 141.825 | 0.001 | 1.0 | 25/25 |
| `aix_library_static_step` | 131.109 | 130.649 | 141.823 | 0.001 | 1.0 | 25/25 |
| `aix_inference` | 131.109 | 130.648 | 141.823 | 0.016 | 1.0 | 25/25 |
| `inference` | 131.092 | 130.629 | 141.817 | 0.009 | 1.0 | 25/25 |
| `p2p_pipelined_exchange` | 131.080 | 130.615 | 141.806 | 1.184 | 1.0 | 25/25 |
| `MPI_Recv` | 100.815 | 124.942 | 124.942 | 100.815 | 2.9 | 24/25 |
| `MPI_Send` | 24.571 | 5.649 | 41.679 | 24.571 | 1.0 | 24/25 |
| `app_prepare_input` | 14.385 | 14.154 | 15.642 | 14.385 | 1.0 | 25/25 |
| `MPI_Allreduce` | 5.938 | 7.129 | 7.129 | 5.938 | 36.9 | 25/25 |
| `solver_step_compute` | 5.235 | 4.789 | 5.539 | 5.235 | 1.0 | 25/25 |
| `MPI_Testsome` | 4.336 | 0.000 | 108.411 | 4.336 | 205.7 | 1/25 |
| `MPI_Bcast` | 0.872 | 0.009 | 0.940 | 0.872 | 2.0 | 25/25 |
| `app_finalize_output` | 0.499 | 0.437 | 0.720 | 0.499 | 1.0 | 25/25 |
| `MPI_Waitall` | 0.165 | 0.138 | 0.464 | 0.165 | 2.0 | 25/25 |
| `MPI_Waitall` | 0.161 | 0.000 | 4.034 | 0.161 | 0.1 | 1/25 |
| `MPI_Gatherv` | 0.075 | 0.059 | 0.109 | 0.075 | 0.0 | 25/25 |
| `MPI_Irecv` | 0.014 | 0.014 | 0.024 | 0.014 | 8.1 | 25/25 |
| `MPI_Isend` | 0.012 | 0.000 | 0.288 | 0.012 | 2.9 | 1/25 |
| `MPI_Isend` | 0.011 | 0.007 | 0.031 | 0.011 | 8.1 | 25/25 |
| `aix_collective_setup` | 0.003 | 0.004 | 0.005 | 0.003 | 1.0 | 25/25 |
| `MPI_Gather` | 0.001 | 0.004 | 0.004 | 0.001 | 0.0 | 25/25 |
| `MPI_Irecv` | 0.001 | 0.000 | 0.024 | 0.001 | 1.0 | 1/25 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 18644.836 | 1.0 | 25/25 |
| `solver_step_ml_warmup` | 1252.392 | 1.0 | 25/25 |
| `solver_setup` | 1003.105 | 1.0 | 25/25 |
| `solver_teardown` | 70.917 | 1.0 | 25/25 |
