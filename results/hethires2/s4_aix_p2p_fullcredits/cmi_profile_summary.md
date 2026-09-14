# s4_aix_p2p_fullcredits (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 173.560 | 173.329 | 175.269 | 0.006 | 1.0 | 25/25 |
| `solver_ml_provider_call` | 173.555 | 173.220 | 175.267 | 0.004 | 1.0 | 25/25 |
| `app_provider_inference` | 158.455 | 158.244 | 169.609 | 0.002 | 1.0 | 25/25 |
| `aix_library_static_step` | 158.453 | 158.243 | 169.609 | 0.001 | 1.0 | 25/25 |
| `aix_inference` | 158.452 | 158.242 | 169.608 | 0.020 | 1.0 | 25/25 |
| `inference` | 158.432 | 158.221 | 169.603 | 0.010 | 1.0 | 25/25 |
| `p2p_pipelined_exchange` | 158.418 | 158.204 | 169.593 | 1.761 | 1.0 | 25/25 |
| `MPI_Send` | 81.926 | 70.599 | 106.547 | 81.926 | 1.0 | 24/25 |
| `MPI_Recv` | 69.683 | 87.566 | 147.152 | 69.683 | 2.9 | 24/25 |
| `app_prepare_input` | 14.606 | 14.510 | 16.838 | 14.606 | 1.0 | 25/25 |
| `MPI_Allreduce` | 6.263 | 6.988 | 7.299 | 6.263 | 36.9 | 25/25 |
| `solver_step_compute` | 5.208 | 4.817 | 5.497 | 5.208 | 1.0 | 25/25 |
| `MPI_Testsome` | 4.862 | 0.000 | 121.554 | 4.862 | 146.1 | 1/25 |
| `MPI_Bcast` | 0.876 | 0.010 | 1.012 | 0.876 | 2.0 | 25/25 |
| `app_finalize_output` | 0.489 | 0.461 | 0.611 | 0.489 | 1.0 | 25/25 |
| `MPI_Waitall` | 0.173 | 0.000 | 4.328 | 0.173 | 0.1 | 1/25 |
| `MPI_Waitall` | 0.172 | 0.166 | 0.415 | 0.172 | 2.0 | 25/25 |
| `MPI_Gatherv` | 0.072 | 0.062 | 0.100 | 0.072 | 0.0 | 25/25 |
| `MPI_Irecv` | 0.013 | 0.015 | 0.015 | 0.013 | 8.1 | 25/25 |
| `MPI_Isend` | 0.011 | 0.000 | 0.271 | 0.011 | 2.9 | 1/25 |
| `MPI_Isend` | 0.010 | 0.007 | 0.030 | 0.010 | 8.1 | 25/25 |
| `aix_collective_setup` | 0.004 | 0.005 | 0.005 | 0.004 | 1.0 | 25/25 |
| `MPI_Gather` | 0.001 | 0.003 | 0.003 | 0.001 | 0.0 | 25/25 |
| `MPI_Irecv` | 0.001 | 0.000 | 0.025 | 0.001 | 1.0 | 1/25 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 21193.656 | 1.0 | 25/25 |
| `solver_setup` | 1028.216 | 1.0 | 25/25 |
| `solver_step_ml_warmup` | 929.901 | 1.0 | 25/25 |
| `solver_teardown` | 62.072 | 1.0 | 25/25 |
