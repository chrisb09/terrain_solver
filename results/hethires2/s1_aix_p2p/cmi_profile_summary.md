# s1_aix_p2p (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 141.971 | 150.438 | 150.438 | 0.025 | 1.0 | 4/4 |
| `solver_ml_provider_call` | 141.946 | 150.340 | 150.340 | 0.003 | 1.0 | 4/4 |
| `app_provider_inference` | 103.798 | 113.516 | 113.516 | 0.001 | 1.0 | 4/4 |
| `aix_library_static_step` | 103.797 | 113.515 | 113.515 | 0.000 | 1.0 | 4/4 |
| `aix_inference` | 103.796 | 113.514 | 113.514 | 0.007 | 1.0 | 4/4 |
| `inference` | 103.789 | 113.507 | 113.507 | 0.006 | 1.0 | 4/4 |
| `p2p_pipelined_exchange` | 103.781 | 113.497 | 113.497 | 12.754 | 1.0 | 4/4 |
| `MPI_Recv` | 43.517 | 0.000 | 66.924 | 43.517 | 2.2 | 3/4 |
| `app_prepare_input` | 35.606 | 34.060 | 37.318 | 35.606 | 1.0 | 4/4 |
| `MPI_Send` | 31.881 | 0.000 | 44.292 | 31.881 | 0.8 | 3/4 |
| `solver_step_compute` | 21.537 | 21.448 | 21.756 | 21.537 | 1.0 | 4/4 |
| `MPI_Testsome` | 15.315 | 61.260 | 61.260 | 15.315 | 160.8 | 1/4 |
| `MPI_Allreduce` | 9.098 | 0.623 | 23.661 | 9.098 | 36.9 | 4/4 |
| `app_finalize_output` | 2.539 | 2.761 | 2.761 | 2.539 | 1.0 | 4/4 |
| `MPI_Bcast` | 0.479 | 0.005 | 0.641 | 0.479 | 2.0 | 4/4 |
| `MPI_Waitall` | 0.307 | 1.227 | 1.227 | 0.307 | 0.8 | 1/4 |
| `MPI_Gatherv` | 0.071 | 0.043 | 0.093 | 0.071 | 0.0 | 4/4 |
| `MPI_Waitall` | 0.069 | 0.092 | 0.096 | 0.069 | 2.0 | 4/4 |
| `MPI_Isend` | 0.014 | 0.015 | 0.015 | 0.014 | 8.1 | 4/4 |
| `MPI_Irecv` | 0.013 | 0.015 | 0.015 | 0.013 | 8.1 | 4/4 |
| `MPI_Isend` | 0.005 | 0.021 | 0.021 | 0.005 | 2.2 | 1/4 |
| `aix_collective_setup` | 0.002 | 0.002 | 0.002 | 0.002 | 1.0 | 4/4 |
| `MPI_Irecv` | 0.001 | 0.006 | 0.006 | 0.001 | 0.8 | 1/4 |
| `MPI_Gather` | 0.001 | 0.000 | 0.003 | 0.001 | 0.0 | 4/4 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 33132.550 | 1.0 | 4/4 |
| `solver_step_ml_warmup` | 12378.200 | 1.0 | 4/4 |
| `solver_setup` | 2649.820 | 1.0 | 4/4 |
| `solver_teardown` | 92.958 | 1.0 | 4/4 |
