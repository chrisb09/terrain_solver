# aix_p2p (96c4g)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 29.994 | 32.239 | 32.239 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 29.992 | 32.152 | 32.152 | 0.003 | 1.0 | 96/96 |
| `app_provider_inference` | 28.928 | 31.276 | 31.498 | 0.001 | 1.0 | 96/96 |
| `aix_library_static_step` | 28.927 | 31.274 | 31.496 | 0.000 | 1.0 | 96/96 |
| `aix_inference` | 28.926 | 31.273 | 31.496 | 0.004 | 1.0 | 96/96 |
| `inference` | 28.922 | 31.268 | 31.492 | 0.004 | 1.0 | 96/96 |
| `p2p_pipelined_exchange` | 28.916 | 31.252 | 31.486 | 0.661 | 1.0 | 96/96 |
| `MPI_Allreduce` | 22.695 | 11.441 | 31.072 | 22.695 | 44.6 | 96/96 |
| `MPI_Recv` | 20.122 | 0.000 | 29.306 | 20.122 | 2.9 | 92/96 |
| `MPI_Send` | 7.543 | 0.000 | 16.098 | 7.543 | 1.0 | 92/96 |
| `MPI_Bcast` | 4.767 | 0.018 | 5.323 | 4.767 | 2.4 | 96/96 |
| `app_prepare_input` | 1.027 | 0.845 | 2.344 | 1.027 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.823 | 0.363 | 0.877 | 0.823 | 0.1 | 96/96 |
| `MPI_Testsome` | 0.583 | 14.882 | 14.882 | 0.583 | 17.1 | 4/96 |
| `solver_step_compute` | 0.473 | 0.909 | 0.973 | 0.473 | 1.1 | 96/96 |
| `MPI_Waitall` | 0.190 | 0.568 | 0.981 | 0.190 | 2.2 | 96/96 |
| `MPI_Isend` | 0.046 | 0.047 | 0.540 | 0.046 | 8.8 | 96/96 |
| `app_finalize_output` | 0.036 | 0.028 | 0.064 | 0.036 | 1.0 | 96/96 |
| `MPI_Gather` | 0.029 | 0.548 | 0.548 | 0.029 | 0.1 | 96/96 |
| `MPI_Irecv` | 0.013 | 0.020 | 0.020 | 0.013 | 8.8 | 96/96 |
| `MPI_Isend` | 0.004 | 0.094 | 0.097 | 0.004 | 2.9 | 4/96 |
| `MPI_Waitall` | 0.002 | 0.061 | 0.061 | 0.002 | 0.1 | 4/96 |
| `aix_collective_setup` | 0.002 | 0.003 | 0.004 | 0.002 | 1.0 | 96/96 |
| `MPI_Irecv` | 0.001 | 0.019 | 0.020 | 0.001 | 1.0 | 4/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 60532.614 | 1.0 | 96/96 |
| `solver_setup` | 57009.035 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 703.954 | 1.0 | 96/96 |
| `solver_teardown` | 46.615 | 1.0 | 96/96 |
