# aix_coll (96c4g)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `MPI_Allreduce` | 56.248 | 15.854 | 60.072 | 56.248 | 44.6 | 96/96 |
| `solver_step_ml_steady` | 26.169 | 28.336 | 28.336 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 26.167 | 28.271 | 28.271 | 0.004 | 1.0 | 96/96 |
| `app_provider_inference` | 25.072 | 27.189 | 27.708 | 0.001 | 1.0 | 96/96 |
| `aix_library_static_step` | 25.071 | 27.187 | 27.706 | 0.000 | 1.0 | 96/96 |
| `aix_inference` | 25.070 | 27.187 | 27.706 | 0.009 | 1.0 | 96/96 |
| `inference` | 25.062 | 27.181 | 27.698 | 0.008 | 1.0 | 96/96 |
| `scatterOutputData` | 16.546 | 0.959 | 23.764 | 0.001 | 1.0 | 96/96 |
| `aix_collective_scatter_mpi` | 16.545 | 0.957 | 23.763 | 0.001 | 1.0 | 96/96 |
| `MPI_Scatterv` | 16.545 | 0.952 | 23.762 | 16.545 | 1.0 | 96/96 |
| `gatherInputData` | 8.091 | 16.556 | 16.556 | 0.001 | 1.0 | 96/96 |
| `aix_collective_gather_mpi` | 8.090 | 16.554 | 16.554 | 0.002 | 1.0 | 96/96 |
| `MPI_Gatherv` | 8.088 | 16.550 | 16.550 | 8.088 | 1.0 | 96/96 |
| `MPI_Bcast` | 3.667 | 0.020 | 4.570 | 3.667 | 2.4 | 96/96 |
| `app_prepare_input` | 1.065 | 1.050 | 2.077 | 1.065 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.844 | 0.375 | 0.951 | 0.844 | 0.1 | 96/96 |
| `solver_step_compute` | 0.531 | 0.197 | 0.877 | 0.531 | 1.1 | 96/96 |
| `inferenceDevice` | 0.411 | 9.406 | 11.004 | 0.001 | 1.0 | 96/96 |
| `aix_controller_inference` | 0.410 | 9.405 | 11.002 | 0.000 | 0.0 | 4/96 |
| `torchInference::inference` | 0.410 | 9.402 | 11.000 | 0.005 | 0.0 | 4/96 |
| `torchInference::forward` | 0.238 | 5.651 | 5.934 | 0.238 | 0.5 | 4/96 |
| `MPI_Waitall` | 0.169 | 0.519 | 1.120 | 0.169 | 2.2 | 96/96 |
| `h2d_copy` | 0.104 | 2.121 | 3.605 | 0.104 | 0.5 | 4/96 |
| `d2h_copy` | 0.062 | 1.477 | 1.591 | 0.062 | 0.5 | 4/96 |
| `MPI_Isend` | 0.041 | 0.032 | 0.050 | 0.041 | 8.8 | 96/96 |
| `app_finalize_output` | 0.026 | 0.027 | 0.052 | 0.026 | 1.0 | 96/96 |
| `MPI_Irecv` | 0.015 | 0.019 | 0.021 | 0.015 | 8.8 | 96/96 |
| `MPI_Gather` | 0.003 | 0.037 | 0.037 | 0.003 | 0.1 | 96/96 |
| `aix_collective_post_scatter` | 0.003 | 0.058 | 0.058 | 0.003 | 1.0 | 96/96 |
| `aix_collective_setup` | 0.002 | 0.003 | 0.004 | 0.002 | 1.0 | 96/96 |
| `inferenceHost` | 0.001 | 0.001 | 0.003 | 0.001 | 1.0 | 96/96 |
| `aix_collective_worker_wait_for_controller` | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 92/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 77581.595 | 1.0 | 96/96 |
| `solver_setup` | 60978.060 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 12529.300 | 1.0 | 96/96 |
| `solver_teardown` | 6.166 | 1.0 | 96/96 |
