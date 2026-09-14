# s7_aix_coll (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 95.361 | 117.049 | 117.049 | 0.003 | 1.0 | 192/192 |
| `solver_ml_provider_call` | 95.358 | 117.046 | 117.046 | 0.002 | 1.0 | 192/192 |
| `app_provider_inference` | 93.923 | 115.738 | 115.738 | 0.001 | 1.0 | 192/192 |
| `aix_library_static_step` | 93.922 | 115.737 | 115.737 | 0.001 | 1.0 | 192/192 |
| `aix_inference` | 93.922 | 115.736 | 115.736 | 0.008 | 1.0 | 192/192 |
| `inference` | 93.914 | 115.723 | 115.723 | 0.003 | 1.0 | 192/192 |
| `MPI_Allreduce` | 86.473 | 56.056 | 105.290 | 86.473 | 36.9 | 192/192 |
| `scatterOutputData` | 49.506 | 14.318 | 99.367 | 0.000 | 1.0 | 192/192 |
| `aix_collective_scatter_mpi` | 49.506 | 14.317 | 99.366 | 0.001 | 1.0 | 192/192 |
| `MPI_Scatterv` | 49.505 | 14.312 | 99.348 | 49.505 | 1.0 | 192/192 |
| `gatherInputData` | 44.054 | 84.496 | 84.496 | 0.001 | 1.0 | 192/192 |
| `aix_collective_gather_mpi` | 44.053 | 84.494 | 84.494 | 0.002 | 1.0 | 192/192 |
| `MPI_Gatherv` | 44.051 | 84.488 | 84.488 | 44.051 | 1.0 | 192/192 |
| `MPI_Bcast` | 6.452 | 9.865 | 10.165 | 6.452 | 2.0 | 192/192 |
| `app_prepare_input` | 1.375 | 1.230 | 3.019 | 1.375 | 1.0 | 192/192 |
| `solver_step_compute` | 0.559 | 0.646 | 1.036 | 0.559 | 1.0 | 192/192 |
| `MPI_Waitall` | 0.375 | 0.571 | 6.631 | 0.375 | 2.0 | 192/192 |
| `inferenceDevice` | 0.347 | 16.817 | 17.398 | 0.001 | 1.0 | 192/192 |
| `aix_controller_inference` | 0.347 | 16.816 | 17.396 | 0.000 | 0.0 | 4/192 |
| `torchInference::inference` | 0.347 | 16.813 | 17.390 | 0.002 | 0.0 | 4/192 |
| `torchInference::forward` | 0.217 | 10.447 | 10.447 | 0.217 | 0.0 | 4/192 |
| `MPI_Gatherv` | 0.140 | 0.083 | 0.337 | 0.140 | 0.0 | 192/192 |
| `h2d_copy` | 0.068 | 2.814 | 4.659 | 0.068 | 0.0 | 4/192 |
| `d2h_copy` | 0.059 | 3.456 | 3.456 | 0.059 | 0.0 | 4/192 |
| `app_finalize_output` | 0.058 | 0.074 | 0.094 | 0.058 | 1.0 | 192/192 |
| `MPI_Irecv` | 0.012 | 0.018 | 0.062 | 0.012 | 8.1 | 192/192 |
| `MPI_Isend` | 0.010 | 0.026 | 0.055 | 0.010 | 8.1 | 192/192 |
| `aix_collective_setup` | 0.003 | 0.005 | 0.008 | 0.003 | 1.0 | 192/192 |
| `aix_collective_post_scatter` | 0.001 | 0.029 | 0.049 | 0.001 | 1.0 | 192/192 |
| `MPI_Gather` | 0.001 | 0.004 | 0.007 | 0.001 | 0.0 | 192/192 |
| `inferenceHost` | 0.000 | 0.001 | 0.001 | 0.000 | 1.0 | 192/192 |
| `aix_collective_worker_wait_for_controller` | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 188/192 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 81240.857 | 1.0 | 192/192 |
| `solver_setup` | 54814.886 | 1.0 | 192/192 |
| `solver_step_ml_warmup` | 536.660 | 1.0 | 192/192 |
| `solver_teardown` | 70.554 | 1.0 | 192/192 |
