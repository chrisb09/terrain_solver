# s1_aix_coll (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 176.853 | 178.089 | 178.089 | 0.013 | 1.0 | 4/4 |
| `solver_ml_provider_call` | 176.840 | 178.041 | 178.041 | 0.004 | 1.0 | 4/4 |
| `app_provider_inference` | 138.819 | 141.411 | 141.411 | 0.001 | 1.0 | 4/4 |
| `aix_library_static_step` | 138.818 | 141.410 | 141.410 | 0.001 | 1.0 | 4/4 |
| `aix_inference` | 138.817 | 141.409 | 141.409 | 0.008 | 1.0 | 4/4 |
| `inference` | 138.810 | 141.402 | 141.402 | 0.038 | 1.0 | 4/4 |
| `scatterOutputData` | 63.162 | 4.905 | 99.967 | 0.001 | 1.0 | 4/4 |
| `aix_collective_scatter_mpi` | 63.161 | 4.903 | 99.966 | 0.002 | 1.0 | 4/4 |
| `MPI_Scatterv` | 63.159 | 4.900 | 99.965 | 63.159 | 1.0 | 4/4 |
| `gatherInputData` | 60.812 | 77.177 | 77.177 | 0.001 | 1.0 | 4/4 |
| `aix_collective_gather_mpi` | 60.811 | 77.175 | 77.175 | 0.002 | 1.0 | 4/4 |
| `MPI_Gatherv` | 60.809 | 77.171 | 77.171 | 60.809 | 1.0 | 4/4 |
| `app_prepare_input` | 35.419 | 33.879 | 37.061 | 35.419 | 1.0 | 4/4 |
| `solver_step_compute` | 21.491 | 21.528 | 21.632 | 21.491 | 1.0 | 4/4 |
| `inferenceDevice` | 14.780 | 59.121 | 59.121 | 0.001 | 1.0 | 4/4 |
| `aix_controller_inference` | 14.780 | 59.120 | 59.120 | 0.001 | 0.2 | 1/4 |
| `torchInference::inference` | 14.779 | 59.117 | 59.117 | 0.018 | 0.2 | 1/4 |
| `torchInference::forward` | 10.228 | 40.910 | 40.910 | 10.228 | 0.2 | 1/4 |
| `h2d_copy` | 2.710 | 10.840 | 10.840 | 2.710 | 0.2 | 1/4 |
| `app_finalize_output` | 2.599 | 2.747 | 2.747 | 2.599 | 1.0 | 4/4 |
| `d2h_copy` | 1.824 | 7.295 | 7.295 | 1.824 | 0.2 | 1/4 |
| `MPI_Allreduce` | 1.790 | 0.411 | 3.814 | 1.790 | 36.9 | 4/4 |
| `MPI_Bcast` | 0.485 | 0.006 | 0.656 | 0.485 | 2.0 | 4/4 |
| `MPI_Waitall` | 0.058 | 0.067 | 0.077 | 0.058 | 2.0 | 4/4 |
| `MPI_Gatherv` | 0.055 | 0.046 | 0.070 | 0.055 | 0.0 | 4/4 |
| `MPI_Isend` | 0.015 | 0.018 | 0.018 | 0.015 | 8.1 | 4/4 |
| `MPI_Irecv` | 0.014 | 0.018 | 0.018 | 0.014 | 8.1 | 4/4 |
| `aix_collective_post_scatter` | 0.013 | 0.049 | 0.049 | 0.013 | 1.0 | 4/4 |
| `aix_collective_setup` | 0.003 | 0.003 | 0.004 | 0.003 | 1.0 | 4/4 |
| `MPI_Gather` | 0.001 | 0.002 | 0.002 | 0.001 | 0.0 | 4/4 |
| `inferenceHost` | 0.000 | 0.001 | 0.001 | 0.000 | 1.0 | 4/4 |
| `aix_collective_worker_wait_for_controller` | 0.000 | 0.000 | 0.000 | 0.000 | 0.8 | 3/4 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 35894.200 | 1.0 | 4/4 |
| `solver_step_ml_warmup` | 12467.800 | 1.0 | 4/4 |
| `solver_setup` | 2611.533 | 1.0 | 4/4 |
| `solver_teardown` | 43.134 | 1.0 | 4/4 |
