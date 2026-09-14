# s5_aix_coll (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 251.382 | 256.856 | 256.856 | 0.004 | 1.0 | 48/48 |
| `solver_ml_provider_call` | 251.379 | 256.854 | 256.854 | 0.004 | 1.0 | 48/48 |
| `app_provider_inference` | 243.179 | 249.643 | 249.643 | 0.002 | 1.0 | 48/48 |
| `aix_library_static_step` | 243.178 | 249.641 | 249.641 | 0.001 | 1.0 | 48/48 |
| `aix_inference` | 243.176 | 249.640 | 249.640 | 0.018 | 1.0 | 48/48 |
| `inference` | 243.158 | 249.621 | 249.621 | 0.014 | 1.0 | 48/48 |
| `scatterOutputData` | 135.023 | 11.601 | 232.702 | 0.001 | 1.0 | 48/48 |
| `aix_collective_scatter_mpi` | 135.023 | 11.600 | 232.701 | 0.001 | 1.0 | 48/48 |
| `MPI_Scatterv` | 135.021 | 11.596 | 232.692 | 135.021 | 1.0 | 48/48 |
| `gatherInputData` | 106.856 | 177.111 | 177.111 | 0.001 | 1.0 | 48/48 |
| `aix_collective_gather_mpi` | 106.855 | 177.107 | 177.107 | 0.004 | 1.0 | 48/48 |
| `MPI_Gatherv` | 106.851 | 177.096 | 177.096 | 106.851 | 1.0 | 48/48 |
| `MPI_Allreduce` | 14.829 | 8.677 | 21.408 | 14.829 | 36.9 | 48/48 |
| `app_prepare_input` | 7.970 | 6.995 | 11.531 | 7.970 | 1.0 | 48/48 |
| `solver_step_compute` | 2.378 | 2.639 | 3.256 | 2.378 | 1.0 | 48/48 |
| `inferenceDevice` | 1.249 | 59.942 | 59.942 | 0.001 | 1.0 | 48/48 |
| `aix_controller_inference` | 1.249 | 59.940 | 59.940 | 0.000 | 0.0 | 1/48 |
| `torchInference::inference` | 1.249 | 59.936 | 59.936 | 0.004 | 0.0 | 1/48 |
| `torchInference::forward` | 0.853 | 40.963 | 40.963 | 0.853 | 0.0 | 1/48 |
| `MPI_Bcast` | 0.815 | 0.804 | 0.933 | 0.815 | 2.0 | 48/48 |
| `MPI_Waitall` | 0.229 | 0.502 | 0.502 | 0.229 | 2.0 | 48/48 |
| `h2d_copy` | 0.227 | 10.899 | 10.899 | 0.227 | 0.0 | 1/48 |
| `app_finalize_output` | 0.226 | 0.210 | 0.282 | 0.226 | 1.0 | 48/48 |
| `d2h_copy` | 0.164 | 7.884 | 7.884 | 0.164 | 0.0 | 1/48 |
| `MPI_Gatherv` | 0.097 | 0.078 | 0.178 | 0.097 | 0.0 | 48/48 |
| `MPI_Irecv` | 0.015 | 0.020 | 0.020 | 0.015 | 8.1 | 48/48 |
| `MPI_Isend` | 0.014 | 0.025 | 0.032 | 0.014 | 8.1 | 48/48 |
| `aix_collective_post_scatter` | 0.008 | 0.385 | 0.385 | 0.008 | 1.0 | 48/48 |
| `aix_collective_setup` | 0.006 | 0.008 | 0.025 | 0.006 | 1.0 | 48/48 |
| `MPI_Gather` | 0.001 | 0.003 | 0.004 | 0.001 | 0.0 | 48/48 |
| `inferenceHost` | 0.000 | 0.001 | 0.001 | 0.000 | 1.0 | 48/48 |
| `aix_collective_worker_wait_for_controller` | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 47/48 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 30080.417 | 1.0 | 48/48 |
| `solver_setup` | 1977.729 | 1.0 | 48/48 |
| `solver_step_ml_warmup` | 527.155 | 1.0 | 48/48 |
| `solver_teardown` | 39.130 | 1.0 | 48/48 |
