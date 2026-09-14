# s2_aix_coll (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 261.162 | 262.007 | 262.007 | 0.017 | 1.0 | 5/5 |
| `solver_ml_provider_call` | 261.146 | 262.005 | 262.005 | 0.003 | 1.0 | 5/5 |
| `app_provider_inference` | 232.359 | 233.077 | 233.516 | 0.001 | 1.0 | 5/5 |
| `aix_library_static_step` | 232.358 | 233.076 | 233.515 | 0.001 | 1.0 | 5/5 |
| `aix_inference` | 232.358 | 233.076 | 233.514 | 0.009 | 1.0 | 5/5 |
| `inference` | 232.349 | 233.068 | 233.505 | 0.006 | 1.0 | 5/5 |
| `gatherInputData` | 120.130 | 168.212 | 168.212 | 0.001 | 1.0 | 5/5 |
| `aix_collective_gather_mpi` | 120.128 | 168.211 | 168.211 | 0.004 | 1.0 | 5/5 |
| `MPI_Gatherv` | 120.125 | 168.206 | 168.206 | 120.125 | 1.0 | 5/5 |
| `scatterOutputData` | 100.372 | 5.658 | 179.494 | 0.001 | 1.0 | 5/5 |
| `aix_collective_scatter_mpi` | 100.371 | 5.657 | 179.493 | 0.003 | 1.0 | 5/5 |
| `MPI_Scatterv` | 100.369 | 5.653 | 179.490 | 100.369 | 1.0 | 5/5 |
| `app_prepare_input` | 26.298 | 26.394 | 26.394 | 26.298 | 1.0 | 5/5 |
| `solver_step_compute` | 17.921 | 20.348 | 20.348 | 17.921 | 1.0 | 5/5 |
| `inferenceDevice` | 11.826 | 59.123 | 59.123 | 0.001 | 1.0 | 5/5 |
| `aix_controller_inference` | 11.824 | 59.122 | 59.122 | 0.000 | 0.2 | 1/5 |
| `torchInference::inference` | 11.824 | 59.119 | 59.119 | 0.016 | 0.2 | 1/5 |
| `torchInference::forward` | 8.191 | 40.953 | 40.953 | 8.191 | 0.2 | 1/5 |
| `MPI_Allreduce` | 5.192 | 1.698 | 10.334 | 5.192 | 36.9 | 5/5 |
| `app_finalize_output` | 2.485 | 2.530 | 3.287 | 2.485 | 1.0 | 5/5 |
| `h2d_copy` | 2.171 | 10.855 | 10.855 | 2.171 | 0.2 | 1/5 |
| `d2h_copy` | 1.447 | 7.233 | 7.233 | 1.447 | 0.2 | 1/5 |
| `MPI_Bcast` | 0.517 | 0.654 | 0.675 | 0.517 | 2.0 | 5/5 |
| `MPI_Waitall` | 0.120 | 0.128 | 0.194 | 0.120 | 2.0 | 5/5 |
| `MPI_Gatherv` | 0.072 | 0.098 | 0.098 | 0.072 | 0.0 | 5/5 |
| `MPI_Isend` | 0.023 | 0.033 | 0.039 | 0.023 | 8.1 | 5/5 |
| `MPI_Irecv` | 0.017 | 0.023 | 0.023 | 0.017 | 8.1 | 5/5 |
| `aix_collective_post_scatter` | 0.011 | 0.053 | 0.053 | 0.011 | 1.0 | 5/5 |
| `aix_collective_setup` | 0.004 | 0.004 | 0.004 | 0.004 | 1.0 | 5/5 |
| `MPI_Gather` | 0.001 | 0.000 | 0.004 | 0.001 | 0.0 | 5/5 |
| `inferenceHost` | 0.000 | 0.001 | 0.001 | 0.000 | 1.0 | 5/5 |
| `aix_collective_worker_wait_for_controller` | 0.000 | 0.000 | 0.000 | 0.000 | 0.8 | 4/5 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 50160.120 | 1.0 | 5/5 |
| `solver_step_ml_warmup` | 13084.620 | 1.0 | 5/5 |
| `solver_setup` | 2102.640 | 1.0 | 5/5 |
| `solver_teardown` | 77.509 | 1.0 | 5/5 |
