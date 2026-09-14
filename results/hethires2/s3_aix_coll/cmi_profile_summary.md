# s3_aix_coll (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 179.647 | 182.795 | 182.795 | 0.004 | 1.0 | 24/24 |
| `solver_ml_provider_call` | 179.644 | 182.733 | 182.733 | 0.005 | 1.0 | 24/24 |
| `app_provider_inference` | 162.929 | 166.463 | 166.463 | 0.003 | 1.0 | 24/24 |
| `aix_library_static_step` | 162.927 | 166.462 | 166.462 | 0.001 | 1.0 | 24/24 |
| `aix_inference` | 162.926 | 166.461 | 166.461 | 0.021 | 1.0 | 24/24 |
| `inference` | 162.904 | 166.452 | 166.452 | 0.011 | 1.0 | 24/24 |
| `scatterOutputData` | 105.105 | 6.167 | 152.651 | 0.001 | 1.0 | 24/24 |
| `aix_collective_scatter_mpi` | 105.104 | 6.166 | 152.650 | 0.001 | 1.0 | 24/24 |
| `MPI_Scatterv` | 105.103 | 6.162 | 152.649 | 105.103 | 1.0 | 24/24 |
| `gatherInputData` | 55.293 | 100.366 | 100.366 | 0.002 | 1.0 | 24/24 |
| `aix_collective_gather_mpi` | 55.291 | 100.364 | 100.364 | 0.003 | 1.0 | 24/24 |
| `MPI_Gatherv` | 55.288 | 100.360 | 100.360 | 55.288 | 1.0 | 24/24 |
| `app_prepare_input` | 16.305 | 15.841 | 16.961 | 16.305 | 1.0 | 24/24 |
| `solver_step_compute` | 5.641 | 5.370 | 5.958 | 5.641 | 1.0 | 24/24 |
| `MPI_Allreduce` | 5.000 | 2.002 | 7.969 | 5.000 | 36.9 | 24/24 |
| `inferenceDevice` | 2.476 | 59.414 | 59.414 | 0.000 | 1.0 | 24/24 |
| `aix_controller_inference` | 2.476 | 59.412 | 59.412 | 0.000 | 0.0 | 1/24 |
| `torchInference::inference` | 2.475 | 59.410 | 59.410 | 0.003 | 0.0 | 1/24 |
| `torchInference::forward` | 1.705 | 40.913 | 40.913 | 1.705 | 0.0 | 1/24 |
| `MPI_Bcast` | 0.696 | 0.013 | 0.749 | 0.696 | 2.0 | 24/24 |
| `h2d_copy` | 0.452 | 10.846 | 10.846 | 0.452 | 0.0 | 1/24 |
| `app_finalize_output` | 0.404 | 0.425 | 0.445 | 0.404 | 1.0 | 24/24 |
| `d2h_copy` | 0.316 | 7.580 | 7.580 | 0.316 | 0.0 | 1/24 |
| `MPI_Waitall` | 0.113 | 0.149 | 0.232 | 0.113 | 2.0 | 24/24 |
| `MPI_Gatherv` | 0.055 | 0.058 | 0.082 | 0.055 | 0.0 | 24/24 |
| `MPI_Isend` | 0.021 | 0.022 | 0.024 | 0.021 | 8.1 | 24/24 |
| `MPI_Irecv` | 0.013 | 0.019 | 0.019 | 0.013 | 8.1 | 24/24 |
| `aix_collective_post_scatter` | 0.013 | 0.294 | 0.294 | 0.013 | 1.0 | 24/24 |
| `aix_collective_setup` | 0.007 | 0.004 | 0.010 | 0.007 | 1.0 | 24/24 |
| `MPI_Gather` | 0.002 | 0.002 | 0.004 | 0.002 | 0.0 | 24/24 |
| `inferenceHost` | 0.000 | 0.001 | 0.001 | 0.000 | 1.0 | 24/24 |
| `aix_collective_worker_wait_for_controller` | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 23/24 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 22144.638 | 1.0 | 24/24 |
| `solver_setup` | 1838.005 | 1.0 | 24/24 |
| `solver_step_ml_warmup` | 394.398 | 1.0 | 24/24 |
| `solver_teardown` | 24.967 | 1.0 | 24/24 |
