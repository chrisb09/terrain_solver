# aix_coll (96c4g_hires)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 53.774 | 54.473 | 56.928 | 0.001 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 53.773 | 54.444 | 56.926 | 0.003 | 1.0 | 96/96 |
| `app_provider_inference` | 48.663 | 50.584 | 53.228 | 0.001 | 1.0 | 96/96 |
| `aix_library_static_step` | 48.661 | 50.582 | 53.226 | 0.001 | 1.0 | 96/96 |
| `aix_inference` | 48.661 | 50.581 | 53.225 | 0.012 | 1.0 | 96/96 |
| `inference` | 48.649 | 50.568 | 53.213 | 0.004 | 1.0 | 96/96 |
| `scatterOutputData` | 31.452 | 2.216 | 47.360 | 0.001 | 1.0 | 96/96 |
| `aix_collective_scatter_mpi` | 31.451 | 2.215 | 47.359 | 0.001 | 1.0 | 96/96 |
| `MPI_Scatterv` | 31.450 | 2.212 | 47.359 | 31.450 | 1.0 | 96/96 |
| `MPI_Allreduce` | 19.373 | 14.795 | 22.847 | 19.373 | 36.9 | 96/96 |
| `gatherInputData` | 16.464 | 31.670 | 31.782 | 0.001 | 1.0 | 96/96 |
| `aix_collective_gather_mpi` | 16.462 | 31.668 | 31.780 | 0.002 | 1.0 | 96/96 |
| `MPI_Gatherv` | 16.460 | 31.662 | 31.774 | 16.460 | 1.0 | 96/96 |
| `app_prepare_input` | 4.997 | 3.751 | 6.776 | 4.997 | 1.0 | 96/96 |
| `solver_step_compute` | 1.349 | 1.253 | 1.999 | 1.349 | 1.0 | 96/96 |
| `MPI_Bcast` | 0.988 | 0.017 | 1.121 | 0.988 | 2.0 | 96/96 |
| `inferenceDevice` | 0.722 | 16.587 | 19.312 | 0.000 | 1.0 | 96/96 |
| `aix_controller_inference` | 0.722 | 16.586 | 19.311 | 0.000 | 0.0 | 4/96 |
| `torchInference::inference` | 0.721 | 16.581 | 19.308 | 0.004 | 0.0 | 4/96 |
| `torchInference::forward` | 0.434 | 10.352 | 10.471 | 0.434 | 0.0 | 4/96 |
| `h2d_copy` | 0.165 | 3.452 | 6.076 | 0.165 | 0.0 | 4/96 |
| `d2h_copy` | 0.118 | 2.687 | 2.980 | 0.118 | 0.0 | 4/96 |
| `app_finalize_output` | 0.111 | 0.106 | 0.149 | 0.111 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.099 | 0.082 | 0.140 | 0.099 | 0.0 | 96/96 |
| `MPI_Waitall` | 0.073 | 0.026 | 0.370 | 0.073 | 2.0 | 96/96 |
| `MPI_Isend` | 0.022 | 0.026 | 0.082 | 0.022 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.011 | 0.021 | 0.021 | 0.011 | 8.1 | 96/96 |
| `aix_collective_setup` | 0.005 | 0.006 | 0.007 | 0.005 | 1.0 | 96/96 |
| `aix_collective_post_scatter` | 0.002 | 0.037 | 0.065 | 0.002 | 1.0 | 96/96 |
| `inferenceHost` | 0.000 | 0.001 | 0.005 | 0.000 | 1.0 | 96/96 |
| `MPI_Gather` | 0.000 | 0.001 | 0.001 | 0.000 | 0.0 | 96/96 |
| `aix_collective_worker_wait_for_controller` | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 92/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 89136.735 | 1.0 | 96/96 |
| `solver_setup` | 61838.781 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 17632.768 | 1.0 | 96/96 |
| `solver_teardown` | 35.114 | 1.0 | 96/96 |
