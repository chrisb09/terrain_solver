# s4_aix_coll (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 250.012 | 253.370 | 253.370 | 0.006 | 1.0 | 25/25 |
| `solver_ml_provider_call` | 250.006 | 253.368 | 253.368 | 0.005 | 1.0 | 25/25 |
| `app_provider_inference` | 235.157 | 247.422 | 247.422 | 0.002 | 1.0 | 25/25 |
| `aix_library_static_step` | 235.155 | 247.421 | 247.421 | 0.001 | 1.0 | 25/25 |
| `aix_inference` | 235.154 | 247.420 | 247.420 | 0.020 | 1.0 | 25/25 |
| `inference` | 235.135 | 247.414 | 247.414 | 0.003 | 1.0 | 25/25 |
| `scatterOutputData` | 141.933 | 7.707 | 225.944 | 0.001 | 1.0 | 25/25 |
| `aix_collective_scatter_mpi` | 141.933 | 7.706 | 225.943 | 0.002 | 1.0 | 25/25 |
| `MPI_Scatterv` | 141.931 | 7.701 | 225.939 | 141.931 | 1.0 | 25/25 |
| `gatherInputData` | 90.755 | 178.833 | 178.833 | 0.002 | 1.0 | 25/25 |
| `aix_collective_gather_mpi` | 90.754 | 178.831 | 178.831 | 0.004 | 1.0 | 25/25 |
| `MPI_Gatherv` | 90.750 | 178.826 | 178.826 | 90.750 | 1.0 | 25/25 |
| `app_prepare_input` | 14.395 | 5.387 | 15.720 | 14.395 | 1.0 | 25/25 |
| `MPI_Allreduce` | 7.718 | 4.901 | 11.725 | 7.718 | 36.9 | 25/25 |
| `solver_step_compute` | 5.265 | 4.374 | 5.535 | 5.265 | 1.0 | 25/25 |
| `inferenceDevice` | 2.433 | 60.811 | 60.811 | 0.001 | 1.0 | 25/25 |
| `aix_controller_inference` | 2.432 | 60.809 | 60.809 | 0.000 | 0.0 | 1/25 |
| `torchInference::inference` | 2.432 | 60.806 | 60.806 | 0.003 | 0.0 | 1/25 |
| `torchInference::forward` | 1.636 | 40.895 | 40.895 | 1.636 | 0.0 | 1/25 |
| `MPI_Bcast` | 0.911 | 0.977 | 0.977 | 0.911 | 2.0 | 25/25 |
| `h2d_copy` | 0.507 | 12.683 | 12.683 | 0.507 | 0.0 | 1/25 |
| `app_finalize_output` | 0.448 | 0.556 | 0.556 | 0.448 | 1.0 | 25/25 |
| `d2h_copy` | 0.286 | 7.150 | 7.150 | 0.286 | 0.0 | 1/25 |
| `MPI_Waitall` | 0.164 | 0.314 | 0.377 | 0.164 | 2.0 | 25/25 |
| `MPI_Gatherv` | 0.075 | 0.107 | 0.107 | 0.075 | 0.0 | 25/25 |
| `MPI_Irecv` | 0.014 | 0.021 | 0.021 | 0.014 | 8.1 | 25/25 |
| `MPI_Isend` | 0.013 | 0.030 | 0.032 | 0.013 | 8.1 | 25/25 |
| `aix_collective_setup` | 0.007 | 0.003 | 0.009 | 0.007 | 1.0 | 25/25 |
| `aix_collective_post_scatter` | 0.002 | 0.040 | 0.040 | 0.002 | 1.0 | 25/25 |
| `MPI_Gather` | 0.002 | 0.000 | 0.007 | 0.002 | 0.0 | 25/25 |
| `inferenceHost` | 0.000 | 0.001 | 0.001 | 0.000 | 1.0 | 25/25 |
| `aix_collective_worker_wait_for_controller` | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 24/25 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 28631.100 | 1.0 | 25/25 |
| `solver_setup` | 1148.056 | 1.0 | 25/25 |
| `solver_step_ml_warmup` | 493.818 | 1.0 | 25/25 |
| `solver_teardown` | 73.281 | 1.0 | 25/25 |
