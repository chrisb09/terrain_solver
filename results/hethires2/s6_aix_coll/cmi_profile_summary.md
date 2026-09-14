# s6_aix_coll (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 55.422 | 57.689 | 57.980 | 0.001 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 55.421 | 57.664 | 57.978 | 0.004 | 1.0 | 96/96 |
| `app_provider_inference` | 50.099 | 52.602 | 53.519 | 0.002 | 1.0 | 96/96 |
| `aix_library_static_step` | 50.097 | 52.601 | 53.517 | 0.001 | 1.0 | 96/96 |
| `aix_inference` | 50.096 | 52.600 | 53.516 | 0.015 | 1.0 | 96/96 |
| `inference` | 50.081 | 52.590 | 53.497 | 0.004 | 1.0 | 96/96 |
| `scatterOutputData` | 31.431 | 2.140 | 47.825 | 0.001 | 1.0 | 96/96 |
| `aix_collective_scatter_mpi` | 31.430 | 2.139 | 47.824 | 0.001 | 1.0 | 96/96 |
| `MPI_Scatterv` | 31.429 | 2.136 | 47.823 | 31.429 | 1.0 | 96/96 |
| `MPI_Allreduce` | 25.684 | 18.405 | 29.047 | 25.684 | 36.9 | 96/96 |
| `gatherInputData` | 17.917 | 33.807 | 33.807 | 0.001 | 1.0 | 96/96 |
| `aix_collective_gather_mpi` | 17.916 | 33.805 | 33.805 | 0.003 | 1.0 | 96/96 |
| `MPI_Gatherv` | 17.913 | 33.800 | 33.800 | 17.913 | 1.0 | 96/96 |
| `app_prepare_input` | 5.205 | 4.931 | 6.926 | 5.205 | 1.0 | 96/96 |
| `solver_step_compute` | 1.372 | 2.102 | 2.102 | 1.372 | 1.0 | 96/96 |
| `MPI_Bcast` | 0.981 | 0.015 | 1.237 | 0.981 | 2.0 | 96/96 |
| `inferenceDevice` | 0.720 | 16.558 | 19.314 | 0.000 | 1.0 | 96/96 |
| `aix_controller_inference` | 0.720 | 16.556 | 19.313 | 0.000 | 0.0 | 4/96 |
| `torchInference::inference` | 0.720 | 16.554 | 19.310 | 0.004 | 0.0 | 4/96 |
| `torchInference::forward` | 0.435 | 10.350 | 10.504 | 0.435 | 0.0 | 4/96 |
| `h2d_copy` | 0.165 | 3.445 | 6.082 | 0.165 | 0.0 | 4/96 |
| `d2h_copy` | 0.116 | 2.681 | 2.958 | 0.116 | 0.0 | 4/96 |
| `app_finalize_output` | 0.113 | 0.127 | 0.141 | 0.113 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.093 | 0.079 | 0.132 | 0.093 | 0.0 | 96/96 |
| `MPI_Waitall` | 0.068 | 0.153 | 0.357 | 0.068 | 2.0 | 96/96 |
| `MPI_Isend` | 0.019 | 0.025 | 0.025 | 0.019 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.011 | 0.020 | 0.020 | 0.011 | 8.1 | 96/96 |
| `aix_collective_setup` | 0.006 | 0.005 | 0.008 | 0.006 | 1.0 | 96/96 |
| `aix_collective_post_scatter` | 0.002 | 0.036 | 0.070 | 0.002 | 1.0 | 96/96 |
| `inferenceHost` | 0.000 | 0.001 | 0.001 | 0.000 | 1.0 | 96/96 |
| `aix_collective_worker_wait_for_controller` | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 92/96 |
| `MPI_Gather` | 0.000 | 0.001 | 0.001 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 68462.745 | 1.0 | 96/96 |
| `solver_setup` | 57450.067 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 504.066 | 1.0 | 96/96 |
| `solver_teardown` | 36.227 | 1.0 | 96/96 |
