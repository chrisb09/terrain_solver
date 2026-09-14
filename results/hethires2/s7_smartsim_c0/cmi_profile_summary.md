# s7_smartsim_c0 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 872.393 | 776.336 | 915.816 | 0.002 | 1.0 | 192/192 |
| `solver_ml_provider_call` | 872.391 | 776.317 | 915.815 | 0.003 | 1.0 | 192/192 |
| `app_provider_inference` | 871.674 | 775.895 | 914.883 | 0.002 | 1.0 | 192/192 |
| `smartsim_library_static_step` | 871.672 | 775.894 | 914.882 | 0.091 | 1.0 | 192/192 |
| `smartsim_run_model` | 432.662 | 391.579 | 457.933 | 432.662 | 1.0 | 192/192 |
| `smartsim_put_tensor` | 308.726 | 216.536 | 372.205 | 308.726 | 2.0 | 192/192 |
| `MPI_Allreduce` | 138.476 | 227.083 | 283.858 | 138.476 | 36.9 | 192/192 |
| `smartsim_unpack_tensor` | 130.192 | 167.610 | 172.374 | 130.192 | 1.0 | 192/192 |
| `MPI_Bcast` | 8.385 | 0.005 | 15.465 | 8.385 | 2.0 | 192/192 |
| `app_prepare_input` | 0.652 | 0.367 | 1.087 | 0.652 | 1.0 | 192/192 |
| `solver_step_compute` | 0.572 | 0.568 | 1.028 | 0.572 | 1.0 | 192/192 |
| `MPI_Waitall` | 0.544 | 5.393 | 7.750 | 0.544 | 2.0 | 192/192 |
| `MPI_Gatherv` | 0.139 | 0.295 | 0.341 | 0.139 | 0.0 | 192/192 |
| `app_finalize_output` | 0.062 | 0.053 | 0.105 | 0.062 | 1.0 | 192/192 |
| `MPI_Irecv` | 0.016 | 0.050 | 0.050 | 0.016 | 8.1 | 192/192 |
| `MPI_Isend` | 0.013 | 0.040 | 0.040 | 0.013 | 8.1 | 192/192 |
| `smartsim_chunk_plan` | 0.001 | 0.001 | 0.002 | 0.001 | 1.0 | 192/192 |
| `MPI_Gather` | 0.000 | 0.004 | 0.004 | 0.000 | 0.0 | 192/192 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 119364.708 | 1.0 | 192/192 |
| `solver_setup` | 9325.024 | 1.0 | 192/192 |
| `solver_step_ml_warmup` | 960.033 | 1.0 | 192/192 |
| `solver_teardown` | 37.263 | 1.0 | 192/192 |
