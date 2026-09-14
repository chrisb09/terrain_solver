# s3_smartsim_per_gpu_db (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 191.203 | 86.208 | 215.542 | 0.004 | 1.0 | 24/24 |
| `solver_ml_provider_call` | 191.199 | 86.150 | 215.540 | 0.007 | 1.0 | 24/24 |
| `app_provider_inference` | 184.048 | 78.349 | 208.469 | 0.004 | 1.0 | 24/24 |
| `smartsim_library_static_step` | 184.044 | 78.344 | 208.465 | 0.177 | 1.0 | 24/24 |
| `smartsim_put_tensor` | 100.995 | 61.134 | 109.152 | 100.995 | 2.0 | 24/24 |
| `smartsim_run_model` | 79.583 | 13.416 | 96.322 | 79.583 | 1.0 | 24/24 |
| `MPI_Allreduce` | 55.015 | 161.068 | 161.068 | 55.015 | 36.9 | 24/24 |
| `app_prepare_input` | 6.630 | 6.596 | 6.740 | 6.630 | 1.0 | 24/24 |
| `solver_step_compute` | 5.793 | 5.359 | 6.212 | 5.793 | 1.0 | 24/24 |
| `smartsim_unpack_tensor` | 3.286 | 3.451 | 3.668 | 3.286 | 1.0 | 24/24 |
| `MPI_Bcast` | 0.654 | 0.014 | 0.736 | 0.654 | 2.0 | 24/24 |
| `app_finalize_output` | 0.514 | 1.194 | 1.194 | 0.514 | 1.0 | 24/24 |
| `MPI_Waitall` | 0.188 | 0.102 | 0.289 | 0.188 | 2.0 | 24/24 |
| `MPI_Gatherv` | 0.061 | 0.062 | 0.089 | 0.061 | 0.0 | 24/24 |
| `MPI_Isend` | 0.029 | 0.026 | 0.031 | 0.029 | 8.1 | 24/24 |
| `MPI_Irecv` | 0.019 | 0.020 | 0.020 | 0.019 | 8.1 | 24/24 |
| `smartsim_chunk_plan` | 0.003 | 0.003 | 0.004 | 0.003 | 1.0 | 24/24 |
| `MPI_Gather` | 0.000 | 0.000 | 0.003 | 0.000 | 0.0 | 24/24 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 32695.721 | 1.0 | 24/24 |
| `solver_setup` | 6111.248 | 1.0 | 24/24 |
| `solver_step_ml_warmup` | 479.865 | 1.0 | 24/24 |
| `solver_teardown` | 35.475 | 1.0 | 24/24 |
