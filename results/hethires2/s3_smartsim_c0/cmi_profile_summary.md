# s3_smartsim_c0 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 263.580 | 187.753 | 294.400 | 0.003 | 1.0 | 24/24 |
| `solver_ml_provider_call` | 263.578 | 187.736 | 294.398 | 0.007 | 1.0 | 24/24 |
| `app_provider_inference` | 256.465 | 180.477 | 287.262 | 0.004 | 1.0 | 24/24 |
| `smartsim_library_static_step` | 256.462 | 180.474 | 287.260 | 0.120 | 1.0 | 24/24 |
| `smartsim_put_tensor` | 192.655 | 127.856 | 215.079 | 192.655 | 2.0 | 24/24 |
| `smartsim_run_model` | 57.505 | 27.446 | 70.324 | 57.505 | 1.0 | 24/24 |
| `MPI_Allreduce` | 56.152 | 133.361 | 133.361 | 56.152 | 36.9 | 24/24 |
| `app_prepare_input` | 6.647 | 6.715 | 6.755 | 6.647 | 1.0 | 24/24 |
| `smartsim_unpack_tensor` | 6.179 | 24.911 | 24.911 | 6.179 | 1.0 | 24/24 |
| `solver_step_compute` | 5.801 | 5.293 | 6.343 | 5.801 | 1.0 | 24/24 |
| `MPI_Bcast` | 0.705 | 0.014 | 0.805 | 0.705 | 2.0 | 24/24 |
| `app_finalize_output` | 0.458 | 0.536 | 0.536 | 0.458 | 1.0 | 24/24 |
| `MPI_Waitall` | 0.182 | 0.058 | 0.347 | 0.182 | 2.0 | 24/24 |
| `MPI_Gatherv` | 0.059 | 0.059 | 0.086 | 0.059 | 0.0 | 24/24 |
| `MPI_Isend` | 0.029 | 0.026 | 0.031 | 0.029 | 8.1 | 24/24 |
| `MPI_Irecv` | 0.018 | 0.020 | 0.020 | 0.018 | 8.1 | 24/24 |
| `smartsim_chunk_plan` | 0.003 | 0.003 | 0.004 | 0.003 | 1.0 | 24/24 |
| `MPI_Gather` | 0.000 | 0.001 | 0.004 | 0.000 | 0.0 | 24/24 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 37614.908 | 1.0 | 24/24 |
| `solver_setup` | 3421.272 | 1.0 | 24/24 |
| `solver_step_ml_warmup` | 684.760 | 1.0 | 24/24 |
| `solver_teardown` | 47.272 | 1.0 | 24/24 |
