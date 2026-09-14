# s3_smartsim_c3 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 171.328 | 37.601 | 291.587 | 0.003 | 1.0 | 24/24 |
| `solver_ml_provider_call` | 171.326 | 37.575 | 291.585 | 0.007 | 1.0 | 24/24 |
| `app_provider_inference` | 164.250 | 30.437 | 284.428 | 0.003 | 1.0 | 24/24 |
| `smartsim_library_static_step` | 164.247 | 30.434 | 284.426 | 0.310 | 1.0 | 24/24 |
| `MPI_Allreduce` | 124.122 | 258.966 | 258.966 | 124.122 | 36.9 | 24/24 |
| `smartsim_token_wait` | 117.931 | 0.000 | 238.205 | 0.002 | 0.9 | 21/24 |
| `MPI_Recv` | 117.929 | 0.000 | 238.203 | 117.929 | 0.9 | 21/24 |
| `smartsim_put_tensor` | 33.244 | 22.243 | 38.550 | 33.244 | 2.0 | 24/24 |
| `smartsim_run_model` | 9.358 | 6.416 | 12.077 | 9.358 | 1.0 | 24/24 |
| `app_prepare_input` | 6.615 | 6.653 | 6.722 | 6.615 | 1.0 | 24/24 |
| `solver_step_compute` | 5.848 | 5.527 | 6.240 | 5.848 | 1.0 | 24/24 |
| `smartsim_unpack_tensor` | 3.394 | 1.506 | 4.667 | 3.394 | 1.0 | 24/24 |
| `MPI_Bcast` | 0.864 | 0.014 | 1.029 | 0.864 | 2.0 | 24/24 |
| `app_finalize_output` | 0.454 | 0.478 | 0.502 | 0.454 | 1.0 | 24/24 |
| `MPI_Waitall` | 0.163 | 0.103 | 0.285 | 0.163 | 2.0 | 24/24 |
| `MPI_Gatherv` | 0.060 | 0.061 | 0.091 | 0.060 | 0.0 | 24/24 |
| `MPI_Isend` | 0.028 | 0.028 | 0.031 | 0.028 | 8.1 | 24/24 |
| `MPI_Irecv` | 0.019 | 0.021 | 0.021 | 0.019 | 8.1 | 24/24 |
| `MPI_Send` | 0.007 | 0.006 | 0.009 | 0.007 | 0.9 | 21/24 |
| `smartsim_chunk_plan` | 0.003 | 0.003 | 0.004 | 0.003 | 1.0 | 24/24 |
| `MPI_Gather` | 0.000 | 0.000 | 0.001 | 0.000 | 0.0 | 24/24 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 36083.525 | 1.0 | 24/24 |
| `solver_setup` | 4154.155 | 1.0 | 24/24 |
| `solver_step_ml_warmup` | 691.637 | 1.0 | 24/24 |
| `solver_teardown` | 39.153 | 1.0 | 24/24 |
