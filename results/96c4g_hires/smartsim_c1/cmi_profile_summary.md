# smartsim_c1 (96c4g_hires)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `MPI_Allreduce` | 622.962 | 1204.500 | 1204.500 | 622.962 | 36.9 | 96/96 |
| `solver_step_ml_steady` | 593.287 | 18.122 | 1180.780 | 0.003 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 593.284 | 18.095 | 1180.780 | 0.007 | 1.0 | 96/96 |
| `app_provider_inference` | 591.168 | 16.194 | 1178.340 | 0.003 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 591.164 | 16.189 | 1178.340 | 0.660 | 1.0 | 96/96 |
| `smartsim_token_wait` | 573.068 | 0.000 | 1157.230 | 0.003 | 1.0 | 95/96 |
| `MPI_Recv` | 573.065 | 0.000 | 1157.230 | 573.065 | 1.0 | 95/96 |
| `smartsim_put_tensor` | 11.578 | 9.918 | 15.068 | 11.578 | 2.0 | 96/96 |
| `smartsim_run_model` | 4.281 | 4.920 | 5.463 | 4.281 | 1.0 | 96/96 |
| `app_prepare_input` | 1.940 | 1.751 | 2.274 | 1.940 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 1.566 | 1.127 | 2.139 | 1.566 | 1.0 | 96/96 |
| `solver_step_compute` | 1.355 | 1.003 | 1.953 | 1.355 | 1.0 | 96/96 |
| `MPI_Bcast` | 1.182 | 0.031 | 1.508 | 1.182 | 2.0 | 96/96 |
| `MPI_Waitall` | 0.474 | 0.489 | 1.885 | 0.474 | 2.0 | 96/96 |
| `app_finalize_output` | 0.169 | 0.142 | 0.226 | 0.169 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.151 | 0.139 | 0.218 | 0.151 | 0.0 | 96/96 |
| `MPI_Isend` | 0.044 | 0.042 | 0.094 | 0.044 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.028 | 0.031 | 0.032 | 0.028 | 8.1 | 96/96 |
| `MPI_Send` | 0.009 | 0.008 | 0.013 | 0.009 | 1.0 | 95/96 |
| `smartsim_chunk_plan` | 0.002 | 0.003 | 0.004 | 0.002 | 1.0 | 96/96 |
| `MPI_Gather` | 0.000 | 0.001 | 0.001 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 127322.594 | 1.0 | 96/96 |
| `solver_setup` | 2095.804 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 983.395 | 1.0 | 96/96 |
| `solver_teardown` | 46.691 | 1.0 | 96/96 |
