# s6_smartsim_c3 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `MPI_Allreduce` | 317.343 | 578.404 | 584.860 | 317.343 | 36.9 | 96/96 |
| `solver_step_ml_steady` | 299.307 | 39.857 | 570.246 | 0.006 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 299.301 | 39.833 | 570.243 | 0.007 | 1.0 | 96/96 |
| `app_provider_inference` | 297.138 | 37.727 | 567.850 | 0.004 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 297.134 | 37.723 | 567.846 | 0.629 | 1.0 | 96/96 |
| `smartsim_token_wait` | 268.780 | 0.000 | 542.221 | 0.016 | 1.0 | 93/96 |
| `MPI_Recv` | 268.764 | 0.000 | 542.217 | 268.764 | 1.0 | 93/96 |
| `smartsim_put_tensor` | 16.744 | 20.056 | 20.383 | 16.744 | 2.0 | 96/96 |
| `smartsim_run_model` | 7.224 | 11.417 | 11.656 | 7.224 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 3.742 | 6.039 | 8.146 | 3.742 | 1.0 | 96/96 |
| `app_prepare_input` | 1.957 | 1.923 | 2.325 | 1.957 | 1.0 | 96/96 |
| `solver_step_compute` | 1.365 | 1.140 | 1.681 | 1.365 | 1.0 | 96/96 |
| `MPI_Bcast` | 1.102 | 0.028 | 1.655 | 1.102 | 2.0 | 96/96 |
| `MPI_Waitall` | 0.525 | 0.500 | 3.068 | 0.525 | 2.0 | 96/96 |
| `app_finalize_output` | 0.199 | 0.175 | 0.672 | 0.199 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.139 | 0.129 | 0.204 | 0.139 | 0.0 | 96/96 |
| `MPI_Isend` | 0.035 | 0.036 | 0.040 | 0.035 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.027 | 0.031 | 0.128 | 0.027 | 8.1 | 96/96 |
| `MPI_Send` | 0.012 | 0.009 | 0.245 | 0.012 | 1.0 | 93/96 |
| `smartsim_chunk_plan` | 0.002 | 0.002 | 0.004 | 0.002 | 1.0 | 96/96 |
| `MPI_Gather` | 0.000 | 0.003 | 0.004 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 66722.135 | 1.0 | 96/96 |
| `solver_setup` | 2095.884 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 439.649 | 1.0 | 96/96 |
| `solver_teardown` | 45.051 | 1.0 | 96/96 |
