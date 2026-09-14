# smartsim_c1 (96c4g)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `MPI_Allreduce` | 286.545 | 521.975 | 521.975 | 286.545 | 44.6 | 96/96 |
| `solver_step_ml_steady` | 231.957 | 6.705 | 468.028 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 231.955 | 6.690 | 468.026 | 0.004 | 1.0 | 96/96 |
| `app_provider_inference` | 231.651 | 6.426 | 467.656 | 0.002 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 231.649 | 6.424 | 467.653 | 0.527 | 1.0 | 96/96 |
| `smartsim_token_wait` | 223.214 | 0.000 | 457.703 | 0.002 | 1.0 | 95/96 |
| `MPI_Recv` | 223.212 | 0.000 | 457.701 | 223.212 | 1.0 | 95/96 |
| `smartsim_put_tensor` | 4.327 | 3.159 | 6.283 | 4.327 | 2.0 | 96/96 |
| `MPI_Bcast` | 3.389 | 0.025 | 3.903 | 3.389 | 2.4 | 96/96 |
| `smartsim_run_model` | 2.850 | 2.548 | 3.734 | 2.850 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 0.721 | 0.606 | 0.932 | 0.721 | 1.0 | 96/96 |
| `MPI_Waitall` | 0.471 | 0.062 | 2.121 | 0.471 | 2.2 | 96/96 |
| `solver_step_compute` | 0.409 | 0.293 | 0.702 | 0.409 | 1.1 | 96/96 |
| `app_prepare_input` | 0.259 | 0.234 | 0.434 | 0.259 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.236 | 0.366 | 0.366 | 0.236 | 0.1 | 96/96 |
| `MPI_Isend` | 0.048 | 0.044 | 0.070 | 0.048 | 8.8 | 96/96 |
| `app_finalize_output` | 0.040 | 0.026 | 0.077 | 0.040 | 1.0 | 96/96 |
| `MPI_Irecv` | 0.021 | 0.019 | 0.026 | 0.021 | 8.8 | 96/96 |
| `MPI_Send` | 0.008 | 0.007 | 0.012 | 0.008 | 1.0 | 95/96 |
| `MPI_Gather` | 0.004 | 0.064 | 0.064 | 0.004 | 0.1 | 96/96 |
| `smartsim_chunk_plan` | 0.001 | 0.001 | 0.004 | 0.001 | 1.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 9617.551 | 1.0 | 96/96 |
| `solver_setup` | 1840.946 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 377.057 | 1.0 | 96/96 |
| `solver_teardown` | 49.741 | 1.0 | 96/96 |
