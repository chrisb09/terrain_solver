# smartsim_c3 (96c4g_hires)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `MPI_Allreduce` | 265.317 | 480.867 | 481.078 | 265.317 | 36.9 | 96/96 |
| `solver_step_ml_steady` | 237.409 | 23.490 | 474.578 | 0.003 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 237.406 | 23.463 | 474.576 | 0.007 | 1.0 | 96/96 |
| `app_provider_inference` | 235.433 | 21.622 | 472.206 | 0.004 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 235.429 | 21.618 | 472.204 | 0.524 | 1.0 | 96/96 |
| `smartsim_token_wait` | 213.555 | 0.000 | 452.332 | 0.002 | 1.0 | 93/96 |
| `MPI_Recv` | 213.553 | 0.000 | 452.329 | 213.553 | 1.0 | 93/96 |
| `smartsim_put_tensor` | 13.236 | 12.361 | 16.844 | 13.236 | 2.0 | 96/96 |
| `smartsim_run_model` | 5.616 | 6.070 | 7.660 | 5.616 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 2.488 | 3.013 | 3.907 | 2.488 | 1.0 | 96/96 |
| `app_prepare_input` | 1.789 | 1.522 | 2.222 | 1.789 | 1.0 | 96/96 |
| `solver_step_compute` | 1.323 | 0.905 | 1.955 | 1.323 | 1.0 | 96/96 |
| `MPI_Bcast` | 0.916 | 0.025 | 1.043 | 0.916 | 2.0 | 96/96 |
| `MPI_Waitall` | 0.437 | 0.559 | 1.846 | 0.437 | 2.0 | 96/96 |
| `app_finalize_output` | 0.178 | 0.313 | 0.553 | 0.178 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.105 | 0.092 | 0.150 | 0.105 | 0.0 | 96/96 |
| `MPI_Isend` | 0.039 | 0.035 | 0.044 | 0.039 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.025 | 0.026 | 0.028 | 0.025 | 8.1 | 96/96 |
| `MPI_Send` | 0.008 | 0.008 | 0.018 | 0.008 | 1.0 | 93/96 |
| `smartsim_chunk_plan` | 0.002 | 0.003 | 0.004 | 0.002 | 1.0 | 96/96 |
| `MPI_Gather` | 0.000 | 0.001 | 0.001 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 55524.568 | 1.0 | 96/96 |
| `solver_setup` | 2102.389 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 458.004 | 1.0 | 96/96 |
| `solver_teardown` | 82.698 | 1.0 | 96/96 |
