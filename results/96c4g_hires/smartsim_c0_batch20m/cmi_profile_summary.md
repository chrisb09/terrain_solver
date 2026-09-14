# smartsim_c0_batch20m (96c4g_hires)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 260.927 | 254.711 | 269.748 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 260.925 | 254.688 | 269.746 | 0.005 | 1.0 | 96/96 |
| `app_provider_inference` | 259.261 | 253.249 | 268.049 | 0.002 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 259.259 | 253.246 | 268.047 | 0.103 | 1.0 | 96/96 |
| `smartsim_put_tensor` | 201.274 | 194.780 | 212.493 | 201.274 | 2.0 | 96/96 |
| `MPI_Allreduce` | 81.009 | 88.700 | 89.617 | 81.009 | 36.9 | 96/96 |
| `smartsim_run_model` | 50.847 | 48.968 | 57.582 | 50.847 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 7.032 | 9.340 | 9.657 | 7.032 | 1.0 | 96/96 |
| `app_prepare_input` | 1.535 | 1.316 | 1.880 | 1.535 | 1.0 | 96/96 |
| `solver_step_compute` | 1.236 | 0.812 | 1.876 | 1.236 | 1.0 | 96/96 |
| `MPI_Bcast` | 1.072 | 0.019 | 1.471 | 1.072 | 2.0 | 96/96 |
| `MPI_Waitall` | 0.393 | 0.887 | 1.676 | 0.393 | 2.0 | 96/96 |
| `app_finalize_output` | 0.125 | 0.117 | 0.171 | 0.125 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.096 | 0.083 | 0.136 | 0.096 | 0.0 | 96/96 |
| `MPI_Isend` | 0.029 | 0.027 | 0.032 | 0.029 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.018 | 0.019 | 0.020 | 0.018 | 8.1 | 96/96 |
| `smartsim_chunk_plan` | 0.002 | 0.003 | 0.004 | 0.002 | 1.0 | 96/96 |
| `MPI_Gather` | 0.000 | 0.001 | 0.001 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 48844.509 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 5571.288 | 1.0 | 96/96 |
| `solver_setup` | 5379.629 | 1.0 | 96/96 |
| `solver_teardown` | 49.720 | 1.0 | 96/96 |
