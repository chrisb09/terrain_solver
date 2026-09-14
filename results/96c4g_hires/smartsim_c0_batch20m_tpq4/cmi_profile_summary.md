# smartsim_c0_batch20m_tpq4 (96c4g_hires)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 283.640 | 284.200 | 292.359 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 283.638 | 284.175 | 292.357 | 0.005 | 1.0 | 96/96 |
| `app_provider_inference` | 281.981 | 282.746 | 290.445 | 0.002 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 281.979 | 282.743 | 290.443 | 0.100 | 1.0 | 96/96 |
| `smartsim_put_tensor` | 218.213 | 215.269 | 226.972 | 218.213 | 2.0 | 96/96 |
| `MPI_Allreduce` | 75.503 | 76.509 | 85.497 | 75.503 | 36.9 | 96/96 |
| `smartsim_run_model` | 56.980 | 59.664 | 62.908 | 56.980 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 6.684 | 7.644 | 10.201 | 6.684 | 1.0 | 96/96 |
| `app_prepare_input` | 1.527 | 1.313 | 1.873 | 1.527 | 1.0 | 96/96 |
| `solver_step_compute` | 1.222 | 0.784 | 1.847 | 1.222 | 1.0 | 96/96 |
| `MPI_Bcast` | 1.085 | 0.021 | 1.243 | 1.085 | 2.0 | 96/96 |
| `MPI_Waitall` | 0.403 | 0.753 | 1.406 | 0.403 | 2.0 | 96/96 |
| `app_finalize_output` | 0.125 | 0.110 | 0.181 | 0.125 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.090 | 0.074 | 0.127 | 0.090 | 0.0 | 96/96 |
| `MPI_Isend` | 0.029 | 0.027 | 0.031 | 0.029 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.018 | 0.019 | 0.020 | 0.018 | 8.1 | 96/96 |
| `smartsim_chunk_plan` | 0.002 | 0.003 | 0.003 | 0.002 | 1.0 | 96/96 |
| `MPI_Gather` | 0.000 | 0.001 | 0.001 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 41079.907 | 1.0 | 96/96 |
| `solver_setup` | 2040.329 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 538.928 | 1.0 | 96/96 |
| `solver_teardown` | 41.044 | 1.0 | 96/96 |
