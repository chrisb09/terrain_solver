# smartsim_c0 (96c4g_hires)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 234.483 | 235.911 | 239.681 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 234.481 | 235.888 | 239.679 | 0.004 | 1.0 | 96/96 |
| `app_provider_inference` | 232.828 | 234.497 | 237.882 | 0.003 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 232.825 | 234.494 | 237.880 | 0.105 | 1.0 | 96/96 |
| `smartsim_put_tensor` | 192.720 | 192.516 | 202.798 | 192.720 | 2.0 | 96/96 |
| `MPI_Allreduce` | 51.439 | 50.447 | 58.992 | 51.439 | 36.9 | 96/96 |
| `smartsim_run_model` | 34.219 | 34.650 | 39.475 | 34.219 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 5.779 | 7.169 | 9.998 | 5.779 | 1.0 | 96/96 |
| `app_prepare_input` | 1.526 | 1.278 | 1.880 | 1.526 | 1.0 | 96/96 |
| `solver_step_compute` | 1.216 | 0.780 | 1.825 | 1.216 | 1.0 | 96/96 |
| `MPI_Bcast` | 0.941 | 0.020 | 1.079 | 0.941 | 2.0 | 96/96 |
| `MPI_Waitall` | 0.408 | 1.135 | 1.706 | 0.408 | 2.0 | 96/96 |
| `app_finalize_output` | 0.122 | 0.108 | 0.240 | 0.122 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.090 | 0.077 | 0.128 | 0.090 | 0.0 | 96/96 |
| `MPI_Isend` | 0.030 | 0.027 | 0.033 | 0.030 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.018 | 0.020 | 0.020 | 0.018 | 8.1 | 96/96 |
| `smartsim_chunk_plan` | 0.002 | 0.003 | 0.004 | 0.002 | 1.0 | 96/96 |
| `MPI_Gather` | 0.000 | 0.001 | 0.001 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 42525.356 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 5380.566 | 1.0 | 96/96 |
| `solver_setup` | 4706.021 | 1.0 | 96/96 |
| `solver_teardown` | 40.677 | 1.0 | 96/96 |
