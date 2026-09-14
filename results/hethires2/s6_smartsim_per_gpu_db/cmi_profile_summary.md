# s6_smartsim_per_gpu_db (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 155.571 | 161.568 | 167.727 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 155.569 | 161.545 | 167.725 | 0.005 | 1.0 | 96/96 |
| `app_provider_inference` | 153.856 | 159.861 | 165.996 | 0.003 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 153.853 | 159.857 | 165.993 | 0.115 | 1.0 | 96/96 |
| `smartsim_put_tensor` | 90.723 | 92.096 | 99.051 | 90.723 | 2.0 | 96/96 |
| `MPI_Allreduce` | 71.573 | 66.697 | 84.409 | 71.573 | 36.9 | 96/96 |
| `smartsim_run_model` | 52.003 | 55.819 | 57.756 | 52.003 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 11.010 | 11.766 | 13.695 | 11.010 | 1.0 | 96/96 |
| `app_prepare_input` | 1.585 | 1.536 | 1.702 | 1.585 | 1.0 | 96/96 |
| `solver_step_compute` | 1.254 | 0.962 | 1.538 | 1.254 | 1.0 | 96/96 |
| `MPI_Bcast` | 1.005 | 0.022 | 2.148 | 1.005 | 2.0 | 96/96 |
| `MPI_Waitall` | 0.497 | 0.116 | 1.589 | 0.497 | 2.0 | 96/96 |
| `app_finalize_output` | 0.123 | 0.143 | 0.215 | 0.123 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.102 | 0.089 | 0.146 | 0.102 | 0.0 | 96/96 |
| `MPI_Isend` | 0.027 | 0.027 | 0.054 | 0.027 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.019 | 0.020 | 0.021 | 0.019 | 8.1 | 96/96 |
| `smartsim_chunk_plan` | 0.002 | 0.003 | 0.003 | 0.002 | 1.0 | 96/96 |
| `MPI_Gather` | 0.000 | 0.001 | 0.001 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 31287.102 | 1.0 | 96/96 |
| `solver_setup` | 5558.495 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 454.200 | 1.0 | 96/96 |
| `solver_teardown` | 45.703 | 1.0 | 96/96 |
