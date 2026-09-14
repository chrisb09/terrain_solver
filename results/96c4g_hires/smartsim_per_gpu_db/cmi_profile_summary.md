# smartsim_per_gpu_db (96c4g_hires)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 139.114 | 149.157 | 161.014 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 139.111 | 149.099 | 161.012 | 0.005 | 1.0 | 96/96 |
| `app_provider_inference` | 137.443 | 147.724 | 159.286 | 0.003 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 137.441 | 147.721 | 159.284 | 0.106 | 1.0 | 96/96 |
| `smartsim_put_tensor` | 78.142 | 82.575 | 93.856 | 78.142 | 2.0 | 96/96 |
| `MPI_Allreduce` | 61.777 | 49.216 | 81.346 | 61.777 | 36.9 | 96/96 |
| `smartsim_run_model` | 47.686 | 52.071 | 54.786 | 47.686 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 11.506 | 12.704 | 15.667 | 11.506 | 1.0 | 96/96 |
| `MPI_Bcast` | 4.758 | 0.020 | 5.271 | 4.758 | 2.0 | 96/96 |
| `app_prepare_input` | 1.538 | 1.235 | 1.884 | 1.538 | 1.0 | 96/96 |
| `solver_step_compute` | 1.222 | 0.801 | 1.842 | 1.222 | 1.0 | 96/96 |
| `MPI_Waitall` | 0.406 | 1.109 | 2.317 | 0.406 | 2.0 | 96/96 |
| `app_finalize_output` | 0.125 | 0.134 | 0.236 | 0.125 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.091 | 0.074 | 0.127 | 0.091 | 0.0 | 96/96 |
| `MPI_Isend` | 0.030 | 0.027 | 0.033 | 0.030 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.019 | 0.019 | 0.020 | 0.019 | 8.1 | 96/96 |
| `smartsim_chunk_plan` | 0.002 | 0.003 | 0.005 | 0.002 | 1.0 | 96/96 |
| `MPI_Gather` | 0.000 | 0.001 | 0.001 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 28909.885 | 1.0 | 96/96 |
| `solver_setup` | 5552.187 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 425.392 | 1.0 | 96/96 |
| `solver_teardown` | 49.474 | 1.0 | 96/96 |
