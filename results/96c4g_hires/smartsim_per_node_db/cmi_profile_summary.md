# smartsim_per_node_db (96c4g_hires)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 236.582 | 231.013 | 242.315 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 236.580 | 230.992 | 242.313 | 0.004 | 1.0 | 96/96 |
| `app_provider_inference` | 234.928 | 229.616 | 240.280 | 0.003 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 234.925 | 229.612 | 240.279 | 0.101 | 1.0 | 96/96 |
| `smartsim_put_tensor` | 195.139 | 188.646 | 205.531 | 195.139 | 2.0 | 96/96 |
| `MPI_Allreduce` | 51.730 | 58.849 | 58.849 | 51.730 | 36.9 | 96/96 |
| `smartsim_run_model` | 34.330 | 35.093 | 38.554 | 34.330 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 5.353 | 5.721 | 8.529 | 5.353 | 1.0 | 96/96 |
| `app_prepare_input` | 1.526 | 1.261 | 1.894 | 1.526 | 1.0 | 96/96 |
| `solver_step_compute` | 1.212 | 0.775 | 1.872 | 1.212 | 1.0 | 96/96 |
| `MPI_Bcast` | 0.890 | 0.020 | 1.183 | 0.890 | 2.0 | 96/96 |
| `MPI_Waitall` | 0.405 | 0.416 | 0.950 | 0.405 | 2.0 | 96/96 |
| `app_finalize_output` | 0.122 | 0.110 | 0.194 | 0.122 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.088 | 0.073 | 0.124 | 0.088 | 0.0 | 96/96 |
| `MPI_Isend` | 0.030 | 0.027 | 0.034 | 0.030 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.019 | 0.019 | 0.026 | 0.019 | 8.1 | 96/96 |
| `smartsim_chunk_plan` | 0.002 | 0.003 | 0.003 | 0.002 | 1.0 | 96/96 |
| `MPI_Gather` | 0.000 | 0.001 | 0.001 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 34655.875 | 1.0 | 96/96 |
| `solver_setup` | 2463.077 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 875.381 | 1.0 | 96/96 |
| `solver_teardown` | 30.772 | 1.0 | 96/96 |
