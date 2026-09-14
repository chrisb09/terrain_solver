# s1_smartsim_c0 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 752.405 | 750.560 | 762.779 | 0.019 | 1.0 | 4/4 |
| `solver_ml_provider_call` | 752.386 | 750.491 | 762.776 | 0.005 | 1.0 | 4/4 |
| `app_provider_inference` | 736.681 | 734.654 | 747.288 | 0.002 | 1.0 | 4/4 |
| `smartsim_library_static_step` | 736.679 | 734.652 | 747.285 | 0.256 | 1.0 | 4/4 |
| `smartsim_put_tensor` | 636.367 | 641.993 | 648.432 | 636.367 | 2.0 | 4/4 |
| `smartsim_run_model` | 87.217 | 80.243 | 96.329 | 87.217 | 1.0 | 4/4 |
| `MPI_Allreduce` | 41.780 | 43.723 | 50.088 | 41.780 | 36.9 | 4/4 |
| `solver_step_compute` | 26.212 | 26.272 | 26.419 | 26.212 | 1.0 | 4/4 |
| `smartsim_unpack_tensor` | 12.838 | 11.977 | 15.335 | 12.838 | 1.0 | 4/4 |
| `app_prepare_input` | 12.585 | 12.591 | 12.617 | 12.585 | 1.0 | 4/4 |
| `app_finalize_output` | 3.115 | 3.241 | 3.260 | 3.115 | 1.0 | 4/4 |
| `MPI_Bcast` | 0.643 | 0.006 | 0.916 | 0.643 | 2.0 | 4/4 |
| `MPI_Waitall` | 0.133 | 0.132 | 0.134 | 0.133 | 2.0 | 4/4 |
| `MPI_Gatherv` | 0.073 | 0.044 | 0.092 | 0.073 | 0.0 | 4/4 |
| `MPI_Isend` | 0.019 | 0.019 | 0.020 | 0.019 | 8.1 | 4/4 |
| `MPI_Irecv` | 0.018 | 0.018 | 0.018 | 0.018 | 8.1 | 4/4 |
| `smartsim_chunk_plan` | 0.001 | 0.001 | 0.001 | 0.001 | 1.0 | 4/4 |
| `MPI_Gather` | 0.001 | 0.000 | 0.003 | 0.001 | 0.0 | 4/4 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 95487.600 | 1.0 | 4/4 |
| `solver_step_ml_warmup` | 6854.970 | 1.0 | 4/4 |
| `solver_setup` | 5709.690 | 1.0 | 4/4 |
| `solver_teardown` | 115.185 | 1.0 | 4/4 |
