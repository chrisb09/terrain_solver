# smartsim_per_node_db (96c4g)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 99.949 | 114.040 | 123.949 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 99.947 | 114.014 | 123.947 | 0.003 | 1.0 | 96/96 |
| `app_provider_inference` | 99.658 | 113.803 | 123.662 | 0.002 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 99.657 | 113.802 | 123.660 | 0.168 | 1.0 | 96/96 |
| `smartsim_put_tensor` | 72.366 | 74.924 | 88.330 | 72.366 | 2.0 | 96/96 |
| `MPI_Allreduce` | 63.294 | 48.725 | 83.395 | 63.294 | 44.6 | 96/96 |
| `smartsim_run_model` | 22.610 | 33.826 | 35.355 | 22.610 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 4.512 | 4.829 | 13.301 | 4.512 | 1.0 | 96/96 |
| `MPI_Bcast` | 2.679 | 0.018 | 3.659 | 2.679 | 2.4 | 96/96 |
| `solver_step_compute` | 0.424 | 0.291 | 1.994 | 0.424 | 1.1 | 96/96 |
| `MPI_Waitall` | 0.374 | 0.455 | 2.132 | 0.374 | 2.2 | 96/96 |
| `app_prepare_input` | 0.254 | 0.182 | 0.592 | 0.254 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.141 | 0.219 | 0.219 | 0.141 | 0.1 | 96/96 |
| `MPI_Isend` | 0.035 | 0.027 | 0.043 | 0.035 | 8.8 | 96/96 |
| `app_finalize_output` | 0.032 | 0.025 | 0.060 | 0.032 | 1.0 | 96/96 |
| `MPI_Irecv` | 0.015 | 0.014 | 0.017 | 0.015 | 8.8 | 96/96 |
| `smartsim_chunk_plan` | 0.001 | 0.001 | 0.001 | 0.001 | 1.0 | 96/96 |
| `MPI_Gather` | 0.001 | 0.011 | 0.011 | 0.001 | 0.1 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 14570.542 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 5259.914 | 1.0 | 96/96 |
| `solver_setup` | 4362.125 | 1.0 | 96/96 |
| `solver_teardown` | 37.343 | 1.0 | 96/96 |
