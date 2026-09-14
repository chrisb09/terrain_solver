# s4_smartsim_c0 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 262.105 | 200.829 | 299.496 | 0.002 | 1.0 | 24/24 |
| `solver_ml_provider_call` | 262.103 | 200.814 | 299.494 | 0.007 | 1.0 | 24/24 |
| `app_provider_inference` | 255.063 | 193.646 | 292.352 | 0.003 | 1.0 | 24/24 |
| `smartsim_library_static_step` | 255.059 | 193.643 | 292.349 | 0.132 | 1.0 | 24/24 |
| `smartsim_put_tensor` | 208.078 | 163.323 | 232.080 | 208.078 | 2.0 | 24/24 |
| `MPI_Allreduce` | 55.909 | 118.146 | 118.146 | 55.909 | 36.9 | 24/24 |
| `smartsim_run_model` | 42.008 | 20.959 | 57.608 | 42.008 | 1.0 | 24/24 |
| `app_prepare_input` | 6.597 | 6.693 | 6.700 | 6.597 | 1.0 | 24/24 |
| `solver_step_compute` | 5.769 | 5.556 | 6.109 | 5.769 | 1.0 | 24/24 |
| `smartsim_unpack_tensor` | 4.839 | 8.847 | 8.847 | 4.839 | 1.0 | 24/24 |
| `MPI_Bcast` | 1.333 | 0.012 | 1.436 | 1.333 | 2.0 | 24/24 |
| `app_finalize_output` | 0.436 | 0.469 | 0.469 | 0.436 | 1.0 | 24/24 |
| `MPI_Waitall` | 0.157 | 0.109 | 0.323 | 0.157 | 2.0 | 24/24 |
| `MPI_Gatherv` | 0.057 | 0.062 | 0.086 | 0.057 | 0.0 | 24/24 |
| `MPI_Irecv` | 0.018 | 0.020 | 0.020 | 0.018 | 8.1 | 24/24 |
| `MPI_Isend` | 0.011 | 0.010 | 0.012 | 0.011 | 8.1 | 24/24 |
| `smartsim_chunk_plan` | 0.003 | 0.002 | 0.003 | 0.003 | 1.0 | 24/24 |
| `MPI_Gather` | 0.000 | 0.000 | 0.003 | 0.000 | 0.0 | 24/24 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 37521.287 | 1.0 | 24/24 |
| `solver_setup` | 3752.600 | 1.0 | 24/24 |
| `solver_step_ml_warmup` | 779.048 | 1.0 | 24/24 |
| `solver_teardown` | 41.579 | 1.0 | 24/24 |
