# s5_smartsim_c3 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 148.472 | 27.793 | 265.183 | 0.003 | 1.0 | 48/48 |
| `solver_ml_provider_call` | 148.469 | 27.723 | 265.181 | 0.005 | 1.0 | 48/48 |
| `app_provider_inference` | 145.180 | 24.801 | 261.561 | 0.002 | 1.0 | 48/48 |
| `smartsim_library_static_step` | 145.177 | 24.797 | 261.560 | 0.225 | 1.0 | 48/48 |
| `MPI_Allreduce` | 128.613 | 251.061 | 251.061 | 128.613 | 36.9 | 48/48 |
| `smartsim_token_wait` | 121.089 | 0.000 | 240.981 | 0.002 | 0.9 | 45/48 |
| `MPI_Recv` | 121.087 | 0.000 | 240.979 | 121.087 | 0.9 | 45/48 |
| `smartsim_put_tensor` | 15.514 | 16.485 | 20.326 | 15.514 | 2.0 | 48/48 |
| `smartsim_run_model` | 6.664 | 6.198 | 9.567 | 6.664 | 1.0 | 48/48 |
| `app_prepare_input` | 3.045 | 2.716 | 3.449 | 3.045 | 1.0 | 48/48 |
| `solver_step_compute` | 2.533 | 1.921 | 3.269 | 2.533 | 1.0 | 48/48 |
| `smartsim_unpack_tensor` | 1.676 | 1.399 | 3.333 | 1.676 | 1.0 | 48/48 |
| `MPI_Bcast` | 0.846 | 0.046 | 1.002 | 0.846 | 2.0 | 48/48 |
| `MPI_Waitall` | 0.241 | 0.230 | 0.497 | 0.241 | 2.0 | 48/48 |
| `app_finalize_output` | 0.239 | 0.201 | 0.284 | 0.239 | 1.0 | 48/48 |
| `MPI_Gatherv` | 0.097 | 0.153 | 0.192 | 0.097 | 0.0 | 48/48 |
| `MPI_Isend` | 0.019 | 0.028 | 0.057 | 0.019 | 8.1 | 48/48 |
| `MPI_Irecv` | 0.019 | 0.021 | 0.042 | 0.019 | 8.1 | 48/48 |
| `MPI_Send` | 0.008 | 0.008 | 0.018 | 0.008 | 0.9 | 45/48 |
| `smartsim_chunk_plan` | 0.002 | 0.003 | 0.003 | 0.002 | 1.0 | 48/48 |
| `MPI_Gather` | 0.000 | 0.002 | 0.002 | 0.000 | 0.0 | 48/48 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 41584.142 | 1.0 | 48/48 |
| `solver_setup` | 4843.572 | 1.0 | 48/48 |
| `solver_step_ml_warmup` | 385.269 | 1.0 | 48/48 |
| `solver_teardown` | 39.284 | 1.0 | 48/48 |
