# s7_smartsim_c3 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `MPI_Allreduce` | 342.658 | 627.224 | 628.088 | 342.658 | 36.9 | 192/192 |
| `solver_step_ml_steady` | 307.479 | 18.553 | 597.374 | 0.002 | 1.0 | 192/192 |
| `solver_ml_provider_call` | 307.477 | 18.494 | 597.372 | 0.003 | 1.0 | 192/192 |
| `app_provider_inference` | 306.780 | 17.996 | 596.465 | 0.001 | 1.0 | 192/192 |
| `smartsim_library_static_step` | 306.778 | 17.994 | 596.464 | 0.341 | 1.0 | 192/192 |
| `smartsim_token_wait` | 291.196 | 0.000 | 583.148 | 0.010 | 1.0 | 189/192 |
| `MPI_Recv` | 291.186 | 0.000 | 583.130 | 291.186 | 1.0 | 189/192 |
| `smartsim_put_tensor` | 8.501 | 10.507 | 16.993 | 8.501 | 2.0 | 192/192 |
| `MPI_Bcast` | 5.497 | 0.015 | 8.670 | 5.497 | 2.0 | 192/192 |
| `smartsim_run_model` | 5.220 | 4.851 | 10.395 | 5.220 | 1.0 | 192/192 |
| `smartsim_unpack_tensor` | 1.512 | 1.597 | 4.255 | 1.512 | 1.0 | 192/192 |
| `app_prepare_input` | 0.631 | 0.440 | 0.875 | 0.631 | 1.0 | 192/192 |
| `solver_step_compute` | 0.584 | 0.460 | 1.179 | 0.584 | 1.0 | 192/192 |
| `MPI_Waitall` | 0.436 | 2.890 | 7.864 | 0.436 | 2.0 | 192/192 |
| `MPI_Gatherv` | 0.150 | 0.345 | 0.392 | 0.150 | 0.0 | 192/192 |
| `app_finalize_output` | 0.063 | 0.055 | 0.174 | 0.063 | 1.0 | 192/192 |
| `MPI_Irecv` | 0.018 | 0.028 | 0.031 | 0.018 | 8.1 | 192/192 |
| `MPI_Isend` | 0.015 | 0.047 | 0.047 | 0.015 | 8.1 | 192/192 |
| `MPI_Send` | 0.006 | 0.004 | 0.022 | 0.006 | 1.0 | 189/192 |
| `smartsim_chunk_plan` | 0.001 | 0.002 | 0.003 | 0.001 | 1.0 | 192/192 |
| `MPI_Gather` | 0.000 | 0.004 | 0.004 | 0.000 | 0.0 | 192/192 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 81103.490 | 1.0 | 192/192 |
| `solver_setup` | 8819.167 | 1.0 | 192/192 |
| `solver_step_ml_warmup` | 1278.664 | 1.0 | 192/192 |
| `solver_teardown` | 54.370 | 1.0 | 192/192 |
