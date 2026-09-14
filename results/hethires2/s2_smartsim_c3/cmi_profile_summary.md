# s2_smartsim_c3 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 564.324 | 491.546 | 777.109 | 0.011 | 1.0 | 4/4 |
| `solver_ml_provider_call` | 564.313 | 491.509 | 777.107 | 0.004 | 1.0 | 4/4 |
| `app_provider_inference` | 549.122 | 476.177 | 761.986 | 0.002 | 1.0 | 4/4 |
| `smartsim_library_static_step` | 549.120 | 476.175 | 761.984 | 0.285 | 1.0 | 4/4 |
| `smartsim_put_tensor` | 391.644 | 427.716 | 455.893 | 391.644 | 2.0 | 4/4 |
| `MPI_Allreduce` | 214.415 | 287.224 | 320.624 | 214.415 | 36.9 | 4/4 |
| `smartsim_token_wait` | 107.017 | 0.000 | 428.066 | 0.001 | 0.2 | 1/4 |
| `MPI_Recv` | 107.016 | 0.000 | 428.064 | 107.016 | 0.2 | 1/4 |
| `smartsim_run_model` | 41.179 | 38.155 | 50.940 | 41.179 | 1.0 | 4/4 |
| `solver_step_compute` | 21.836 | 22.028 | 22.028 | 21.836 | 1.0 | 4/4 |
| `app_prepare_input` | 12.324 | 12.405 | 12.405 | 12.324 | 1.0 | 4/4 |
| `smartsim_unpack_tensor` | 8.993 | 9.945 | 11.781 | 8.993 | 1.0 | 4/4 |
| `app_finalize_output` | 2.864 | 2.923 | 2.949 | 2.864 | 1.0 | 4/4 |
| `MPI_Bcast` | 0.488 | 0.006 | 0.658 | 0.488 | 2.0 | 4/4 |
| `MPI_Waitall` | 0.111 | 0.122 | 0.136 | 0.111 | 2.0 | 4/4 |
| `MPI_Gatherv` | 0.067 | 0.045 | 0.087 | 0.067 | 0.0 | 4/4 |
| `MPI_Irecv` | 0.016 | 0.017 | 0.017 | 0.016 | 8.1 | 4/4 |
| `MPI_Isend` | 0.008 | 0.008 | 0.009 | 0.008 | 8.1 | 4/4 |
| `MPI_Send` | 0.002 | 0.008 | 0.008 | 0.002 | 0.2 | 1/4 |
| `smartsim_chunk_plan` | 0.001 | 0.001 | 0.001 | 0.001 | 1.0 | 4/4 |
| `MPI_Gather` | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 | 4/4 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 85001.700 | 1.0 | 4/4 |
| `solver_setup` | 3752.628 | 1.0 | 4/4 |
| `solver_step_ml_warmup` | 758.762 | 1.0 | 4/4 |
| `solver_teardown` | 38.433 | 1.0 | 4/4 |
