# s1_smartsim_c3 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 568.154 | 534.014 | 792.622 | 0.011 | 1.0 | 4/4 |
| `solver_ml_provider_call` | 568.143 | 533.976 | 792.620 | 0.004 | 1.0 | 4/4 |
| `app_provider_inference` | 552.962 | 518.587 | 777.613 | 0.002 | 1.0 | 4/4 |
| `smartsim_library_static_step` | 552.960 | 518.585 | 777.612 | 0.242 | 1.0 | 4/4 |
| `smartsim_put_tensor` | 380.144 | 462.732 | 462.732 | 380.144 | 2.0 | 4/4 |
| `MPI_Allreduce` | 225.396 | 259.674 | 366.371 | 225.396 | 36.9 | 4/4 |
| `smartsim_token_wait` | 115.762 | 0.000 | 463.048 | 0.000 | 0.2 | 1/4 |
| `MPI_Recv` | 115.761 | 0.000 | 463.046 | 115.761 | 0.2 | 1/4 |
| `smartsim_run_model` | 47.993 | 47.760 | 55.308 | 47.993 | 1.0 | 4/4 |
| `solver_step_compute` | 22.013 | 21.985 | 22.176 | 22.013 | 1.0 | 4/4 |
| `app_prepare_input` | 12.284 | 12.365 | 12.365 | 12.284 | 1.0 | 4/4 |
| `smartsim_unpack_tensor` | 8.815 | 7.808 | 14.147 | 8.815 | 1.0 | 4/4 |
| `app_finalize_output` | 2.894 | 3.019 | 3.019 | 2.894 | 1.0 | 4/4 |
| `MPI_Bcast` | 0.471 | 0.007 | 0.631 | 0.471 | 2.0 | 4/4 |
| `MPI_Waitall` | 0.078 | 0.095 | 0.101 | 0.078 | 2.0 | 4/4 |
| `MPI_Gatherv` | 0.068 | 0.043 | 0.090 | 0.068 | 0.0 | 4/4 |
| `MPI_Isend` | 0.018 | 0.019 | 0.019 | 0.018 | 8.1 | 4/4 |
| `MPI_Irecv` | 0.017 | 0.018 | 0.018 | 0.017 | 8.1 | 4/4 |
| `MPI_Send` | 0.002 | 0.008 | 0.008 | 0.002 | 0.2 | 1/4 |
| `smartsim_chunk_plan` | 0.001 | 0.001 | 0.001 | 0.001 | 1.0 | 4/4 |
| `MPI_Gather` | 0.000 | 0.001 | 0.001 | 0.000 | 0.0 | 4/4 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 94068.075 | 1.0 | 4/4 |
| `solver_setup` | 6185.645 | 1.0 | 4/4 |
| `solver_step_ml_warmup` | 5771.645 | 1.0 | 4/4 |
| `solver_teardown` | 37.116 | 1.0 | 4/4 |
