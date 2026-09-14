# smartsim_c3 (96c4g)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `MPI_Allreduce` | 142.421 | 239.735 | 240.240 | 142.421 | 44.6 | 96/96 |
| `solver_step_ml_steady` | 106.795 | 11.878 | 212.409 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 106.793 | 11.865 | 212.407 | 0.003 | 1.0 | 96/96 |
| `app_provider_inference` | 106.532 | 11.627 | 212.068 | 0.002 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 106.530 | 11.625 | 212.067 | 0.284 | 1.0 | 96/96 |
| `smartsim_token_wait` | 94.648 | 0.000 | 199.768 | 0.002 | 1.0 | 93/96 |
| `MPI_Recv` | 94.646 | 0.000 | 199.766 | 94.646 | 1.0 | 93/96 |
| `smartsim_put_tensor` | 5.903 | 5.037 | 9.868 | 5.903 | 2.0 | 96/96 |
| `smartsim_run_model` | 4.326 | 4.718 | 6.933 | 4.326 | 1.0 | 96/96 |
| `MPI_Bcast` | 3.506 | 0.022 | 3.598 | 3.506 | 2.4 | 96/96 |
| `smartsim_unpack_tensor` | 1.361 | 1.760 | 2.727 | 1.361 | 1.0 | 96/96 |
| `MPI_Waitall` | 0.439 | 0.480 | 2.950 | 0.439 | 2.2 | 96/96 |
| `solver_step_compute` | 0.406 | 0.192 | 0.670 | 0.406 | 1.1 | 96/96 |
| `app_prepare_input` | 0.221 | 0.211 | 0.422 | 0.221 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.219 | 0.659 | 0.659 | 0.219 | 0.1 | 96/96 |
| `MPI_Isend` | 0.046 | 0.037 | 0.078 | 0.046 | 8.8 | 96/96 |
| `app_finalize_output` | 0.036 | 0.024 | 0.097 | 0.036 | 1.0 | 96/96 |
| `MPI_Irecv` | 0.018 | 0.018 | 0.021 | 0.018 | 8.8 | 96/96 |
| `MPI_Send` | 0.007 | 0.006 | 0.017 | 0.007 | 1.0 | 93/96 |
| `MPI_Gather` | 0.003 | 0.039 | 0.039 | 0.003 | 0.1 | 96/96 |
| `smartsim_chunk_plan` | 0.001 | 0.001 | 0.002 | 0.001 | 1.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 7086.926 | 1.0 | 96/96 |
| `solver_setup` | 2055.848 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 274.211 | 1.0 | 96/96 |
| `solver_teardown` | 26.772 | 1.0 | 96/96 |
