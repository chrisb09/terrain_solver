# s4_smartsim_c3 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 185.031 | 45.620 | 316.415 | 0.005 | 1.0 | 24/24 |
| `solver_ml_provider_call` | 185.027 | 45.557 | 316.413 | 0.007 | 1.0 | 24/24 |
| `app_provider_inference` | 177.985 | 38.339 | 309.256 | 0.003 | 1.0 | 24/24 |
| `smartsim_library_static_step` | 177.982 | 38.337 | 309.253 | 0.292 | 1.0 | 24/24 |
| `MPI_Allreduce` | 136.804 | 277.243 | 277.243 | 136.804 | 36.9 | 24/24 |
| `smartsim_token_wait` | 129.643 | 0.000 | 262.550 | 0.002 | 0.9 | 21/24 |
| `MPI_Recv` | 129.641 | 0.000 | 262.547 | 129.641 | 0.9 | 21/24 |
| `smartsim_put_tensor` | 36.612 | 27.873 | 39.869 | 36.612 | 2.0 | 24/24 |
| `smartsim_run_model` | 8.465 | 7.663 | 11.007 | 8.465 | 1.0 | 24/24 |
| `app_prepare_input` | 6.577 | 6.742 | 6.742 | 6.577 | 1.0 | 24/24 |
| `solver_step_compute` | 5.858 | 5.737 | 6.196 | 5.858 | 1.0 | 24/24 |
| `smartsim_unpack_tensor` | 2.961 | 2.187 | 3.322 | 2.961 | 1.0 | 24/24 |
| `MPI_Bcast` | 0.957 | 0.012 | 1.157 | 0.957 | 2.0 | 24/24 |
| `app_finalize_output` | 0.457 | 0.471 | 0.518 | 0.457 | 1.0 | 24/24 |
| `MPI_Waitall` | 0.163 | 0.113 | 0.329 | 0.163 | 2.0 | 24/24 |
| `MPI_Gatherv` | 0.058 | 0.062 | 0.090 | 0.058 | 0.0 | 24/24 |
| `MPI_Irecv` | 0.019 | 0.021 | 0.033 | 0.019 | 8.1 | 24/24 |
| `MPI_Isend` | 0.011 | 0.011 | 0.012 | 0.011 | 8.1 | 24/24 |
| `MPI_Send` | 0.007 | 0.008 | 0.010 | 0.007 | 0.9 | 21/24 |
| `smartsim_chunk_plan` | 0.003 | 0.002 | 0.004 | 0.003 | 1.0 | 24/24 |
| `MPI_Gather` | 0.000 | 0.000 | 0.001 | 0.000 | 0.0 | 24/24 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 37230.779 | 1.0 | 24/24 |
| `solver_setup` | 3486.591 | 1.0 | 24/24 |
| `solver_step_ml_warmup` | 382.942 | 1.0 | 24/24 |
| `solver_teardown` | 36.994 | 1.0 | 24/24 |
