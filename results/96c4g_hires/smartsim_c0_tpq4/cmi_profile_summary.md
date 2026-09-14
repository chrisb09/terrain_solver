# smartsim_c0_tpq4 (96c4g_hires)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 244.056 | 243.288 | 251.506 | 0.002 | 1.0 | 96/96 |
| `solver_ml_provider_call` | 244.054 | 243.268 | 251.504 | 0.004 | 1.0 | 96/96 |
| `app_provider_inference` | 242.407 | 241.901 | 249.855 | 0.003 | 1.0 | 96/96 |
| `smartsim_library_static_step` | 242.404 | 241.898 | 249.853 | 0.103 | 1.0 | 96/96 |
| `smartsim_put_tensor` | 202.584 | 201.936 | 212.577 | 202.584 | 2.0 | 96/96 |
| `MPI_Allreduce` | 61.074 | 63.290 | 66.331 | 61.074 | 36.9 | 96/96 |
| `smartsim_run_model` | 34.931 | 35.027 | 40.021 | 34.931 | 1.0 | 96/96 |
| `smartsim_unpack_tensor` | 4.784 | 4.778 | 7.917 | 4.784 | 1.0 | 96/96 |
| `app_prepare_input` | 1.521 | 1.255 | 1.890 | 1.521 | 1.0 | 96/96 |
| `solver_step_compute` | 1.213 | 0.782 | 1.824 | 1.213 | 1.0 | 96/96 |
| `MPI_Bcast` | 0.964 | 0.020 | 1.140 | 0.964 | 2.0 | 96/96 |
| `MPI_Waitall` | 0.400 | 0.588 | 1.359 | 0.400 | 2.0 | 96/96 |
| `app_finalize_output` | 0.122 | 0.106 | 0.203 | 0.122 | 1.0 | 96/96 |
| `MPI_Gatherv` | 0.092 | 0.078 | 0.131 | 0.092 | 0.0 | 96/96 |
| `MPI_Isend` | 0.029 | 0.026 | 0.032 | 0.029 | 8.1 | 96/96 |
| `MPI_Irecv` | 0.019 | 0.018 | 0.020 | 0.019 | 8.1 | 96/96 |
| `smartsim_chunk_plan` | 0.002 | 0.002 | 0.005 | 0.002 | 1.0 | 96/96 |
| `MPI_Gather` | 0.000 | 0.001 | 0.001 | 0.000 | 0.0 | 96/96 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 35720.877 | 1.0 | 96/96 |
| `solver_setup` | 2107.440 | 1.0 | 96/96 |
| `solver_step_ml_warmup` | 549.343 | 1.0 | 96/96 |
| `solver_teardown` | 39.652 | 1.0 | 96/96 |
