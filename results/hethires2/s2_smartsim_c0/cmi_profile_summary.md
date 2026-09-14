# s2_smartsim_c0 (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 662.589 | 666.947 | 666.947 | 0.011 | 1.0 | 4/4 |
| `solver_ml_provider_call` | 662.578 | 666.910 | 666.910 | 0.004 | 1.0 | 4/4 |
| `app_provider_inference` | 647.412 | 651.617 | 651.617 | 0.002 | 1.0 | 4/4 |
| `smartsim_library_static_step` | 647.410 | 651.616 | 651.616 | 0.170 | 1.0 | 4/4 |
| `smartsim_put_tensor` | 586.240 | 589.488 | 591.590 | 586.240 | 2.0 | 4/4 |
| `MPI_Allreduce` | 78.300 | 73.819 | 85.640 | 78.300 | 36.9 | 4/4 |
| `smartsim_run_model` | 50.848 | 50.515 | 52.471 | 50.848 | 1.0 | 4/4 |
| `solver_step_compute` | 21.776 | 21.939 | 21.982 | 21.776 | 1.0 | 4/4 |
| `app_prepare_input` | 12.294 | 12.368 | 12.368 | 12.294 | 1.0 | 4/4 |
| `smartsim_unpack_tensor` | 10.151 | 11.332 | 11.332 | 10.151 | 1.0 | 4/4 |
| `app_finalize_output` | 2.868 | 2.920 | 2.955 | 2.868 | 1.0 | 4/4 |
| `MPI_Bcast` | 0.478 | 0.006 | 0.646 | 0.478 | 2.0 | 4/4 |
| `MPI_Waitall` | 0.108 | 0.116 | 0.123 | 0.108 | 2.0 | 4/4 |
| `MPI_Gatherv` | 0.066 | 0.046 | 0.085 | 0.066 | 0.0 | 4/4 |
| `MPI_Irecv` | 0.016 | 0.017 | 0.017 | 0.016 | 8.1 | 4/4 |
| `MPI_Isend` | 0.009 | 0.008 | 0.010 | 0.009 | 8.1 | 4/4 |
| `smartsim_chunk_plan` | 0.001 | 0.001 | 0.001 | 0.001 | 1.0 | 4/4 |
| `MPI_Gather` | 0.001 | 0.002 | 0.002 | 0.001 | 0.0 | 4/4 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 81254.800 | 1.0 | 4/4 |
| `solver_setup` | 3761.790 | 1.0 | 4/4 |
| `solver_step_ml_warmup` | 848.357 | 1.0 | 4/4 |
| `solver_teardown` | 36.697 | 1.0 | 4/4 |
