# s5_aix_p2p_fullcredits (hethires2)

## Steady-State Coupling Metrics (Normalized Per ML Step)

| Region | Comm Mean (ms) | Controller/Rank0 (ms) | Max Rank (ms) | Self (ms) | Visits/Steady Step | Ranks |
|:---|---:|---:|---:|---:|---:|---:|
| `solver_step_ml_steady` | 156.627 | 163.104 | 164.048 | 0.004 | 1.0 | 48/48 |
| `solver_ml_provider_call` | 156.623 | 163.001 | 164.045 | 0.003 | 1.0 | 48/48 |
| `app_provider_inference` | 148.460 | 156.120 | 156.804 | 0.001 | 1.0 | 48/48 |
| `aix_library_static_step` | 148.459 | 156.119 | 156.802 | 0.001 | 1.0 | 48/48 |
| `aix_inference` | 148.458 | 156.118 | 156.801 | 0.014 | 1.0 | 48/48 |
| `inference` | 148.444 | 156.099 | 156.784 | 0.007 | 1.0 | 48/48 |
| `p2p_pipelined_exchange` | 148.433 | 156.084 | 156.762 | 0.688 | 1.0 | 48/48 |
| `MPI_Recv` | 84.000 | 79.941 | 137.918 | 84.000 | 2.9 | 47/48 |
| `MPI_Send` | 61.152 | 76.121 | 101.863 | 61.152 | 1.0 | 47/48 |
| `MPI_Allreduce` | 22.999 | 17.493 | 35.108 | 22.999 | 36.9 | 48/48 |
| `app_prepare_input` | 7.908 | 6.698 | 9.490 | 7.908 | 1.0 | 48/48 |
| `MPI_Testsome` | 2.522 | 0.000 | 121.053 | 2.522 | 1.2 | 1/48 |
| `solver_step_compute` | 2.337 | 1.535 | 3.961 | 2.337 | 1.0 | 48/48 |
| `MPI_Bcast` | 0.982 | 0.029 | 1.756 | 0.982 | 2.0 | 48/48 |
| `app_finalize_output` | 0.252 | 0.180 | 0.503 | 0.252 | 1.0 | 48/48 |
| `MPI_Waitall` | 0.207 | 0.224 | 0.862 | 0.207 | 2.0 | 48/48 |
| `MPI_Gatherv` | 0.102 | 0.154 | 0.200 | 0.102 | 0.0 | 48/48 |
| `MPI_Waitall` | 0.059 | 0.000 | 2.818 | 0.059 | 0.1 | 1/48 |
| `MPI_Irecv` | 0.014 | 0.016 | 0.017 | 0.014 | 8.1 | 48/48 |
| `MPI_Isend` | 0.013 | 0.020 | 0.027 | 0.013 | 8.1 | 48/48 |
| `MPI_Isend` | 0.010 | 0.000 | 0.479 | 0.010 | 2.9 | 1/48 |
| `aix_collective_setup` | 0.003 | 0.005 | 0.006 | 0.003 | 1.0 | 48/48 |
| `MPI_Irecv` | 0.003 | 0.000 | 0.131 | 0.003 | 1.0 | 1/48 |
| `MPI_Gather` | 0.001 | 0.003 | 0.003 | 0.001 | 0.0 | 48/48 |

## Lifecycle Phase Totals (One-Off Cumulative)

| Phase | Duration (ms) | Total Visits | Ranks |
|:---|---:|---:|---:|
| `terrain_solver` | 21409.265 | 1.0 | 48/48 |
| `solver_setup` | 1897.976 | 1.0 | 48/48 |
| `solver_step_ml_warmup` | 421.887 | 1.0 | 48/48 |
| `solver_teardown` | 107.532 | 1.0 | 48/48 |
