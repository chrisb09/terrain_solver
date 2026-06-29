# Progress Summary

*Related Documents:*
- [Score-P Instrumentation Plan](file:///hpcwork/ro092286/smartsim/mini_app/scorep_instrumentation_plan.md)
- [Instrumentation Progress](file:///home/ro092286/.gemini/antigravity-cli/brain/90ac1304-b1f7-4ab9-b97a-d4916b87fe28/instrumentation_progress.md)

## Current Plan Status
We are currently at **Step 4 — PhyDLL DL side (C++ + Python)**. We have resolved the python context manager/nesting conflict by launching Python with `--noinstrumenter`, and resolved PATH isolation issues inside `mpirun` using `env`. The validation job `1473539` has been submitted and is currently pending/running on the `devel` partition (using the `default` account).

* **Previous Step (Step 3):** Completed successfully. Validated `profile.cubex` in `scorep_PHYDLL_1470871_rank_0/`.
  * `phydll_recv`: 2.10s (45.6%) — waiting for Python DL inference execution
  * `solver_setup`: 1.22s (26.6%)
  * `phydll_send`: 0.30s (6.5%)
  * `solver_main_loop`: 0.16s (3.4%)
  * `phydll_prepack`: 0.05s (1.1%)
  * `solver_step_ml`: 0.04s (0.9%)
  * `solver_teardown`: 0.04s (0.9%)

## 1. Resolved CMake Double Instrumentation
We fixed the `SCOREP_MPP_SYSTEM` mismatch error during the CMake configure step. The issue was that `scorep_instrument_target()` was called on both `terrain_solver` (and test executables) AND its dependency `cpp_ml_interface_library`. Because the library propagates the property `INTERFACE_SCOREP_MPP_SYSTEM` due to `UNIQUE` constraints in CMake, calling the macro on the executables again caused a conflict. 

We resolved this by removing `scorep_instrument_target()` for the executables and instead directly setting the target property:
```cmake
set_target_properties(<target> PROPERTIES SCOREP_MPP_SYSTEM "mpi")
```
This satisfied the CMake `COMPATIBLE_INTERFACE_STRING` requirement without double-instrumenting the targets, allowing compilation to succeed.

## 2. Fixed Score-P PAPI Initialization Crash
After successful compilation, running the smoke test yielded an immediate crash from Score-P: `Error: Could not initialize PAPI library: PAPI_add_event(0:"PAPI_TOT_INS") (fatal)`. The node on the `devel` partition does not natively support the `PAPI_TOT_INS` event due to missing counter access (likely permissions / `perf_event_paranoid`).

To fix this, we modified `proper_slurm_job.sh` and `submit_smoke_test.sh` to explicitly set `export SCOREP_METRIC_PAPI=""` instead of allowing it to default to `PAPI_TOT_INS`. 

## 3. Smoke Test Allocation
As per your instructions, the smoke test was submitted to the `devel` partition using the `default` account (`--partition=devel --account=default`), while all output and artifacts are preserved in your `thes2181` directory structure. 

## 4. Resolved: HDF5 Runtime Version Mismatch
We identified and resolved the HDF5 runtime version mismatch error (`Headers are 1.12.2, library is 1.12.1`). The mismatch occurred because `proper_slurm_job.sh` prepended `/usr/lib64` globally to `LD_LIBRARY_PATH` to resolve `libnvidia-ml.so` on GPU nodes, which shadowed the correct HDF5 module library with the older system library at `/usr/lib64/libhdf5.so.200`.

To fix this, we modified `proper_slurm_job.sh` to:
1. Detect whether we are on a GPU node (`USE_GPU == 1`).
2. Only append the CUDA stub directory `/cvmfs/.../CUDA/12.4.0/lib/stubs` to `LD_LIBRARY_PATH` on CPU nodes (where native NVML is absent) to satisfy linking without overriding module paths.
3. Remove global prepending of `/usr/lib64`.

This successfully resolved the HDF5 mismatch, allowing the solver to run to completion and produce a valid `profile.cubex` (e.g. `scorep_AIX_1454981/profile.cubex`).

