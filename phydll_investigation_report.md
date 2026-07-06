# PhyDLL Multi-GPU & Score-P Investigation Report

## 1. Background & Objective
We are optimizing the physics-coupling package **PhyDLL** to support **multi-GPU data-parallel inference** (Plan A of `PLAN.md`). 
Specifically:
*   In a multi-GPU environment (e.g., one GPU node containing 4 GPUs), we want each DL rank (client process) to map to a separate GPU (GPU 0, 1, 2, 3) in a round-robin fashion.
*   To achieve this, each DL client must find its node-local rank among all DL processes running on that node.
*   We use unit tests in `module_test` to validate these changes locally on the login node (running on CPU as a fallback) before deploying to the cluster.

---

## 2. Chronology of Issues & Solutions Tried

### Issue A: MPMD Communicator Splits & Deadlocks
*   **The Error:** When running `module_test` with `PROVIDER=PHYDLL`, the test suite hung indefinitely.
*   **What we Tried:** Initially, we added manual MPI splits at the startup of `dl_client.cpp` and `phydll_dl_client.py` on `MPI_COMM_WORLD` using `color = 1`.
*   **Why it failed:** 
    *   In MPI, splits on `MPI_COMM_WORLD` are collective operations.
    *   The Python DL client wrapper calls `dll.init("dl")`, which *internally* splits `MPI_COMM_WORLD` again.
    *   This created a mismatch: the solver split once, whereas the Python client split twice. This mismatched sequence caused a deadlock.
*   **The Remedy:** 
    *   We reverted the startup splits back to `MPI_UNDEFINED` to keep the MPMD handshake clean.
    *   Instead, we query the communicator containing only DL ranks *after* PhyDLL initializes, using the library's built-in API:
        *   C++: `phydll_get_local_mpi_comm()`
        *   Python: `dll.get_local_mpi_comm()`
    *   We then perform the node-local shared memory split (`MPI_Comm_split_type`) on this isolated communicator. This gets the node-local rank safely and avoids all global deadlocks.

### Issue B: CMake Include Path Ordering Bug
*   **The Error:** Compiling the C++ client standalone in `module_test` threw:
    ```
    phydll_dl_runtime.hpp:16:10: fatal error: phydll.h: No such file or directory
    ```
*   **Why it failed:** In `dl_clients/CMakeLists.txt`, `target_include_directories` used `${PHYDLL_BUILD_DIR}/include` on line 18, but `PHYDLL_BUILD_DIR` was defined on line 24. Sourcing it standalone left the path empty (pointing to `/include`).
*   **The Remedy:** Reordered `dl_clients/CMakeLists.txt` to initialize `PHYDLL_BUILD_DIR` at the very top.

### Issue C: Static Linker Duplicate Symbols under Score-P
*   **The Error:** Linking `phydll_dl_client` under Score-P threw:
    ```
    multiple definition of `scorep_subsystems`
    multiple definition of `scorep_constructor`
    ```
*   **Why it failed:** 
    *   `phydll_dl_runtime` was compiled as a static library.
    *   Additionally, the CMake `Scorep` package macro `scorep_instrument_target` was being called on `phydll_dl_client` and `phydll_dl_runtime` while the compiler was *also* set to the Score-P wrappers (`scorep-mpicxx`).
    *   This dual configuration caused both CMake and the compiler wrappers to generate separate initialization code (`phydll_dl_client_scorep_adapter_init.c.o` and `phydll_dl_client.scorep_init.o`), clashing during linking.
*   **The Remedy:** 
    *   Changed `phydll_dl_runtime` to a **SHARED** library target (`add_library(phydll_dl_runtime SHARED...)`).
    *   Commented out the redundant `scorep_instrument_target` calls in `dl_clients/CMakeLists.txt` since the `scorep-mpicxx` compiler wrapper automatically instruments all compiled sources.
    *   **Result:** The C++ DL client compiles, links, and runs completely successfully with Score-P wrappers active!

---

## 3. The Asymmetric Instrumentation Deadlock (Current Status)

### What works
*   Compiling and running the **C++ DL client** with Score-P wrapper compilers (`scorep-mpicxx`) and linking against the shared runtime compiles and links 100% successfully.
*   The C++ solver and C++ DL client execute to completion without hangs, resolving the communicator split issue via `phydll_get_local_mpi_comm()`.
*   Compiling and running the **Python DL client** without Score-P wrappers (using standard `g++`/`gcc`) compiles, links, and runs completely successfully, fully validating our Python local communicator split via `dll.get_local_mpi_comm()`.

### What doesn't
*   Running the **Python DL client** MPMD execution while the C++ solver is compiled with Score-P (`--mpp=mpi`) hangs indefinitely at `MPI_Init` due to the asymmetric instrumentation deadlock described below. This is an inherent limitation of Score-P MPI profiling when non-C++ processes run in the same `MPI_COMM_WORLD`.

### The Cause
1.  The C++ solver is compiled with `--mpp=mpi` (meaning Score-P MPI profiling wrappers are active).
2.  At startup inside `MPI_Init`, the Score-P runtime attempts internal synchronization/collectives across all ranks in `MPI_COMM_WORLD`.
3.  The Python DL client is run via the standard Python interpreter (not instrumented by C++ Score-P wrappers).
4.  The Python rank returns from `MPI_Init` immediately and never participates in Score-P's internal initialization collectives, deadlocking the solver ranks inside `MPI_Init`.

---

## 4. Proposed Remedies

1.  **For C++ DL Client runs:** No changes needed. Both solver and client are instrumented, allowing them to cooperate.
2.  **For Python DL Client runs:** 
    *   **Option A:** Run the solver with `SCOREP_MPI_ENABLE_GROUPS=none` at runtime (to see if it disables the `MPI_Init` collective handshake).
    *   **Option B (Recommended):** Build a separate non-Score-P binary for Python runs, or default back to `--mpp=none` if we need a single binary that supports both C++ and Python DL run modes without deadlocks.

