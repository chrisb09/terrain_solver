# ML Transaction Measurement Contract

## Purpose

Measure one steady-state ML transaction consistently across the terrain solver,
the application module, and each ML library. The same user-region hierarchy is
used for low-overhead Score-P profiles and for filtered Score-P traces.

## Transaction Hierarchy

```text
ML transaction wall time
  solver input preparation
  application preprocessing
    app_prepare_input
    normalization / layout conversion / feature construction
  library static step
    library input hand-off
    library-side processing
      receive and reshape
      H2D
      inference
      D2H
      response preparation
    library output hand-off
  application postprocessing
    app_finalize_output
    output reordering / denormalization / reconstruction
  solver output copy
  solver output application
```

The application-preprocessing and application-postprocessing nodes remain in
the hierarchy even when their current terrain-solver implementation is a no-op.
This keeps results comparable when normalization, feature construction, or
postprocessing is enabled later.

## Library Mapping

| Library | Input hand-off | Library-side processing | Output hand-off |
| --- | --- | --- | --- |
| PhyDLL | `phydll_send` | DL-client receive/reshape, H2D, inference, D2H | DL-client send, `phydll_recv` |
| AIX | `gatherInputData` | `inferenceDevice` and host/device work | `scatterOutputData` |
| SmartSim | `smartsim_put_tensor` | `smartsim_run_model` | `smartsim_unpack_tensor` |

`phydll_library_static_step`, `aix_library_static_step`, and
`smartsim_library_static_step` delimit the complete library invocation. They
are intentionally provider-specific names so profiles from multiple libraries
remain unambiguous.

## Profile View

Profiles answer: *what is the typical work performed by a solver rank?*

For every steady ML step, record mutually exclusive solver-side leaves on each
solver rank. Aggregate a leaf as the arithmetic mean over solver ranks and
steady steps. Compute the root using the same rank-step samples.

All solver ranks execute the same configured step count. Do not include PhyDLL
DL-client ranks or AIX service ranks in this solver-rank average.

Do not construct a transaction wall-time sunburst by summing independent CUBE
`max_time` values. Those values may be inclusive, overlap in time, and select a
different slowest rank per region.

## Trace View

Traces answer: *which dependent work determines the transaction makespan?*

Trace the manual regions above and MPI events, while filtering compiler-level
function instrumentation. For PhyDLL and AIX, include both solver and
library-side ranks. MPI send/receive and collective events connect the ranks
into one request dependency graph.

The critical path is the longest causal chain through this graph. A solver wait
and library-side work that occur concurrently must not be added as wall time.
For SmartSim, the synchronous `put_tensor`, `run_model`, and `unpack_tensor`
spans are sufficient solver-side transaction measurements; server-side work is
already contained in their blocking elapsed times.

## Phase Makespan

To report the post-warmup phase makespan, synchronize once immediately after
warmup, time the remaining steps locally, then reduce the local durations with
MPI `MIN`, `SUM`, and `MAX`. Report all three values. The maximum is the
slowest-rank phase duration; the mean is average solver-rank work.

## Instrumentation Overhead

Each provider comparison should have paired runs with identical input, model,
rank placement, warm-up, and steady-step counts:

1. Native build with `USE_SCOREP=0`.
2. Score-P-instrumented build with profiling and tracing disabled, measuring
   compiler/user/MPI wrapper overhead without writing a profile.
3. Score-P profile with MPI and user regions enabled and PAPI metrics disabled.

Report per-transaction solver-rank `min`, arithmetic mean, and `max`, plus
whole-job wall time. Use a separate short tracing run for causal analysis; do
not use tracing as the performance baseline. For PhyDLL, every MPI process in
the MPMD job must use the same wrapped/native mode. In particular, do not mix
Score-P-wrapped solver ranks with an unwrapped Python DL rank.

## Provider Roles

AIx `inferenceDevice` is controller-only work. Report the controller rank's
device time separately from worker collective residence in `MPI_Scatterv`.
Worker `scatterOutputData` time is a blocking-collective interval and includes
arrival wait; it is not an output-bandwidth measurement. PhyDLL `phydll_send`
and `phydll_recv` are also blocking protocol envelopes. Use nested MPI events,
DL-client copy regions, and traces to separate transfer, waiting, and
inference.

The current diagnostic build adds Score-P counters for direct SmartSim tensor
bytes and PUT/RUN/UNPACK requests, AIx logical input/output bytes and device
batch count, and PhyDLL fixed-field representation bytes. PhyDLL DL-client
regions also separate large input/output vector allocation and the copy from
the receive aggregate into the frame. Set `PHYDLL_IO_LOG_BARRIERS=0` for a
controlled PhyDLL run without the barriers used by its logging helper; keep
the default for the baseline so the effect can be measured explicitly.
