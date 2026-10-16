# Grouped GPU Refresh

This is a new campaign, not a change to legacy pilot defaults. No job has been
submitted by implementing or testing this harness. The main operator owns launch.

## Matrix

- 24 timing allocations: two models, four hardware layouts, three repetitions.
  Each allocation runs PHYDLL readiness off/on in order for repeats 1 and 3,
  on/off for repeat 2. Each case is a fresh process.
- Four profile allocations: one per hardware layout. Each includes the four
  PhyDLL cases, six missing AIx collective/default cases, two missing SmartSim
  shared GiantMLP c0 cases, and 16 missing SmartSim shared c1/c3 cases (both
  models, every hardware layout). No already-run matching cases are repeated.
- Total: 28 independent allocations, 88 logical cases. No dependencies.
- WaterCNN uses 7680x4320, chunk 60, cap 20000000; timing is 202 simulation
  steps, 101 ML calls (one warm, 100 steady). GiantMLP uses 480x288, chunk 16,
  cap 500000; timing is 12 steps, six calls (one warm, five steady).
- Every profile case uses 12 steps, one warm and five steady calls, shared MPI
  profiles, tracing off, sampling off. Packed transport uses production numeric
  double wire behavior; no codec configuration is introduced. Solver/model
  float storage is unchanged.
- SmartSim chain cases are shared DB, TPQ1, intra-op 8, inter-op 1, with
  `SMARTSIM_MPI_SEQUENTIAL_PUT` set to 0, 1, or 3 as the case specifies. c1/c3
  stagger uploads; they do not change TPQ. PHYDLL uses topology pinning, budget
  24 or 96, one thread, one or four DL workers in addition to solver ranks.
  Two-node layouts retain empty grid overrides for autogrid.
- Runtime tar creation is disabled: this campaign uses the shared runtime
  directly, so packaging it while another case installs profile dependencies
  is unnecessary and unsafe.
- Both heterogeneous components are exclusive `rwth0792` / `c23g`, `mem=0`.
  Both request GPU GRES, as required by c23g and in the previous surrogate
  allocations. CPU-side solver ranks have their GPUs hidden at runtime.

## Build Contract

The profile retry uses `profile_retry_groups()`: 6/11/10/4 cases on
24c1g/96c4g/24p24c1g/96p96c4g, with 35/45/55/25 minute limits respectively.
It excludes the four successful 24c1g SmartSim backend-smoke cases and the
four PhyDLL plus one AIx successful 96p96c4g profiles. All 31 retry cases
use 12 steps and profiling; the original campaign matrix is unchanged.

## Thesis Groups

`thesis_native_groups()` fills the 22 native SmartSim cells missing from the
earlier retest campaigns: shared c1/c3 on every layout, per-GPU DB on 24c1g and
24p24c1g, and shared c0 on 24p24c1g, for both models. It uses the matrix's three
repeats, split per hardware/model/repeat (24 allocations, 66 cases) with the
202-step WaterCNN and 12-step GiantMLP conventions.

`thesis_trace_groups()` validates one short capture per library before the full
expansion: on 24c1g, PhyDLL default, AIx p2p1 (with the P2P timeline CSV),
SmartSim shared and per-GPU; on 24p24c1g, PhyDLL default and AIx p2p1. Trace
mode runs 12 steps, enables both profiling and OTF2 tracing, and validates a
nonempty `traces.otf2` plus, for pipelined AIx, a nonempty `aix_p2p_timeline_*`
CSV. `export_trace_intervals.py` converts a trace into the renderer CSV
(`OTF2_PRINT` selects the executable); `build_thesis_aix_variants.sbatch` builds
the isolated affine/NUMA DSOs needed for the remaining AIx cells.

The separately owned `build_gpu_refresh.sbatch` must finish before either
dry-run or submit. Set `GPU_REFRESH_BUILD_JOB_ID_ENV` to that completed job ID.
The harness requires matching solver/DL markers for each mode, and matching
identity across both modes. It never queues a build or adds a build dependency.

Paths relative to the workspace:

- `mini_app/solver_cpp/build_gpu_refresh_{plain,scorep}/terrain_solver`
- `CPP-ML-Interface/dl_clients/build-gpu-refresh-{plain,scorep}/phydll_dl_client`
- `CPP-ML-Interface/extern/phydll/build-gpu-refresh/lib`
- Each solver and DL directory has `BUILD_READY`; the solver has `build.env`.
- Provenance is `solver_cpp/build_gpu_refresh_MODE/provenance-BUILD_JOB_ID`.

The marker is the build agent's exact format, followed by SHA-256 records for
the provenance checksum manifests:

```text
BUILD_READY job=JOB_ID mode=plain completed=UTC_TIMESTAMP validation=cpu-build-only runtime_validation=pending
SHA256  ABSOLUTE_PROVENANCE_PATH/binaries.sha256
SHA256  ABSOLUTE_PROVENANCE_PATH/source.sha256
```

`build.env` defines `SOLVER_BIN`, `PHYDLL_DL_CLIENT`, `PHYDLL_BUILD_DIR`,
`USE_SCOREP`, `SCOREP_MPP=mpi`, `PROVENANCE`, `GPU_REFRESH_LIBRARY_PATH` and
exports `LD_LIBRARY_PATH`. The adapter sources this trusted build artifact and
restores its library prefix after driver environment setup. It maps the legacy
PhyDLL `realpath` lookup to the fresh library and blocks driver compilation
before any build-directory deletion. Cache, registry, source and binary hashes,
and Score-P linkage evidence are checked before submission and between cases.
Timing requires no Score-P linkage; profile requires native MPI Score-P linkage.
CPU-build-only readiness does not claim successful GPU runtime validation.

The operator's `USER`, `LOGNAME`, `HOME` and runtime identity are inherited with
Slurm `--export=ALL`. `GPU_REFRESH_RUNTIME_ROOT_ENV` optionally overrides the
SmartSim Python runtime root; the default is `<workspace>/python`. The submitter
exports `GPU_REFRESH_MINI_ENV`, campaign token and build identity explicitly.
`GPU_REFRESH_SWATCH` optionally selects a watcher executable, defaulting to
`$HOME/scripts/swatch`; exactly one consolidated registration occurs only after
all 28 successful submissions. Dry-run invokes neither sbatch nor watcher.

## Commands

Read-only inspection, requiring no build or allocation:

```bash
python3 mini_app/gpu_refresh_campaign.py --list
python3 mini_app/gpu_refresh_campaign.py --describe timing_24p24c1g_watercnn8k_r2
python3 -m unittest mini_app/test_gpu_refresh_campaign.py -v
```

After the build is READY, preview using a new campaign token and the actual
completed build job ID:

```bash
GPU_REFRESH_BUILD_JOB_ID_ENV=BUILD_JOB_ID bash mini_app/submit_gpu_refresh.sh --dry-run gpu_refresh_preview_unique
```

Main operator only, not executed during implementation:

```bash
GPU_REFRESH_BUILD_JOB_ID_ENV=BUILD_JOB_ID bash mini_app/submit_gpu_refresh.sh --submit gpu_refresh_production_unique
```

Dry-run writes `results/gpu_refresh/TOKEN/{ready.json,groups.csv,cases.csv}` and
prints exactly 28 sbatch commands. Use a different fresh token for submission.
Templates are `gpu_refresh_{24c1g,96c4g,24p24c1g,96p96c4g}.sbatch`.
Submission overrides time in both heterogeneous components: WaterCNN timing
60/40 minutes on single nodes, 75 on two nodes; GiantMLP timing 20 minutes;
profiles 55 single-node or 65 two-node minutes.

## Results And Safety

Allocation output is `results/gpu_refresh/TOKEN/GROUP_JOB_ID/RUN_ID/` with a
unique run ID containing campaign, group, hardware, repetition, mode, Slurm job,
case order, model, provider and variant. No existing campaign or case directory
is reused. The adapter redirects HDF5 and timing outputs here; driver logs have
unique run IDs under `logs/`. Per-case artifacts include config, environment,
driver hash, log, profile paths and manifest. Each group also has a manifest.

The runner terminates its own driver process group and drains numeric Slurm
steps in its exclusive allocation, excluding batch/extern steps. It removes
only its run-specific DB signal files. A failed case is recorded and subsequent
cases run only after verified safe cleanup and renewed build checks. Unsafe
cleanup stops the group; any case failure makes the group fail. No user-wide
`pkill`, allocation cancellation, or cross-job cleanup is used.

Tests use temporary builds, fake sbatch/watcher executables and mocked runtime
launches. They check the entire matrix, readiness order, missing-case selection,
caps, grids, templates, time overrides, fresh build guards, shell adapter,
safe-failure continuation and manifests. They never launch real jobs.

## MissingProfileBatch

`missing65_profile_groups()` selects eight allocations (one per hardware/model),
65 missing cases: 62 AIx across default/affine/numa layouts and collective,
p2p1/p2p24 communication, plus three SmartSim per-GPU cases (24c1g GiantMLP;
24p24c1g both models). Completed matching profiles are excluded. Each case is
profile-only: 12 steps, six ML calls (one warm, five steady), no OTF2 tracing.

AIx uses `mini_app/solver_cpp/build_thesis_aix_variants/{layout}_scorep/aix-install/lib`,
not a shared default library. Before launch, require verified layout-specific
`READY`, `checksums.sha256`, and `library.ldd.txt`; also check the actual solver
`ldd` resolution under the case's library environment points to that layout's
`libAIxeleratorService.so`. CPU-build readiness is not GPU runtime validation.
Variant build job `5026388` uses `rwth2150` / `c23ms` (replacing cancelled
`5025857`). CPU launcher `5026232` uses `thes2181` / `c23ms` and waits for its
successful completion, then runs `submit_missing65_profiles.sbatch` to validate
readiness and submit all eight allocations with `smoke_first=True`. The first
allocation (24c1g WaterCNN, covering all three AIx layouts) gates the remaining
seven with `afterok`; invalid dependencies cancel rather than wait indefinitely.
The GPU profile jobs do not exist until that launcher runs successfully.
Build/launch snotify watcher: `5026394`, on `thes2181` / `c23ms` (replacing
cancelled `5026254`); the launcher registers a separate
consolidated snotify watcher for all eight GPU allocations once submitted.
GPU allocations remain `rwth0792` / `c23g`, including both surrogate components.

Build `5026388` failed CPU loader validation after compiling the default Score-P
DSO: `libnvidia-ml.so.1` is unavailable on CPU nodes. Its dependent launcher
`5026232` was cancelled without submitting GPUs. Retry build `5027130`
(`--resume`) preserves completed variants and uses an isolated, checksummed
NVML stub only for CPU `ldd -r`; the directory never enters `runtime_ld`.
The GPU adapter rejects that directory and missing dependencies. Replacement
launcher: `5027132`; replacement build/launch snotify: `5027133`.
Retry `5027130` completed successfully in 3:46 with all six variants ready;
launcher `5027132` also completed. Campaign `missing65_20261015_5027132`
contains 65 cases in GPU jobs `5027177`, `5027178`, `5027179`, `5027180`,
`5027181`, `5027183`, `5027185`, `5027187`. First job `5027177` gates the
remaining seven; consolidated GPU snotify watcher is `5027189`.

First GPU job `5027177` timed out after repeated startup OOMs, with no completed
ML calls. Accounting shows 499800M allocated to the job but only 2G to each
solver step: the CPU launcher's `SLURM_MEM_PER_NODE=2048` leaked through exports.
Submission now strips parent `SLURM_*`, `SBATCH_*`, and `SRUN_*` settings before
calling sbatch; each runtime case explicitly uses `SLURM_MEM_PER_NODE=0` and
removes per-CPU/per-GPU memory settings. Missing65 groups fail fast after a
failed case. Retry campaign `missing65_20261016_memory_retry1` has first GPU
job `5050769`, seven dependent jobs `5050770`, `5050771`, `5050772`, `5050773`,
`5050775`, `5050777`, `5050779`, and consolidated snotify watcher `5050781`.
The original dependent jobs were cancelled without running. Limits are unchanged.

Per-group limits in minutes (WaterCNN / GiantMLP):

| Hardware | Minutes |
| --- | --- |
| 24c1g | 13 / 19 |
| 96c4g | 17 / 17 |
| 24p24c1g | 18 / 19 |
| 96p96c4g | 17 / 17 |

Budgets use six calls times conservative native mean bounds: AIx WaterCNN/GiantMLP
735/2468, 231/780, 566/2561, 201/776 ms in table order; SmartSim 4000 ms
WaterCNN and 2469/2712 ms GiantMLP single/two-node. Add startup per case (AIx
30 s, or 45 s at 96 ranks; SmartSim 150/60 s WaterCNN/GiantMLP) and 30 s
cleanup/profile flush per case, multiply by 1.5 for headroom, add 60 s allocation
overhead, then round up to minutes. `budget.json` records the estimates.

For a manual retry only (do not duplicate the queued launcher), after variant
readiness and linkage checks pass, submit with a
fresh, unused token through the API (reused solver build identity `4938224`,
not the pending variant build ID); use `notifier="snotify"`:

```bash
GPU_REFRESH_BUILD_JOB_ID_ENV=4938224 python3 -c 'from mini_app.gpu_refresh_campaign import submit, missing65_profile_groups; submit("--submit", "missing65_profiles_FRESH_UNIQUE_TOKEN", missing65_profile_groups(), notifier="snotify", smoke_first=True)'
```
