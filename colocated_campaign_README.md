# Co-Located Interface Campaign

The campaign uses a Bash-compatible shared launcher with bounded DB cleanup. Each job
runs five fresh conditions sequentially by calling
`bash single_node_96c_4gpu.sh`; its embedded SBATCH directives are comments, not
allocation requests. Production has five jobs per platform, smoke and pilot one.
Production replicate r starts at condition r, wrapping cyclically, so every
condition occupies every position once. Production jobs may schedule independently.

| Phase | Solver Steps Per Condition | CPU Job Limit | GPU Job Limit | Jobs Per Platform | Warmup ML Calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke | 6 | 45 min | 60 min | 1 | 0 |
| pilot | 20 | 60 min | 90 min | 1 | 3 |
| production | 66 | 150 min | 120 min | 5 | 3 |

Limits cover the entire five-condition job, not each child. Order in replicate 1:
Direct no terrain preload, Direct terrain preload, CMI baseline, CMI profiling,
CMI tracing. No-preload means `FORCE_TERRAIN_UPLOAD_EACH_STEP_ENV=1`, not
`LD_PRELOAD`. Profiling and tracing share the same instrumented binary, with only
the requested recording mode enabled. All other conditions use non-Score-P builds.

## Fixed Settings

- One exclusive node, 96 MPI ranks, explicit 12x8 grid, one CPU per rank.
- One fresh shared-layout SmartSim DB co-located on that node, budget 96 CPUs.
  Solver and DB share those CPUs through the launcher's overlap/exact steps.
- CPU: c23ms, 384x216, TPQ=6, intra=16, inter=1, no GPUs.
- GPU: c23g, four GPUs, 1920x1080, TPQ=1, intra=8, inter=1.
- Giant Torch split model, chunk=12, solver batch=50000, only final save,
  fresh trajectory/state per condition, rendering and NIC sampling disabled.
- Runtime staging and installer archive creation are disabled; existing shared
  Python environments are used directly.
- `CUDA_LAUNCH_BLOCKING=1` explicitly and consistently, including DB inheritance.
  Retain for this campaign; changing it requires a separate campaign identity.
- `APPLY_CUDA_STUBS_ENV=1` lets all CPU solver conditions resolve NVML through
  the existing wrapper. GPU nodes retain the real driver library. Build-time
  `ldd` validation also supplies the toolkit stub path on driverless `devel`.
- `SKIP_PAPI_METRICS=1`, empty `SCOREP_METRIC_PAPI`. A Slurm export check precedes
   each child, and the solver wrapper explicitly checks these settings at every actual
  solver rank after launcher/module setup. All 96 rank receipts are required.
- Config uses `[library]`, matching solver runtime overrides, and preserves
  `config_keybalance.toml` MinMax normalization. Solver overrides device/model/DB
  settings. The same config is passed throughout; Direct bypasses CMI config.

## Build Prerequisites

Build all three binaries freshly from the **same source revision and worktree**
before using this campaign. `build_colocated_campaign.sbatch` builds and validates
all three variants on `devel` under account `default`. Select distinct
directory names under `solver_cpp` explicitly (no absolute paths or symlinks to
historical builds):

```bash
export CAMPAIGN_DIRECT_BUILD=build_colocated_direct_fresh
export CAMPAIGN_CMI_BUILD=build_colocated_cmi_fresh
export CAMPAIGN_SCOREP_BUILD=build_colocated_scorep_fresh
export CAMPAIGN_ID=colocated_campaign_001
bash submit_colocated_campaign.sh smoke cpu
bash submit_colocated_campaign.sh smoke gpu
```

Default is preview only. Add `--submit` intentionally to submit. Repeat for
`pilot` and `production` only after reviewing the preceding phase. Account
defaults to `rwth2150` for CPU and `rwth0792` for GPU; override with
`CAMPAIGN_ACCOUNT`. Optional `CAMPAIGN_DEPENDENCY=afterok:JOBID` gates all jobs.
Alternatively set `CAMPAIGN_BUILD_JOB=JOBID`: submission depends on that build
and allocations load its validated `logs/build_colocated_campaign_JOBID/build-dirs.env`.
Use a new
campaign ID for each attempt; do not overwrite old results.

Without a pending build dependency, scripts require existing binaries, CMake caches,
and `BUILD_READY` validation markers. They record their hashes,
source HEAD and solver diff, model and artifact-manifest hashes, config and
launcher hashes, and selected environment settings per condition. This records
identity but cannot prove that pre-existing binaries were freshly built from the
recorded source; build provenance is a prerequisite, not inferred from mtimes.
CPU uses the launcher's existing CPU Torch conversion before staging; the
original model hash is recorded, and child model-stage logs describe conversion.

## Outputs And Analysis

Job records live under
`logs/colocated_campaign/<id>/<platform>_<phase>_r<rep>_<jobid>/` with `runs.tsv`,
per-condition child logs, identities and rank environment receipts. Unique
`RUN_ID_ENV` and `JOB_NAME_ENV` isolate DB hostname/close files, local cache,
prepared inputs and trajectories. The controller resets its job-ID experiment
directory on each invocation, rather than retaining per-condition experiments.
The launcher keeps
trajectories under the job record's `trajectories/<run_id>/` and extra logs under
`logs/*<run_id>*`. Score-P tags include the condition and unique run ID under
`scorep_runs/`; tracing retains the launcher's node-local copy-back behavior.
On child failure the wrapper stops, avoiding reuse of a possibly live DB.

```bash
python3 summarize_colocated_campaign.py logs/colocated_campaign/colocated_campaign_001 cpu production
```

The analyzer accepts structured `STEP_TIMING` records or legacy `Step` records
without counting both. It requires complete five-condition jobs and complete
step logs, excludes failed/incomplete jobs, drops the first three **ML calls**
for pilot/production and none for smoke. It averages remaining ML-step times
within each condition/job before taking paired differences and paired percentages
across jobs. Calls are never pooled as independent replicates. The reported
95% Student-t CI uses paired job differences; single-job phases have no CI.
These are rank-zero ML-step elapsed times, not globally synchronized critical
path measurements (`MPI_SYNC_MODE=none` is the launcher default).

## Runtime Validation

The shared launcher passes Bash syntax checks. Startup has a 300-second deadline
and shutdown a 120-second deadline, configurable through
`SMARTSIM_STARTUP_TIMEOUT_ENV` and `SMARTSIM_SHUTDOWN_TIMEOUT_ENV`.
Failure cleanup cancels only invocation-owned DB steps, never the allocation.
Unconfirmed cleanup returns 125; the campaign aborts on any child failure.
Cluster runtime propagation and real cleanup remain untested until smoke runs.
Runtime receipts fail closed if propagation is wrong. Pilots validate complete
five-condition logs before returning success. Production can be queued with
`CAMPAIGN_DEPENDENCY=afterok:PILOT_JOBID`, and GPU pilots with
`CAMPAIGN_DEPENDENCY=afterok:SMOKE_JOBID`. This gates execution on job success,
not scientific review; inspect pilot timing before production starts and adjust
pending allocation limits if necessary.
