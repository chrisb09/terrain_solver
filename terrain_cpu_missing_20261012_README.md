# Terrain Missing CPU Campaign, 2026-10-12

Scope: terrain files in `mini_app` only. No commits, shared builds, libraries,
CMI sources, GPU campaign files, or legacy measurements are changed.
Slurm accounting confirmed all tasks of patched PhyDLL arrays `4940251` and
`4940252` completed with exit `0:0`; these arrays are not resubmitted.
The user's queue contained no duplicate terrain missing campaign before submission.

## Treatments

Production memory: AIx (`AIx` in the runner), `SS_DEFAULT`, and
`SS_TPQ96_I1_B96`, both watercnn and giant, five reps each: 30 runs.
Six steps, two validation-only steady ML calls, external /proc PSS/RSS sampler
at one second. These retain the original Score-P-instrumented memory treatment:
solver `solver_cpp/build_scorep_cmi_fresh/terrain_solver`, and the original DL
binary `CPP-ML-Interface/dl_clients/build-scorep/phydll_dl_client` for preflight.
Loaded C library approval is
`dd551169e436d07e367fc5304389eafba3999f43587a20af0ec0fc8b8c87f553`.
No PhyDLL memory run is included in this new campaign. Memory timings are not
plain timing evidence. AIx census validation requires 96 solver, zero DL, zero
DB processes; SmartSim requires 96 solver, zero DL, one DB.

Production timing: AIx, patched PhyDLL readiness false/true, and all four
SmartSim configurations, both models, five reps each: 70 runs.
22 steps, ML step 2 warmup, ten steady synchronized ML calls at steps 4..22.
Uses `MODE=timing`, no Score-P environment setup, no profiler, no sampler,
no internal memory logging, telemetry, or PhyDLL I/O logging/barriers.
The application-native synchronized STEP_TIMING timers remain enabled.
Binding, rank grid, models, input files, batch policies and thread settings
are preserved from the existing terrain runner. PhyDLL uses the existing
`config_terrain_phydll_{false,true}.toml` files, not the old default config.

## Actual Plain Build

No new build job is necessary. Reused, read-only, already published plain build:
`solver_cpp/build_gpu_refresh_plain`, build job `4938224`, with provenance
`solver_cpp/build_gpu_refresh_plain/provenance-4938224`.
DL client: `CPP-ML-Interface/dl_clients/build-gpu-refresh-plain/phydll_dl_client`.
CMI, DL runtime, C PhyDLL core, and plain AIx dependency hashes are attested by
the published manifest. Plain AIx is the existing `AIxeleratorService/INSTALL`
artifact, not the instrumented `INSTALL-SCOREP` artifact.

The runtime gate checks the published marker and manifest hashes, source and
artifact hashes, `WITH_SCOREP=OFF`, actual OpenMPI compiler paths, no Score-P
or compiler-instrumentation flags, all three compiled providers, patched
readiness registry, Torch DL support, resolved dependencies/relocations, matched
C library, exact plain AIx path, and no Score-P or compiler-hook symbols in
the six application/provider artifacts. Both executable dependency closures
must contain no `libscorep`. This is an actual uninstrumented build, not an
instrumented binary with profiling disabled. Current source hashes passed,
including patched field-view/buffer-reuse PhyDLL sources. Gate checks repeat
in every timing task and binary hashes repeat after every configuration.
Plain C library SHA-256:
`3abfc405906bac756ff9f20cd9627f6a4441f46d397a09e05de566696eb76b7c`.

## Submission And Gates

`submit_terrain_cpu_missing.sh` is dry-run by default. It preflights both builds,
checks the queue and fresh campaign namespaces, clears inherited SBATCH options,
and uses explicit exports. `--submit` records IDs immediately in
`terrain_cpu_missing_terrain_missing_20261012_jobs.csv`.

Three separate six-step smoke arrays precede their respective production arrays.
Smoke memory: two tasks, six runs. Smoke non-PhyDLL timing: two tasks, ten runs.
Smoke PhyDLL timing: four tasks, four runs, both models/readiness variants.
Production arrays require `afterok` of their complete respective smoke array.
Every run validates native steady timing coverage; memory also validates sampler
coverage/read quality/cleanup and configuration-specific direct census.
Timing requires no Score-P directory or memory artifacts, and records blank cubex,
`scorep=0`, `memory_sampler=0`, plain build and readiness metadata.
Legacy performance/memory still require their Score-P archive.

Each allocation: exclusive CPU c23mm node, account rwth2150, 96 CPUs, 480G,
six-hour limit; each array has concurrency limit two. The three independent
arrays can occupy up to six nodes combined. Maximum requested budget is 48
task allocations x 6 hours = 288 node-hours, including smoke, not an estimate
of actual runtime. Run timeout 7200 seconds, DB startup timeout 180 seconds.

Other-group tasks 0..4: watercnn reps 1..5; 5..9: giant reps 1..5.
PhyDLL tasks 0..9: readiness false; 10..19: readiness true, with the same
model/rep mapping modulo ten. PhyDLL output task index is remapped modulo ten;
readiness remains explicit in its dataset namespace and result metadata.

Production outputs under `results/terrain_cpu_final/`:

- `terrain_missing_20261012_other/memory/`: 30 runs.
- `terrain_missing_20261012_other/timing/`: 50 runs.
- `terrain_missing_20261012_readiness_false/timing/`: 10 runs.
- `terrain_missing_20261012_readiness_true/timing/`: 10 runs.

Smoke namespaces use `terrain_missing_20261012_smoke` instead of the production
prefix. Roots are fresh and existing roots are rejected, including requeues.
Do not pool smoke, memory, Score-P performance or readiness variants.
The old final analysis now accepts supported memory subsets, including AIx;
it still does not support plain `timing` mode. No analysis/plot job is submitted.

## Validation

- Shell syntax checks for runner, plain gate, campaign wrapper, submission helper.
- Original suites: 40 tests passed before the added census gate.
- Final combined suites: 45 tests passed, including five new mapping/timing/census
  and submission-manifest contracts, existing runner, live /proc sampler isolation,
  diagnostics, PhyDLL retest and final analysis.
- Read-only plain build gate and original Score-P memory linkage/input preflight.
- Full six-command submission dry-run, no jobs submitted by dry-run.
- All six scheduler `sbatch --test-only` resource/array checks passed. Test-only
  reservation previews `4952343`..`4952348` are not submitted job IDs.
- `git diff --check` passed for edited tracked terrain files.

Allocation runtime smoke status is recorded below; offline tests alone are not
runtime evidence. Queue delay, runtime DB setup, exact short-run sampler coverage and
concurrent changes to attested build sources/dependencies may block execution;
gates fail closed and production dependencies then remain blocked.

## Submitted Jobs

| Job ID | Stage | Treatment | Array | Runs | Dependency |
|---|---|---|---|---:|---|
| 4952358 | Smoke | Memory, AIx/default/TPQ96 | 0,5%2 | 6 | None |
| 4952359 | Smoke | Plain timing, five non-PhyDLL configs | 0,5%2 | 10 | None |
| 4952360 | Smoke | Plain timing, patched PhyDLL false/true | 0,5,10,15%2 | 4 | None |
| 4952361 | Production | Memory, AIx/default/TPQ96 | 0-9%2 | 30 | afterok:4952358 |
| 4952362 | Production | Plain timing, five non-PhyDLL configs | 0-9%2 | 50 | afterok:4952359 |
| 4952363 | Production | Plain timing, patched PhyDLL false/true | 0-19%2 | 20 | afterok:4952360 |
| 4952582 | Retry Smoke | Plain timing, five non-PhyDLL configs, isolated DB libraries | 0,5%2 | 10 | None |
| 4952588 | Replacement Production | Plain timing, five non-PhyDLL configs | 0-9%2 | 50 | afterok:4952582 |
| 4952823 | Clean Replacement | Memory, watercnn rep 1, AIx/default/TPQ96 | 0 | 3 | None; memory smoke already passed |

All submissions and dependencies were verified with Slurm. No build job submitted; reuse of
published plain build job `4938224` is fully preflighted, so no build dependency
is needed. Slurm logs: `logs/terrain_cpu_missing_ARRAYID_TASK.log`.

### Runtime Follow-Up

Memory smoke `4952358` completed both tasks with exit `0:0`: all six runs
(AIx/default/TPQ96, both models) passed logging, sampler and census validation.
Memory production `4952361` was released and started.

First non-PhyDLL timing smoke `4952359` passed AIx on both models, but failed
SS_DEFAULT at backend initialization. RedisAI logged undefined Torch JIT symbol
`parseSchemaOrName`: the solver's plain libtorch directory in LD_LIBRARY_PATH
overrode RedisAI's own backend libraries. Dependent production `4952362` was
automatically CANCELLED by Slurm after the smoke failed; no production task ran.
Failed evidence is preserved without alteration.

The terrain runner now captures its canonical pre-gate library path and uses it
only for timing-mode controller/DB launches. Solver/DL retain their attested plain
library path. Controller environment is captured in `controller.env` with
`controller_environment.sha256`. Legacy modes and shared controller are unchanged.
The runner/mapping tests (13) and shell syntax passed after this correction.
Retry smoke scheduler preflight passed (`4952574` is a test-only preview, not a job).

Only the failed non-PhyDLL timing smoke/production were replaced: `4952582` uses
fresh `terrain_missing_20261012_smoke2_other/timing` output; `4952588` uses the
original untouched production namespace and depends on `4952582`. No memory or
PhyDLL array was duplicated. Pending retry is not claimed successful.
Two extra smoke allocations raise the maximum requested budget to 300 node-hours;
the cancelled production array consumes no runtime allocation.

Patched PhyDLL plain timing smoke `4952360` completed all four tasks with exit
`0:0` (both models, both readiness variants). Production `4952363` was released.
The final combined 45-test suite passed again after the DB library-path correction.

Memory production task `4952361_0` has Slurm exit `2:0` despite three complete
SUCCESS measurement rows: editing the shared terrain runner to fix timing while
that task was running shifted Bash's script read offset and caused a trailing
quote/EOF parse error after the final measurement. This is a runner bookkeeping
failure, not a claim that the Slurm task succeeded. All evidence is preserved.
The strict existing raw-memory validator was run on its AIx, SS_DEFAULT and TPQ96
CSV/JSON artifacts: all passed; AIx reported one sampling-overrun quality warning,
which is not an invalid measurement. No duplicate memory task was submitted.
Subsequent tasks loaded the corrected runner and continued normally. The wrapper
now reads the complete runner into `bash -c` for future submissions to avoid this
live-file-offset hazard; already submitted Slurm wrapper snapshots are unchanged.
No more runner edits are planned while the submitted arrays execute.

## Clean Memory Task-0 Replacement

Follow-up authorization requires a clean rerun because Slurm task `4952361_0`
failed. Its three original measurement rows are now **superseded and excluded**,
even though their raw artifacts passed validation. Original files are preserved;
`job_4952546_0/watercnn/rep1/SUPERSEDED.json` explicitly records the exclusion.
`terrain_cpu_final_analysis.collect` rejects a selection containing that marker,
so old SUCCESS rows cannot be silently counted as production measurements.

The isolated replacement submits only array task `0`: watercnn rep 1, six steps,
AIx, SS_DEFAULT, and SS_TPQ96, in new dataset
`terrain_missing_20261012_memory_task0_clean1_other/memory/`.
It does not resubmit tasks 1..9, giant, timing, or either smoke campaign. No shell
runner, wrapper, gate, or submission helper is edited during this follow-up.
All analysis/test changes are completed before submitting the rerun.

For final memory analysis, explicitly select original allocation job roots for
tasks 1..9 plus the new replacement memory root. Do not pass the entire original
memory root: it contains the superseded task and will fail closed. The accepted
replacement must have Slurm COMPLETED/0:0, three SUCCESS rows and passing memory
validation. Until then, watercnn rep 1 is missing, not counted from old outputs.
Accepted production coverage remains 30 runs, not 33. No analysis job is submitted.

Submitted clean replacement **4952823**, array task **0 only**, no dependency
because memory smoke `4952358` already completed successfully. `scontrol` confirms
the exact array, six-hour exclusive c23mm/96-CPU/480G allocation and new campaign
export. It is pending on Priority, not yet accepted as a completed measurement.
No build or other measurement job was submitted in this follow-up.

Before submission: isolated PLAN_ONLY output confirmed memory/watercnn/rep1,
six steps and exactly three configurations; shell syntax and diff checks passed;
15 campaign/analysis tests passed, including superseded-SUCCESS rejection.
Scheduler `--test-only` passed; preview `4952815` is not a submitted job.
Original runner/wrapper/gate/helper files were not edited. Only analysis/test,
supersession metadata and campaign documentation/ID records changed.

Latest follow-up scheduler snapshot:

- `4952582`: both replacement timing-smoke tasks pending on Priority; no runtime
  logs exist and no failure has occurred yet. No new gate fix is warranted.
- `4952588`: all ten non-PhyDLL timing production tasks pending on afterok:4952582.
- `4952363`: tasks 0..5 COMPLETED/0:0; tasks 6..7 running; tasks 8..19 pending.
- `4952361`: tasks 1..4 COMPLETED/0:0; tasks 5..6 running; tasks 7..9 pending.
  Task 0 remains FAILED/2:0 and excluded, superseded by pending `4952823_0`.
- `4952823_0`: clean replacement pending on Priority.

No indefinite wait was performed. Timing-smoke runtime and the new memory rerun
remain unverified until they execute; future failures must remain excluded.
