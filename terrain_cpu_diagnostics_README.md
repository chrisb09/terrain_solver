# Targeted CPU Terrain Diagnostics

This is a separate, review-before-submit runner, not a campaign or submission gate.
It does not build binaries, submit other jobs, or edit existing benchmark code.
One `SUITE` and one `MODEL` are required per exclusive, single-node allocation.
Defaults: `c23mm`, 480 GB, 96 solver ranks, 12 x 8 decomposition, 10 steps,
three-hour allocation limit. PhyDLL also launches 96 C++ DL ranks on that node.

Submitted jobs are recorded in `terrain_cpu_diagnostics_jobs.csv`: AIx
WaterCNN/Giant `4617648`/`4617644`, PhyDLL `4617647`/`4617645`, and SmartSim
`4617649`/`4617646`. Analysis job `4617661` depends on all six succeeding and
runs `terrain_cpu_diagnostics_finalize.sbatch`. It writes a combined report
under `campaign_figures/diagnostics_<analysis_job>/`; it never submits a full
campaign. Earlier `4599130`-`4599135` attempts were incomplete due to MPI
consuming the plan stream and are excluded from these comparisons.

## Treatments

| Suite | A / off | B / on | Default schedule | Runs per allocation |
|---|---|---|---|---:|
| `aix` | No sampler | Existing external `terrain_memory_sampler.py`, 1 s interval | ABBA ABBA | 8 |
| `phydll` | `PHYDLL_IO_LOG_BARRIERS=0` | `PHYDLL_IO_LOG_BARRIERS=1` | ABBA | 4 |
| `smartsim` | DB batch size 0 | DB batch size TOTAL_BS / TPQ | Six treatments, then exact reverse | 12 |

AIx defaults to four matched repetitions per condition (two ABBA cycles).
PhyDLL defaults to two matched repetitions per condition (one ABBA cycle).
Within each ABBA cycle, pair 1 is off/on and pair 2 is on/off. `ABBA_CYCLES=1`
reduces AIx to four runs; `ABBA_CYCLES=2` doubles PhyDLL to eight runs.
SmartSim uses at least two repeats (`REPEATS=2`), with each repeat containing
each config and both batching conditions; every even repeat reverses the entire
previous odd repeat's order. All treatments within a job use the same binary
and node. Models are separate allocations, not a pooled node comparison.

| SmartSim config | TPQ | Intra | WaterCNN on batch target | Giant on batch target |
|---|---:|---:|---:|---:|
| TPQ1_I96 | 1 | 96 | 8294400 | 46080 |
| TPQ12_I8 | 12 | 8 | 691200 | 3840 |
| TPQ96_I1 | 96 | 1 | 86400 | 480 |

WaterCNN TOTAL_BS is 8294400; Giant TOTAL_BS is 46080. RedisAI minimum batch size
and minimum batch timeout are explicitly zero for *both* conditions. These are
batch targets, not proof that RedisAI executed a full target batch.
The positive client cap `--ml-batch-size 10000000` is separate and unchanged.
No implicit/default SmartSim config is included: all three explicitly set TPQ,
intra-op threads, inter-op threads=1, and DB CPU allocation=96.

## Controls

The campaign's environment setup, model inputs, CPU device, solver arguments,
and flat AIx/PhyDLL versus split SmartSim model layouts are retained.
`STEPS=10` includes ML warmup at step 2 and four steady ML steps at 4/6/8/10.
Every fresh process/model/DB run includes the same warmup; no condition gets a
separate priming run. Regular steps and warmup are excluded from latency summaries.
The final trajectory save is after the final timing line, as in the campaign.

- The external sampler runs **only** for AIx on, including startup/readiness and
  stop/wait/summary checks. Off never starts a monitor or needs tracked PIDs.
- Internal memory logs are off: `MLCOUPLING_MEM_LOG_VERBOSE=0` and
  `MLCOUPLING_MEM_LOG_COLLECTIVE=0`; timing sync is explicitly held at 1.
- `PHYDLL_TELEMETRY=0` and `SMARTSIM_FLAG_TELEMETRY=0` are explicit. PhyDLL
  text output is held on with `PHYDLL_IO_LOG=1` for both barrier treatments:
  printing remains enabled independently of `PHYDLL_IO_LOG_BARRIERS=0/1`.
  Logging, barriers, telemetry, threads, and Score-P exports are identical on both MPMD
  command segments. No sampler or other monitor runs for PhyDLL or SmartSim.
- Score-P profiling stays on, tracing/PAPI off. Each run has a new profile
  directory; overwrite is disabled. Profiles include warmup and steady callpaths.
- MPI is unbound/oversubscribed as in the campaign, with binding reports logged.
  No OMP/MKL/OpenBLAS thread count or OMP dynamic override is imposed or explicitly
  exported through MPI. AIx keeps the campaign's runtime thread defaults, not a
  forced single-thread treatment. PhyDLL LibTorch intra/inter threads remain
  explicitly 1; SmartSim retains the three explicit RedisAI thread configs.
  Inherited benchmark/affinity environment controls are cleared before setup.
- Binary and input hashes, resolved shared libraries, full exported environment,
  CPU topology, batch affinity, Slurm allocation, and shell-quoted commands are saved.
  Full environment files can contain inherited credentials; review before sharing.
- Helper and sampler Python default to the absolute canonical venv executable
  `/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python`,
  pinned and checked before the first plan call. No ambient `python3` is used by
  the runner. Overrides must be absolute executable paths; do not resolve the
  interpreter symlink out of its venv, which can change Python's environment.

Existing executable defaults are the campaign's `build_scorep_cmi_fresh/terrain_solver`,
`dl_clients/build-scorep/phydll_dl_client`, and SmartSim Python. The runner checks
executability and all model/input files; it does not compile anything. Executable
existence does **not** prove the loaded PhyDLL library implements the logging-barrier
switch or the solver implements memory-log controls. Use the current runtime builds;
inspect saved library resolution and Score-P barrier evidence before interpreting
a zero treatment difference. For PhyDLL, `inputs.sha256` also hashes the real
`libphydll.so` paths resolved by `ldd` for both solver and DL client; identical
paths are deduplicated, different libraries are both hashed. Missing/unresolved
libraries fail preflight. The opt-in logging build must honor `PHYDLL_IO_LOG=1`
independently of the barrier flag. `PHYDLL_TELEMETRY=0` is harmless provenance,
not a real telemetry switch in the current library.

## Lifecycle And Failure

Each SmartSim controller starts from that run's directory, so its experiment
directory and unique `close_driver_<run_id>.txt` are isolated. After solver exit,
the runner touches that done file and waits up to 60 seconds for natural
`exp.stop`. It never uses a process-name-wide kill or a sampler to stop Redis.
Before controller startup it snapshots existing numeric steps in this job. At
cleanup, only new numeric Slurm steps in this allocation may be cancelled,
and the controller timeout wrapper may be terminated after its saved PID/start
identity is rechecked. Any forced cleanup,
controller error, surviving step, or Slurm query failure invalidates the run
and stops the suite. Do not start unrelated `srun` steps in this allocation while
the runner owns it; newly created steps during a controller launch are its scope.
The enclosing allocation remains the final containment boundary on cancellation.

`RUN_TIMEOUT=1800` bounds each solver invocation; `DB_START_TIMEOUT=180` bounds
DB readiness. INT/TERM/EXIT clean up active resources. SIGKILL cannot execute a
shell trap. Sampler failure, inadequate coverage, live memory read errors, missing
or duplicate steady timing lines, absent/ambiguous profiles, solver failure, or
cleanup failure produces a failed row and nonzero exit, with no next treatment.
Startup failures leave a real `solver.log` explaining that the solver was not
launched, and an empty profile path rather than a fabricated archive location.
Failures before a run starts can leave only plan/preflight artifacts and a nonzero
Slurm status. Analysis rejects failed or incomplete plans, not just missing pairs.
Solver/MPI, controller, and sampler launches read stdin from `/dev/null`, never
the scheduling loop's `plan.tsv`. This prevents MPI stdin forwarding from draining
the remaining treatments after run01. Cleanup's shell reads use explicit files
or here-strings; they do not read the plan stream.

## Submission

Run from `/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app`; existing `logs/`
must exist before submitting. These commands are instructions, not executed jobs.

```bash
sbatch --job-name=terrain_cpu_diagnostics_aix_watercnn --time=01:00:00 --export=ALL,SUITE=aix,MODEL=watercnn terrain_cpu_diagnostics.sbatch
sbatch --job-name=terrain_cpu_diagnostics_aix_giant --time=01:00:00 --export=ALL,SUITE=aix,MODEL=giant terrain_cpu_diagnostics.sbatch
sbatch --job-name=terrain_cpu_diagnostics_phydll_watercnn --time=01:00:00 --export=ALL,SUITE=phydll,MODEL=watercnn terrain_cpu_diagnostics.sbatch
sbatch --job-name=terrain_cpu_diagnostics_phydll_giant --time=01:00:00 --export=ALL,SUITE=phydll,MODEL=giant terrain_cpu_diagnostics.sbatch
sbatch --job-name=terrain_cpu_diagnostics_smartsim_watercnn --time=01:00:00 --export=ALL,SUITE=smartsim,MODEL=watercnn terrain_cpu_diagnostics.sbatch
sbatch --job-name=terrain_cpu_diagnostics_smartsim_giant --time=03:00:00 --export=ALL,SUITE=smartsim,MODEL=giant terrain_cpu_diagnostics.sbatch
```

Submit/review in suite order rather than launching a full campaign. Scheduling
queue wait is not included in runtime estimates. Previous 22-step production
rep1 walls were AIx 69/71 s, PhyDLL 54/142 s (WaterCNN/Giant). The 10-step AIx
suites should usually take roughly 5-15 minutes; PhyDLL roughly 3-10 minutes.
SmartSim WaterCNN should take roughly 15-30 minutes. Giant TPQ1/I96 disabled
previously took 1771 s for 22 steps (about 160-164 s per inference): five ML
calls in a 10-step run imply about 820 s before startup/shutdown. Two disabled
repeats alone cost about 27 minutes; if both on repeats cost similarly, these
four runs cost about 55 minutes. With other configs and lifecycle overhead,
budget roughly 65-90 minutes, not a one-hour devel allocation. Three hours is
a scheduling safety margin, not a promise that all timeouts fit sequentially.
Batching performance is intentionally unknown; fail-fast may terminate sooner.

Overrides: `STEPS` (even >=6), `ABBA_CYCLES`, `REPEATS` (>=2), `RUN_TIMEOUT`,
`DB_START_TIMEOUT`, `MEMORY_INTERVAL` (sampler accepts 0.2-1 s), `HELPER_PYTHON`, `MONITOR_PYTHON`,
`SOLVER_BIN`, `PHYDLL_DL_CLIENT`, `SMARTSIM_PYTHON`. Rank layout and thread
treatments are deliberately not tunable in this diagnostic runner.

## Outputs And Analysis

Unique root: `results/terrain_cpu_diagnostics/<job-name>_<job-id>/`. Existing
roots are refused. The outer Slurm log is `logs/terrain_cpu_diagnostics_<job-id>.log`.

```text
<root>/
  plan.tsv, results.csv, inputs.sha256, allocation.env
  slurm_job.txt, lscpu.txt, batch_affinity.txt, *_libraries.txt
  run01/ ... run08/ (AIx), run04/ (PhyDLL), or run12/ (SmartSim)
    command.sh, run.env, solver.log, traj.h5, scorep/*.cubex
    controller_command.sh, controller.log, db_host.txt, smartsim_experiments/... (SmartSim)
    memory_*.csv, memory_summary.json, memory_monitor.log, memory.ready/stop (AIx on only)
    cleanup.log (only if a controller step required cancellation)
  analysis/
    steady_steps.csv, per_run.csv, conditions.csv
    paired.csv, paired_summary.csv, report.md
    scorep_forward_barrier.csv (optional)
```

`results.csv` records order, matched rep, condition, TPQ/intra, batch target,
sampler/barrier settings, steps, solver wall time (excludes DB startup/shutdown),
status, log and profile paths. Interrupted rows instead record elapsed time to
cleanup, and are never included in analysis. All raw timing lines/profiles are retained.
Successful suites automatically create latency analysis; re-run with profiles:

```bash
/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python terrain_cpu_diagnostics.py --root results/terrain_cpu_diagnostics/terrain_cpu_diagnostics_phydll_giant_JOBID --scorep
```

Replace `JOBID` with the actual submitted ID. `per_run.csv` reports four-step
means and medians plus rank-min/rank-max summaries. `conditions.csv` aggregates
replicate means/medians with equal weight. `paired.csv` compares on versus off
within each matched rep, and `paired_summary.csv` aggregates these differences
and ratios. No individual step or rank is treated as an independent replicate,
and no pseudo-independent confidence intervals are reported. One node and only
2-4 matched repeats cannot establish machine-wide statistical confidence.

Optional Score-P analysis imports existing `extract_scorep_tables.extract` and
its parser, without editing them or using campaign discovery. It preserves
callpaths, rank scope/count, steady visits, raw time, and normalized inclusive
and exclusive milliseconds per rank per steady step for forward and MPI barrier
regions. These are not wall latency and nested inclusive rows must not be summed.
Solver steady/warmup scope and PhyDLL DL visit counts are validated; ambiguous
DL warmup aggregation raises an error instead of guessing subtraction. Missing
regions are not invented zeroes. The existing Cube parsing tools must be available
in the analysis environment (`source ../CPP-ML-Interface/env_scorep.sh` if needed).

## Local Checks

```bash
bash -n terrain_cpu_diagnostics.sbatch
TMPDIR=/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode PYTHONDONTWRITEBYTECODE=1 /hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python terrain_cpu_diagnostics_test.py -v
```

Fixtures cover ABBA pairing, reverse SmartSim order, both models' batch targets,
monitor treatment scope, steady/warmup filtering, missing/duplicate/nonfinite
timings, complete manifest checks, paired means, replicate weighting, and reused
Score-P MPMD rank-scope normalization/warmup rejection, scoped Slurm-step cleanup,
Slurm-query failures, sampler stop/wait failure propagation, loaded-library hashes,
Score-P output values/source attribution, and startup-failure manifest paths.
No compute jobs are run.
The shell stdin regression runs all three ordered plans with a mock child that
attempts to drain stdin. It also removes the solver redirection as a negative
control, reproducing the one-run truncation and incomplete-plan rejection.
