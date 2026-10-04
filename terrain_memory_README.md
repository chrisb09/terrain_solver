# No-Batching CPU Terrain Rerun

The campaign uses external memory sampling and steady-callpath Score-P analysis.
New outputs stay separate from the earlier automatically batched campaign.

## Automated Workflow

Validation jobs `4580941` (WaterCNN) and `4580942` (Giant) run all six configs
with six simulation steps, giving two steady inference steps. Gate job `4580957`
depends on both succeeding. It validates memory, latency, and Score-P coverage
before submitting a production array of ten tasks (two models x five reps),
with at most two tasks running concurrently. Each task runs six configs.

`validate_submit_terrain_nobatch.sbatch` writes the production/analysis job IDs
to its figure directory's `production_submission.txt`. Production results use
`results/terrain_nobatch/validated_<gate_job>/`. The dependent
`analyze_terrain_nobatch.sbatch` generates memory, latency boxplots, mean latency,
application phase stacks, and grouped/stacked provider detail plots after all
production tasks succeed. A failed validation does not submit production jobs.

Earlier validation jobs `4578776`/`4578777` exposed forced cleanup of
controller-side Python helpers. Their gates were canceled before submission.
The current campaign sets `SMARTSIM_FLAG_TELEMETRY=0` on controller launch:
optional SmartSim telemetry is not needed for inference or external sampling.
Command lines are now recorded to identify any surviving support process.

## Campaign

Defaults: `watercnn` and `giant`, five repetitions each, 22 steps, 96 solver
ranks on a 12 x 8 grid. Six configs are run sequentially:
`AIX`, `PHYDLL_CPP`, `SS_DEFAULT`, `SS_TPQ12_I8_B96`, `SS_TPQ96_I1_B96`,
`SS_TPQ1_I96_B96`. `SS_TPQ1_I1_B1` is removed.

Every SmartSim launch explicitly passes `--ml-db-batch-size 0`,
`--ml-db-min-batch-size 0`, and `--ml-db-min-batch-timeout 0`. The separate
client-side `--ml-batch-size 10000000` cap is retained; the solver requires it
to be positive. It is not RedisAI auto-batching.

Run from the mini_app directory. `logs/` must exist before `sbatch` starts.
Examples below are instructions, not commands executed by the implementation.

```bash
# All six configs, one model/rep, shorter steps, production rank layout.
sbatch --time=00:45:00 --export=ALL,MODEL=watercnn,REP=1,STEPS=6,RUN_TIMEOUT=420 campaign_cpu_terrain.sbatch

# Smaller rank smoke. Do not compare its performance to the 96-rank campaign.
sbatch --time=00:30:00 --export=ALL,MODEL=watercnn,REP=1,STEPS=6,MPI_RANKS=4,RANK_GRID_X=2,RANK_GRID_Z=2,RUN_TIMEOUT=420 campaign_cpu_terrain.sbatch

# Single configuration, including DB startup and shutdown sampling.
sbatch --time=00:15:00 --export=ALL,MODEL=watercnn,REP=1,STEPS=6,CONFIGS=SS_DEFAULT,RUN_TIMEOUT=420,MEMORY_INTERVAL=0.5 campaign_cpu_terrain.sbatch

# Production array: one model/rep per allocation, after validation.
sbatch --array=0-9%2 campaign_cpu_terrain.sbatch

# Recommended production scheduling unit: one model/rep per allocation.
# Repeat for each model and REP=1..5, after smoke validation.
sbatch --export=ALL,MODEL=giant,REP=1 campaign_cpu_terrain.sbatch
```

Overrides: `MODEL` or space-separated `MODELS`; `REP` or `REPS` (number of
repetitions starting at one); space-separated `CONFIGS`; `STEPS`, `MPI_RANKS`,
`RANK_GRID_X`, `RANK_GRID_Z`, `RUN_TIMEOUT`, `DB_START_TIMEOUT`,
`MEMORY_INTERVAL` (0.2-1 seconds); `MONITOR_PYTHON`, `SOLVER_BIN`,
`PHYDLL_DL_CLIENT`, `SMARTSIM_PYTHON`; `OUTPUT_BASE`, `RUN_ROOT`, `LOG_ROOT`.
For space-separated lists, set/export the variable and use `--export=ALL`.
Rank-grid product must equal solver rank count. PhyDLL retains one C++ DL
process per solver rank. SmartSim thread settings and 96-CPU DB binding are
unchanged, even in smaller-rank smoke tests.

Default output root is `results/terrain_nobatch/job_<job>_<timestamp>_<pid>`.
An explicit `RUN_ROOT` must not already exist. Each model/rep has `results.csv`
and config subdirectories containing trajectory, Score-P profile, and memory
artifacts. Logs are under `<root>/logs/terrain_nobatch_<model>_r<rep>_<cfg>_<job>.log`;
controller logs use the same new namespace. Slurm's outer log is
`logs/terrain_nobatch_<job>.log` (override with `sbatch --output=...`). Missing
step 4 in very short smoke is empty in results, not zero. Failures are counted
and give the campaign a nonzero exit status.

`plot_terrain_nobatch.py --campaign-root <root>` validates and plots the new
namespace. `extract_scorep_tables.py --campaign-root <root>` extracts phases,
and `plot_scorep_phases.py` renders them. Do not merge these logs into the old
campaign. The legacy `plot_campaign_boxplots.py` still targets the old data.

## Memory Measurement

The sampler starts before DB/solver launch and samples through controller
shutdown. It is external to MPI. The campaign sets and exports
`MLCOUPLING_MEM_LOG_VERBOSE=0` and `MLCOUPLING_MEM_LOG_COLLECTIVE=0`. Use a solver
binary rebuilt from the current source implementing these controls; an older
binary may still run memory collectives inside timed steps. The solver's
memory parser was fixed and its Score-P binary rebuilt. Existing timing/physics collectives and
post-run solver memory summary behavior are not changed.

Each config has a unique `TERRAIN_MEMORY_TOKEN`, explicitly exported to both
MPMD applications. The sampler reads same-UID `/proc` processes and identifies
tagged processes plus descendants, retaining `(PID, starttime)` identities
through reparenting and checking identities after memory reads and before
cleanup signals. Roles use exact real executable paths for solver and DL;
Redis/Valkey executable names identify DB. Child processes inherit the nearest
application role; controllers, timeout, MPI/Slurm launchers are `support`.
The sampler and batch shell are excluded. No node-wide or user-wide process
name matching/kill is used.

Redis can live beneath `slurmstepd`, not the controller. If its token is stripped,
the fallback requires the exact configuration port, same Slurm job evidence
from environment or cgroup path, same UID, and a PID/start identity absent from
the pre-launch baseline. It cannot safely infer a DB whose token, job evidence,
and process ancestry are all inaccessible. Coverage metadata makes this failure
visible rather than silently claiming a complete measurement. `/proc` visibility
and PSS permissions must be validated on compute nodes. This helper is single-node.

Per-config files:

- `memory_samples.csv`: total and solver/DL/DB/support RSS and PSS sums from the
  same sweep, read counts, direct process census, timestamps, actual interval,
  and sweep duration. The requested default interval is one second.
- `memory_processes.csv`: per-sample PID/starttime, parent, executable, command line, role,
  attribution source, cgroup path, RSS/PSS, and read errors.
- `memory_summary.json`: sampled simultaneous total/role peaks, unique and peak
  direct process counts, expected counts, simultaneous coverage samples, errors,
  lifecycle versus live read errors, actual interval statistics, overruns,
  maximum sweep duration, cgroup availability, and accounting caveats.
- `memory_cgroups.csv`: readable cgroup current/lifetime peak counters explicitly
  labelled `diagnostic_only_not_config`; qualifying fresh common scopes are
  labelled `fresh_exclusive_config` separately.
- `memory_monitor.log`, `memory.ready`, `memory.stop`: diagnostics and lifecycle.

RSS double-counts shared mappings. PSS apportions physical pages among all
sharers, including processes outside the measured set; it is a better
process-set footprint estimate, not a cgroup charge. No thread RSS is separately
summed. Reads within a sweep are sequential, not an atomic snapshot. Sampled
peaks are lower bounds and are **not sums of independent process HWMs** or sums
of separate role peaks. Missing RSS/PSS readings make the affected sample sum
blank; corresponding JSON peaks use only complete reads for that sample.
Processes that start and exit between samples can be missed. Inspect coverage
and errors before trusting a peak; `coverage_ok` means a same-sweep census reached
all expected direct role counts at least once, not that every read succeeded.

## Cgroup Investigation

Read-only inspection on this cluster found cgroup v2 mounted at `/sys/fs/cgroup`
on the login node. `scontrol show config` reports `proctrack/cgroup`,
`task/cgroup,task/affinity`, `jobacct_gather/cgroup`, 30-second accounting,
`CgroupPlugin=autodetect`, and `CgroupJobIdPaths=no`. Compute-node paths and
permissions were subsequently exercised in the smoke allocations. The sampler
found 96 solver processes, 96 DL processes for PhyDLL, and Redis for SmartSim.

The campaign's solver `mpirun` originates in the reused batch cgroup, whereas
the unchanged SmartSim controller starts Redis using a separate Slurm step.
A common batch/job ancestor therefore does not provide a fresh configuration
peak. Moving only the sampler into another step would not fix this; nor would
subtracting an earlier `memory.peak`. Slurm may own the hierarchy, so no cgroup
creation, process migration, or counter reset is attempted.

Whole-config current/peak is captured only when the common cgroup covering
**all selected processes** was absent from the baseline hierarchy, all current
members are selected, and the same exclusive scope remains valid at every
nonempty sweep. An unreadable baseline hierarchy or scope subtree also rejects
qualification. Otherwise whole-config cgroup fields are null. The expected
result for this campaign topology is unavailable. Diagnostic step/batch peaks
are never promoted into configuration peaks and must not be added together.
The automatic whole-config qualification currently supports cgroup v2 with a
root-mounted hierarchy; other layouts still use `/proc` measurement. Cgroup
accounting also includes file cache/kernel charges and differs from RSS/PSS.

## Validation

Local checks, without Slurm/MPI:

```bash
bash -n campaign_cpu_terrain.sbatch
TMPDIR=/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode PYTHONDONTWRITEBYTECODE=1 python3 terrain_memory_sampler_test.py -v
```

Tests cover role/descendant discovery, reparenting, PID reuse, unrelated-job and
preexisting-DB exclusion, exact-port boundaries, cgroup exclusivity, missing PSS,
and a live two-process memory workload plus unrelated process. The live test
checks same-sweep totals and scoped cleanup, preserving the unrelated process.

Before production, run one six-config smoke on each model. Confirm 96 solver
processes, 96 additional DL processes only for PhyDLL, and a Redis process only
for SmartSim are seen **simultaneously**, with readable PSS. Check process CSV
against `ps`/`/proc/<pid>/{cgroup,environ,smaps_rollup}` while a config is active.
Specifically validate the Redis step's token/job evidence and port. Confirm no
previous-config processes survive and inspect any `cleanup_sigkill_pids`.
Treat `coverage_ok=false` and live missing-PSS failures as invalid evidence.
Normal exit/PID-reuse races leave incomplete samples that are excluded from
peaks. Total peaks must have complete reads and the exact expected direct census.
Long sweeps are reported as quality warnings, not silently treated as one-second
atomic measurements.

Repeat a representative run at 0.2 and 1.0 seconds to assess peak convergence
and timing perturbation; a 192-process PhyDLL sweep can exceed the requested
interval. Compare independent process-set PSS with diagnostic job/step counters
and `sacct` only as accounting cross-checks, not equivalent metrics. For a true
kernel whole-config lifetime peak, run each config in a separate allocation,
or arrange a fresh common scope containing both MPI and DB launches with Slurm
administrator support. The 6-hour default may not fit all 60 runs; schedule one
model/rep per job after measuring smoke durations.
