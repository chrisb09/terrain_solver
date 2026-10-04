# Final Terrain CPU Runner

`terrain_cpu_final.sbatch` is a separate, immutable-output rerun. It does not
build, submit other jobs, invoke campaign analysis, or change shared libraries.
Performance and memory evidence must remain separate.

## Settings

One exclusive c23mm node, 96 CPUs, 480G; 96 solver ranks on a 12x8 grid.
MPI launches are oversubscribed with MPI binding disabled; `pin_wrapper.sh`
pins AIx solver ranks and pairs PhyDLL solver/DL ranks on the same 96 cores.
SmartSim solver launches remain unbound. PhyDLL adds 96 C++ DL ranks,
with explicit Torch intra/inter-op threads 1/1. AIx receives no new OMP/thread
overrides; inherited treatment/thread variables are cleared before the canonical
environment setup. Score-P profiling is on, tracing/PAPI metrics off, synchronized
step timing on. Both PhyDLL MPMD programs receive the same Score-P, logging,
timing and (memory mode only) token exports. `PHYDLL_IO_LOG=0` and
`PHYDLL_IO_LOG_BARRIERS=0` are explicit. SmartSim telemetry and internal memory
verbose/collective logging are off.

| Configuration | TPQ / Intra | Water DB Threshold | Giant DB Threshold |
|---|---|---:|---:|
| AIx | unchanged | n/a | n/a |
| PHYDLL_CPP | Torch 1/1 | n/a | n/a |
| SS_DEFAULT | literal library defaults, omit threads and set_cpus | 0 | 0 |
| SS_TPQ1_I96_B96 | 1 / 96 | 8294400 | 46080 |
| SS_TPQ12_I8_B96 | 12 / 8 | 691200 | 3840 |
| SS_TPQ96_I1_B96 | 96 / 1 | 0 | 0 |

All SmartSim minimum batch sizes and timeouts are zero. AIx records TPQ/intra/inter
as 0/0/0 and `thread_settings=unforced`: zero means unknown/unforced, not zero
runtime threads. `SS_DEFAULT` records nominal TPQ/intra/inter values 1/1/1,
`thread_settings=omitted_nominal`, and both omission flags set. These values are
not claims about effective RedisAI defaults. Explicit PhyDLL and SmartSim inter-op
settings are 1; their `thread_settings=explicit` applies to the configured inference
thread settings (PhyDLL TPQ=0 is not applicable).

## Preflight

The helper/sampler interpreter is explicitly
`CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python`, not ambient Python.
Before loading the canonical environment or invoking Python, bootstrap clears
`PYTHONHOME`, `PYTHONPATH`, `PYTHONOPTIMIZE`, and `LD_PRELOAD`. Required EasyBuild
module roots must be nonempty existing directories, matching the prior validation
gate: OpenSSL, Clang, GCCcore, OpenMPI, HDF5, PnetCDF, CUDA, cuDNN, imkl and hwloc.
Missing roots/library path or launch commands fail before creating a result root.
The solver and DL executable must resolve the same C `libphydll.so`, with no
missing dependencies. Provide `APPROVED_PHYDLL_SHA256`, the SHA-256 of the
reviewed rebuilt library implementing opt-in logging and barriers. Preflight
checks that hash, both switch strings in the binary, and the current C source's
opt-in defaults. Source and binary-string checks alone cannot establish build
provenance; approving the matching rebuilt artifact is the operator's responsibility.
No build is performed. Preflight records loaded-library listings, model/input,
binary, configuration, controller and helper hashes, full environment, environment
hash, Slurm allocation and CPU topology. Missing files, stale hashes, failed
helper validation, missing/duplicate cubex or inadequate memory coverage fail fast.

## Commands

Run from `mini_app`; its `logs/` directory must exist for Slurm output. Set the
approved hash in your submission environment before using these commands.
These commands are documentation only; no jobs were submitted during implementation.

```bash
bash -n terrain_cpu_final.sbatch
python3 -m unittest terrain_cpu_final_test.py
PLAN_ONLY=1 bash terrain_cpu_final.sbatch
PLAN_ONLY=1 MODE=memory SLURM_ARRAY_TASK_ID=9 bash terrain_cpu_final.sbatch

# All six configurations, six-step performance smoke, both models, rep 1.
sbatch --array=0,5 --export=ALL,MODE=performance,STEPS=6 terrain_cpu_final.sbatch
# Selected memory smoke, both models, rep 1.
sbatch --array=0,5 --export=ALL,MODE=memory,STEPS=6 terrain_cpu_final.sbatch
# Approved 60-run performance rerun: two models x five reps x six configurations.
sbatch --export=ALL,MODE=performance terrain_cpu_final.sbatch
# Separate optional memory campaign: two models x five reps x three configurations.
sbatch --export=ALL,MODE=memory terrain_cpu_final.sbatch
# TPQ-only memory subset (CONFIGS is space-separated).
CONFIGS='SS_TPQ1_I96_B96 SS_TPQ12_I8_B96' sbatch --export=ALL,MODE=memory terrain_cpu_final.sbatch
```

The production array defaults to `0-9%2`, limiting concurrency to two allocations.
Array indices 0..4 map to watercnn reps 1..5, 5..9 to giant reps 1..5.
Without an array task ID, the runner selects watercnn rep 1. `CONFIGS` accepts
any unique subset of the six exact names, including mixed-case `AIx`.
`STEPS` must be even and at least six. Default performance steps are 22:
ML step 2 is warmup; steps 4,6,...,22 are ten steady observations. A six-step
smoke gives two. Defaults are `RUN_TIMEOUT=3000`, `DB_START_TIMEOUT=180`,
`MEMORY_INTERVAL=1.0` (sampler accepts 0.2..1.0).

## Evidence

Each task creates a fresh namespace:
`results/terrain_cpu_final/DATASET_ID/{performance|memory}/job_JOBID_TASK/model/repN/`.
Existing roots are rejected, including requeues. `results.csv` has one row per
configuration with explicit mode, monitoring, policy, logging, thread/omission,
batch thresholds, counts, status, log and full cubex paths. Raw step-4 decimals
are preserved and performance means use Decimal, not rounded shell conversions.
Integer step counts are explicit. Memory rows deliberately leave timing summary
columns blank, even though raw synchronized timings and Score-P remain available
for validation. Never pool memory-run timings into performance results.

Each row includes absolute `provenance` (`ROOT/provenance.json`), `run_environment`
(`config/run.env`), and `environment_sha256` (`config/environment.sha256`) paths,
plus the `run_environment_sha256` digest of that exact environment snapshot.
Failed startup rows may have a blank digest when no launch environment was written.
The analysis's strict `--expected-provenance` option approves stable library/logging
fields; it also validates device, mode, sampler/logging/internal-memory, telemetry,
timing, rank-grid, minimum batch and policy metadata. Configuration-specific
effective variables are audited using the row's hash-attested run environment,
not the allocation-level environment in provenance. The final analysis verifies
environment hashes, launch commands, input hashes, and the approved loaded library.
`memory_sampler`, `phydll_logging`, and `internal_memory` columns also use the
analysis-recognized names, alongside the detailed runner fields.

Performance mode never launches an external memory sampler. Memory mode launches
the existing external sampler **before** the controller and MPI, tags both MPMD
programs, and requires simultaneous direct coverage of all 96 solver ranks,
96 DL ranks for PhyDLL, and one DB for SmartSim. Missing direct coverage or live
read failures invalidate the run. Natural process-exit read races are distinguished
from live read errors by the existing sampler. The controller gets a 60-second
natural shutdown grace; only new numeric Slurm steps in this allocation may be
cancelled, and cancellation marks failure. Controller identity is checked before
signalling. Sampling stops only after controller shutdown/step-drain attempts;
sampler cleanup only targets tracked PID/start identities. Forced SIGKILL invalidates
memory evidence. See `terrain_memory_README.md` for sampled PSS/RSS caveats.

The recommended memory default includes `PHYDLL_CPP`: logging-off differs from
the previous campaign, so previous memory figures are not guaranteed to match.
This adds ten short runs; TPQ-only memory remains available as shown above.

## Budget

Full performance: 60 solver launches and 600 steady ML observations, on ten
allocations. Default memory: 30 launches and 60 validation-only observations.
TPQ-only memory: 20 launches. Smoke pairs: 12 performance or 6 default memory
launches. Runtime depends on the measured node/model latency: approximately
sum of 11 ML calls per configuration for performance, or 3 for memory, plus
I/O, startup, profiling and DB shutdown. Six hours per performance task reserves
at most 60 exclusive node-hours for the ten-task array (5760 CPU-hours);
memory has the same conservative allocation limit unless overridden with
`sbatch --time`. Queue delay is not included. No node-time estimate is claimed
from memory-instrumented timings.
# Submitted Workflow

Performance smoke array `4644308` and memory smoke array `4644309` each ran
WaterCNN/Giant rep 1 (tasks 0 and 5). Gate `4644312` validated all effective
exports, pin wrappers (AIx and PhyDLL pinned, SmartSim unbound), batch settings,
Score-P profiles, and sampled memory.

The gate automatically submitted the production campaign:
- Performance array `4645178` (60 runs: 2 models x 5 reps x 6 configs, 22 steps,
  Score-P profiling on, memory sampling off, PhyDLL logging barriers off, AIx/PhyDLL
  pinned, SmartSim unbound).
- Memory array `4645181` (30 runs: 2 models x 5 reps x 3 configs, 6 steps, PSS/RSS
  sampling on, paired settings; dependent on performance finishing).
- Analysis `4645185` (automated figure generation, dependent on performance and memory).

Production data is under `results/terrain_cpu_final/production_4644312/`, separated into
`performance/` and `memory/`. Figures use `campaign_figures/terrain_cpu_final_production_4644312/`.
Job IDs are recorded in `terrain_cpu_final_jobs.csv`.

The short memory follow-ups cover TPQ1, TPQ12, and PhyDLL C++ with the final
logging and batching policies. Existing AIx, literal-default SmartSim, and TPQ96
memory measurements remain reference data, clearly labeled by their original
campaign and duration. Memory-run timings are never pooled into performance.
