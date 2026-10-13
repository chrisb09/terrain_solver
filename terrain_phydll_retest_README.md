# Terrain CPU PhyDLL Master Retest

No jobs were submitted and no commits were made during implementation.
Run commands from `/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app`.

## Submission

```bash
# Read-only prerequisite check and exact dependency command preview.
bash build_terrain_phydll_retest.sbatch --preflight
bash submit_terrain_phydll_retest.sh

# Explicitly submit build, dependent performance, then dependent memory arrays.
RETEST_CAMPAIGN=phydll_master_20261012 bash submit_terrain_phydll_retest.sh --submit
```

The helper is dry-run by default. Only `--submit` invokes `sbatch`. It clears
inherited `SBATCH_*` options and uses explicit export lists. Its equivalent
commands are:

```bash
build=$(sbatch --parsable --export=NONE build_terrain_phydll_retest.sbatch)
build=${build%%;*}
performance=$(sbatch --parsable --dependency="afterok:$build" \
  --export=MODE=performance,RETEST_CAMPAIGN=phydll_master_20261012 \
  terrain_phydll_retest.sbatch)
performance=${performance%%;*}
sbatch --dependency="afterok:$build:$performance" \
  --export=MODE=memory,RETEST_CAMPAIGN=phydll_master_20261012 \
  terrain_phydll_retest.sbatch
```

## Methodology

Each array has 20 tasks, limited to two concurrent exclusive c23mm allocations.
Tasks 0..9 use readiness false, tasks 10..19 readiness true. Within each group,
0..4 are watercnn reps 1..5, 5..9 giant reps 1..5. The wrapper delegates to
`terrain_cpu_final.sbatch`, selecting only `PHYDLL_CPP`. No old workflow is
invoked: `terrain_cpu_final_workflow.sbatch` submits its original mixed-provider
campaign and is deliberately not appropriate for this retest.

Preserved treatment: 96 solver + 96 native C++ DL ranks, paired core pinning,
12x8 grid, CPU, Torch intra/inter 1/1, synchronized step timing, Score-P MPI/user
profiling, no tracing/PAPI, telemetry/internal memory/logging/barriers off.
Water uses `prep_watercnn_4k.h5` (8294400 samples); giant uses
`prep_giant_240x192.h5` (46080). Original CPU TorchScript models and HDF5/I/O
settings remain unchanged. PhyDLL TOML inference cap stays 500000, as in the
original final runner; its CLI cap does not override this TOML setting.

Performance: 22 steps, warm ML step 2, ten steady calls at steps 4..22, five
repetitions per model/readiness. Memory: separate six-step runs, same five reps,
two validation-only steady calls, external sampler at 1 second, requiring
simultaneous direct coverage of 96 solver + 96 DL processes. Memory timings
are not performance evidence. Total: 20 performance and 20 short memory runs.

The new profile gate parses the merged MPMD Cube archive and requires all 96
solver and all 96 DL rank trees, the expected per-rank steady receive/forward/send
counts, and no output-send MPI logging barriers. A cubex alone is not success.
Zero-copy can legitimately remove or reduce allocation/copy regions; do not
require old allocation visit counts as evidence of coverage.

## Builds And Evidence

Dedicated tree: `solver_cpp/build_terrain_phydll_retest/`, containing the solver,
`cpp-ml-interface-build/`, `dl-client/`, and `phydll/{lib,include}/`.
The C core uses the existing Makefile `compile` target with an explicit BUILD
override; no default build/install, Fortran/Python installation, or clean target.
CMI is configured with PhyDLL only, avoiding AIx bootstrap/shared installs.
SmartRedis and CUDA-enabled libtorch are read-only existing dependencies.

Build preflight checks the actual remote master with `git ls-remote`, equality
with clean local CMI HEAD, the zero-copy ancestor and field-view/buffer-reuse
code, clean PhyDLL source, and prerequisite paths. On 2026-10-12 the remote and
local master were `9fc6085a1113dda76bdc0c107bee7c73f59e7e1e`, including zero-copy
commit `7952128`. No checkout/reset/fetch changes are performed. If the remote
advances before the build, the job refuses to compile a stale local checkout.

Ready-build reuse is read-only, conditional on matching master and checked source
and binary hashes. An unfinished tree is refused, not overwritten or resumed.
Sources are hashed before and after build and archived. The atomic BUILD_READY
marker attests hash manifests; runtime checks marker, sources, binaries, registry,
caches, MPI/user Score-P linkage, relocations and the dedicated resolved C library.
The existing logging-switch/hash gate remains active, approving the newly built
dedicated library through its attested hash rather than the old hardcoded hash.

Outputs stay compatible with the final runner layout:
`results/terrain_cpu_final/CAMPAIGN_readiness_{false,true}/{performance,memory}/job_JOBID_TASK/MODEL/repN/PHYDLL_CPP/`.
Task in this path is the model/rep index 0..9; readiness is in the dataset name.
Each has `solver.log`, `command.sh`, `run.env`, `scorep/*.cubex`,
`profile_validation.json`, and memory evidence where applicable. `results.csv`
and allocation provenance are in their original parent locations. Provenance
also records readiness, build path, config and build-manifest hashes. Never pool
readiness variants, or memory and performance data, when analyzing.

## Validation And Risks

```bash
bash -n build_terrain_phydll_retest.sbatch terrain_phydll_retest.sbatch \
  terrain_phydll_retest_gate.sh submit_terrain_phydll_retest.sh terrain_cpu_final.sbatch
python3 -m unittest test_terrain_phydll_retest terrain_cpu_final_test \
  terrain_memory_sampler_test terrain_cpu_diagnostics_test
PLAN_ONLY=1 MODE=memory SLURM_ARRAY_TASK_ID=19 bash terrain_phydll_retest.sbatch
```

CPU build/runtime have not been executed. The dedicated build requests 32 CPUs,
180 GB, and 4 hours on c23ms (`rwth2150`). CUDA-enabled libtorch still needs
the site CPU stub resolution used by the existing CPU treatment; build stubs
must not be embedded in RUNPATH. Build and runtime use existing site modules
and Score-P/Cube installation. Runtime relocation failures, missing Cube
tooling, rank-count/visit mismatches or short-run memory coverage fail the task.
The source gates intentionally reject concurrent source edits after building;
they do not modify or stop other sessions. Heavy compilation is allocation-only.
Full campaign analysis/plots are not automatically submitted; the original
mixed-provider analysis must not be used unchanged to pool these variants.

Implementation validation passed: all five shell syntax checks, remote/master
and prerequisite preflight, submission-helper dry-run (no sbatch), and 31 tests
across the new retest tests and the existing final-runner, memory-sampler and
diagnostics suites. New tests cover all 40 model/readiness/treatment mappings,
invalid indices, missing DL ranks, incorrect steady counts and logging barriers.
