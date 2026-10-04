# Final CPU Terrain Analysis

Analysis only: no submissions or solver runs.
Run from `mini_app` using the campaign's pinned Python environment:

```sh
PYTHON=../CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python
# Smoke: exactly 12 runs, warmup step 2 plus steady steps 4 and 6
"$PYTHON" terrain_cpu_final_analysis.py --campaign-root results/terrain_final_smoke --reps 1 --steady-steps 2 --output-dir campaign_figures/final_smoke
# Performance: exactly 60 runs; all ten steady steps per run
"$PYTHON" terrain_cpu_final_analysis.py --campaign-root results/terrain_final_performance --output-dir campaign_figures/final_performance
# Separate memory follow-up: five reps, TPQ1/12 and optionally PhyDLL C++
"$PYTHON" terrain_cpu_final_analysis.py --mode memory --campaign-root results/terrain_final_memory --steady-steps 2 --reference-memory campaign_figures/nobatch_4584230/memory_latency/memory_latency_replicates.csv --output-dir campaign_figures/final_memory
"$PYTHON" -m unittest test_terrain_cpu_final_analysis
```

The actual runner's roots are `results/terrain_cpu_final/<DATASET_ID>/performance` and `results/terrain_cpu_final/<DATASET_ID>/memory` (`DATASET_ID` defaults to `final`); substitute those for the illustrative roots above. Roots may be dataset directories (analysis selects the requested mode), mode directories containing `job_*/model/repN/results.csv`, or individual `job_*` roots. Select only the intended dataset; duplicates, failures, mixed modes, and unexpected repetitions are errors. Memory campaigns with 22 steps use `--steady-steps 10`. The memory validator audits sampler artifacts, not timing logs; memory timings never enter performance plots.

## Runner Contract

Required CSV columns match `terrain_cpu_final.sbatch`:

```text
model,rep,config,status,cubex,log,mode,db_batch,total_bs,
phydll_io_log,phydll_io_log_barriers,internal_mem_verbose,internal_mem_collective,
monitoring,timing_sync,solver_ranks,dl_ranks,rank_grid,db_min_batch,db_timeout,
omit_threads,omit_set_cpus,tpq,intra,inter,telemetry,steps,steady_steps,policy,
thread_settings,provenance,run_environment,environment_sha256,run_environment_sha256,
memory_sampler,phydll_logging,internal_memory,binding
```
Models are `watercnn` and `giant`; reps are literal `1` through `--reps`; status is `SUCCESS`; mode is `performance` or `memory`. Config aliases follow `extract_scorep_tables.ALIASES` (including existing runner names). Performance requires all six configurations; memory requires TPQ1 and TPQ12, optionally `PHYDLL_CPP`.

`total_bs` must be exactly 8294400 for WaterCNN (3840x2160) or 46080 for Giant (240x192). `db_batch` must be total for TPQ1, total/12 for TPQ12 (exactly divisible), and literal `0` for default and TPQ96. Non-SmartSim configs use `0`. Performance `log` points to the complete `STEP_TIMING` log, including warmup step 2 and every expected even steady step. `cubex` points to its unique archived profile. Paths may be absolute, relative to `results.csv`, or relative to the job root; profile basenames may resolve under `config/scorep`. Memory artifacts remain under `repN/<original-config>/memory_summary.json` and `memory_samples.csv`.

New data is always strictly validated. Logging, barriers, telemetry and both internal-memory fields must be 0; timing sync must be 1. Monitoring and `memory_sampler` are 0 for performance or 1 for memory; `phydll_logging` and `internal_memory` aliases must also be 0. Solver ranks are 96, DL ranks are 96 only for PhyDLL, and the rank grid is `12x8`. Steps and steady steps must match the analysis invocation. Minimum DB batch and timeout are 0. AIx has TPQ/intra/inter=0/0/0 and `thread_settings=unforced`, not an asserted zero-thread runtime. Other configurations have inter=1. Only default has both omit flags set to 1 and `thread_settings=omitted_nominal` (recorded TPQ/intra are nominal 1/1 but omitted from effective RedisAI configuration). Remaining configurations have `thread_settings=explicit`. The policy string must be `final_total_per_queue_except_default_and_TPQ96`.

For each manifest, analysis derives `manifest.parents[2]/provenance.json` and requires the runner's `provenance`, `run_environment`, and `environment_sha256` paths to match that job/config layout. Recorded provenance CPU/mode/policy metadata must agree with the run. Input hashes must match files, and solver/DL library listings must resolve the same approved PhyDLL library. It verifies `run.env` against both the file attestation and CSV `run_environment_sha256`, then checks stable effective exports, SmartSim/PhyDLL thread controls, mode and monitoring token. Multiline exported module functions are not policy inputs; the full snapshot is still hash-attested. `command.sh` is audited for CPU, MPI counts, rank grid, steps, model layout, batch policy, attested model/input/config paths, and MPI forwarding of the semantic exports. This audits recorded launch evidence, not direct remote-process inspection. It never reads the analysis process's environment to infer run policy.

Input-hash verification intentionally checks current files against the recorded job-start hashes. Editing the runner or another hashed input after a job starts makes that dataset fail validation. Finalize those inputs before launch and preserve them; analysis does not refresh hashes or weaken this check. Legacy reference records and artifacts are handled separately and are not rewritten or subjected to the new-runner CSV schema.

The built-in stable expected provenance is:

```json
{
  "phydll_default_logging": 0,
  "phydll_default_barriers": 0,
  "approved_phydll_sha256": "dd551169e436d07e367fc5304389eafba3999f43587a20af0ec0fc8b8c87f553"
}
```

`--expected-provenance expected.json` may supply a subset of these three fields, for example to approve a newly reviewed rebuilt library hash. Logging/barrier defaults must remain 0; a full 64-character lowercase SHA-256 is required. Job IDs, paths and other varying allocation environment fields are deliberately not compared wholesale. Missing provenance, missing runtime evidence, mismatched hashes, and unsafe exports are errors, never silently unverified.

## Artifacts And Interpretation

Model display labels are `[TS] WaterCNN` and `[TS] Giant MLP`: `[TS]` identifies
the terrain-solver source class, distinguishing it from TBL. Model keys and
artifact filenames are unchanged. Selective display-only rerenders use retained
replicate CSVs and rendering functions, not `main()`; memory retains maxima of
per-run sampled peaks, and the library phase legends retain their grouped borders.

Performance writes all-step boxplots (two model panels, six colors, readable labels, fixed-decimal ticks, median-second labels above boxes, and a common independent-run-count caption), step CSV, replicate CSV, mean CSV, and validated Score-P phase tables/plots. Latency defaults to logarithmic scale; `--latency-scale linear` remains available. There are no per-config `n=5` labels. Extraction and phase rendering run as subprocesses using `--python` (defaults to the pinned SmartSim interpreter). Existing simplified phase groups are retained. Score-P coverage validation uses latency rows; it does not require memory artifacts despite the validator's `memory_rows` parameter name.

Memory writes new-run replicate/mean CSVs plus `memory_maxima.csv`. Plots show
the maximum of each metric's audited per-run simultaneous total peaks across
replicates, with fixed-decimal maximum-GiB labels and no standard-deviation bars.
Means remain supplemental CSVs, not plotted peak values; separate role/PID peaks
are never summed. Optional legacy comparison imports only AIx/default/TPQ96, plus
legacy PhyDLL if no new PhyDLL exists. It re-audits JSON/CSV sampler scope and exact
census, checks legacy successful manifests, and requires config-specific zero-batch
evidence from archived batch logs for SmartSim. Old TPQ1/12 are never imported.
New PhyDLL is preferred; legacy PhyDLL's different logging policy is disclosed.
Missing evidence causes rejection rather than assuming a matching policy.

`report.json` separates new and legacy records, records metadata and artifact hashes, runner provenance, policy and quality warnings, and each configuration's source dataset. Memory peaks cover lifetime including warmup, solver/DL/DB/support; RSS double-counts sharing while PSS apportions it. These are non-atomic sampled lower bounds, not cgroup charges or summed independent process peaks. Six-step and 22-step durations differ; comparisons are not claims of identical observation windows. Tests execute the actual runner's Python `record()` block against synthetic logs/profiles and fake provenance/runtime artifacts, never its launch code. They do not publish fabricated measurements.
