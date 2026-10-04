This is a mini-app for first testing smartsim and later the CPP-ML-Interface. It runs a simple 2D simulation that is split across ranks using MPI.

## CPU Terrain Campaign

- [Final runner and validated job IDs](terrain_cpu_final_README.md): separate performance and memory campaigns, CPU pinning, and submission gate.
- [Final analysis](terrain_cpu_final_analysis_README.md): strict provenance checks, steady Score-P phases, `[TS]` plot labels, and maximum sampled memory peaks. Mean memory CSVs remain supplemental.
- [Targeted diagnostics](terrain_cpu_diagnostics_README.md): paired sampler, PhyDLL logging-barrier, and SmartSim batching treatments.
- [No-batching campaign and memory accounting](terrain_memory_README.md): external same-sweep PSS/RSS sampling and legacy reference evidence.

`terrain_cpu_final_jobs.csv` and `terrain_cpu_diagnostics_jobs.csv` are compact
job manifests, not raw measurements. The solver timing and opt-in memory-log
changes are already in commit `63f3fec`; these workflows require matching built
binaries and the reviewed PhyDLL library in the sibling `CPP-ML-Interface` checkout.
The shared `hh3_render.py` helper is included because the CPU plots import it;
other HH3/GPU/matrix sources are separate work.

## Evidence Retention

Raw CPU evidence remains in `results/terrain_cpu_final/production_4644312/`,
`results/terrain_cpu_diagnostics/`, and the `results/terrain_nobatch*` namespaces.
Derived figures/tables remain in `campaign_figures/terrain_cpu_final_production_4644312/`,
`campaign_figures/nobatch_*`, and `campaign_figures/diagnostics_*`; outer job logs
remain in `logs/`. These local locations are not a verified backup or external archive.

Keep raw HDF5, cubex/Score-P, model/input data, configuration snapshots, launch
commands, environment attestations, logs, and generated PNG/PDF out of ordinary
Git. Preserve the original evidence and arrange a separately verified data archive
before removing any local copy. Full environment snapshots and generated analysis
reports may contain inherited credentials; review/redact them before sharing.
Ignore rules only reduce status noise, do not back up evidence, and do not remove
tracked files or record existing artifact deletions. Do not regenerate validated
campaigns merely to populate Git.

## Offline Checks

Run from `mini_app` with the pinned interpreter; no MPI campaign is launched:

```bash
TMPDIR=../CPP-ML-Interface/tmp/opencode PYTHONDONTWRITEBYTECODE=1 \
  ../CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python -m unittest \
  terrain_memory_sampler_test terrain_cpu_diagnostics_test terrain_cpu_final_test \
  test_scorep_phase_helpers test_scorep_phase_plot test_terrain_nobatch_plot \
  test_terrain_nobatch_validation test_terrain_cpu_final_analysis
```
