#!/bin/bash
#SBATCH --job-name=terrain_diag_analysis
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/terrain_cpu_diagnostics_analysis_%j.log
set -eo pipefail
cd /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh
PYTHON=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python
"${PYTHON}" - <<'PY'
import csv
import os
from pathlib import Path
from terrain_cpu_diagnostics import aggregate, write_csv

with Path('terrain_cpu_diagnostics_jobs.csv').open(newline='') as stream:
    jobs = list(csv.DictReader(stream))
names = [(f'terrain_diag_{"ss" if r["suite"] == "smartsim" else r["suite"]}_{r["model"]}',
          r['job_id']) for r in jobs]
paired, conditions = [], []
for name, job in names:
    root = Path('results/terrain_cpu_diagnostics') / f'{name}_{job}'
    aggregate(root, scorep=True)
    for filename, destination in [('paired_summary', paired), ('conditions', conditions)]:
        with (root / 'analysis' / f'{filename}.csv').open(newline='') as stream:
            destination.extend({**r, 'root': str(root.resolve())} for r in csv.DictReader(stream))
expected = {(m, s, c) for m in ('watercnn', 'giant')
            for s, configs in [('aix', ('aix',)), ('phydll', ('phydll',)),
                               ('smartsim', ('TPQ1_I96', 'TPQ12_I8', 'TPQ96_I1'))]
            for c in configs}
if len(paired) != len(expected) or {(r['model'], r['suite'], r['config']) for r in paired} != expected:
    raise ValueError('Incomplete diagnostic suite coverage')
output = Path('campaign_figures') / f'diagnostics_{os.environ["SLURM_JOB_ID"]}'
output.mkdir(exist_ok=False)
write_csv(output / 'paired_summary.csv', paired)
write_csv(output / 'conditions.csv', conditions)
lines = ['# Controlled CPU Diagnostics', '',
         'Each suite/model compares conditions on one exclusive node. No new full campaign was submitted.',
         'On means PSS sampler enabled (AIx), logging barriers enabled (PhyDLL), or TOTAL_BS/TPQ aggregation (SmartSim).',
         'Replicate means, not individual rank/step observations, are the comparison units.', '',
         '| Model | Suite | Config | Pairs | Off mean (ms) | On mean (ms) | Mean paired delta (ms) | Median on/off ratio |',
         '|---|---|---|---:|---:|---:|---:|---:|']
lookup = {(r['model'], r['suite'], r['config'], r['condition']): r for r in conditions}
for r in paired:
    key = (r['model'], r['suite'], r['config'])
    off = float(lookup[(*key, 'off')]['mean_of_rep_means_ms'])
    on = float(lookup[(*key, 'on')]['mean_of_rep_means_ms'])
    lines.append(f'| {key[0]} | {key[1]} | {key[2]} | {r["pairs"]} | {off:.3f} | {on:.3f} | '
                 f'{float(r["mean_paired_delta_ms"]):.3f} | {float(r["median_paired_ratio"]):.3f} |')
lines += ['', '## Sources', '']
lines.extend(f'- `{r["root"]}/analysis/report.md`' for r in paired)
(output / 'report.md').write_text('\n'.join(lines) + '\n')
print(f'All three diagnostic comparisons completed for both models: {output.resolve()}')
PY
