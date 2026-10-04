#!/bin/bash
#SBATCH --job-name=terrain_nb_analysis
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/terrain_nobatch_analysis_%j.log
set -euo pipefail
cd /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
unset PYTHONHOME PYTHONPATH PYTHONOPTIMIZE LD_PRELOAD USE_SCOREP
set +u
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh
set -u
: "${LD_LIBRARY_PATH:?Missing library path}"
for name in EBROOTOPENSSL EBROOTCLANG EBROOTGCCCORE EBROOTOPENMPI EBROOTHDF5 EBROOTPNETCDF EBROOTCUDA EBROOTCUDNN EBROOTIMKL EBROOTHWLOC; do
    root="${!name:-}"
    [[ -n "${root}" && -d "${root}" ]] || { printf 'Missing module root: %s\n' "${name}" >&2; exit 1; }
done
ANALYSIS_PYTHON=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3
[[ -x "${ANALYSIS_PYTHON}" ]] || exit 1
for command in mpirun timeout; do command -v "${command}" >/dev/null; done
"${ANALYSIS_PYTHON}" - <<'PY'
import numpy, pandas, matplotlib
import subprocess
from extract_scorep_tables import validate_campaign_coverage
from analyze_cmi_scorep_profiles import find_cube_dump
subprocess.run([find_cube_dump(), '-h'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
PY
ROOT="${CAMPAIGN_ROOT:?Set CAMPAIGN_ROOT}"
FIGURES="${PWD}/campaign_figures/nobatch_${SLURM_JOB_ID}"
"${ANALYSIS_PYTHON}" plot_terrain_nobatch.py --campaign-root "${ROOT}" --steady-steps 10 --output-dir "${FIGURES}/memory_latency"
"${ANALYSIS_PYTHON}" extract_scorep_tables.py --campaign-root "${ROOT}" --steady-steps 10 --output-dir "${FIGURES}/phases"
export TERRAIN_ANALYSIS_FIGURES="${FIGURES}"
"${ANALYSIS_PYTHON}" - <<'PY'
import csv
import os
from pathlib import Path
from extract_scorep_tables import validate_campaign_coverage
path = Path(os.environ['TERRAIN_ANALYSIS_FIGURES'])
with (path / 'memory_latency/memory_latency_replicates.csv').open() as stream:
    rows = list(csv.DictReader(stream))
with (path / 'phases/phase_replicates.csv').open() as stream:
    phases = list(csv.DictReader(stream))
validate_campaign_coverage(rows, phases, reps=range(1, 6), steady_steps=10)
print(f'Validated 60 configuration runs. Figures and tables: {path}')
PY
"${ANALYSIS_PYTHON}" plot_scorep_phases.py --results-input "${FIGURES}/phases/phase_means.csv" --output-dir "${FIGURES}/phases"
