#!/bin/bash
#SBATCH --job-name=terrain_nb_gate
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/terrain_nobatch_gate_%j.log
set -euo pipefail
MINI_APP_DIR=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
cd "${MINI_APP_DIR}"
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
for command in sbatch mpirun timeout; do command -v "${command}" >/dev/null; done
"${ANALYSIS_PYTHON}" - <<'PY'
import numpy, pandas, matplotlib
import subprocess
from extract_scorep_tables import validate_campaign_coverage
from analyze_cmi_scorep_profiles import find_cube_dump
subprocess.run([find_cube_dump(), '-h'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
PY
ROOT="${VALIDATION_ROOT:?Set VALIDATION_ROOT to the two-model smoke results}"
FIGURES="${MINI_APP_DIR}/campaign_figures/nobatch_validation_${SLURM_JOB_ID}"
"${ANALYSIS_PYTHON}" plot_terrain_nobatch.py --campaign-root "${ROOT}" --steady-steps 2 --output-dir "${FIGURES}/memory_latency"
"${ANALYSIS_PYTHON}" extract_scorep_tables.py --campaign-root "${ROOT}" --steady-steps 2 --output-dir "${FIGURES}/phases"
# Refuse submission unless every model/configuration has exactly one validated profile.
export TERRAIN_GATE_FIGURES="${FIGURES}"
"${ANALYSIS_PYTHON}" - <<'PY'
import csv
import os
from pathlib import Path
from extract_scorep_tables import validate_campaign_coverage
path = Path(os.environ['TERRAIN_GATE_FIGURES'])
with (path / 'memory_latency/memory_latency_replicates.csv').open() as stream:
    rows = list(csv.DictReader(stream))
with (path / 'phases/phase_replicates.csv').open() as stream:
    phases = list(csv.DictReader(stream))
validate_campaign_coverage(rows, phases, reps=(1,), steady_steps=2)
print('Memory, steady latency, and Score-P validation passed for both models and all six configurations.')
PY
"${ANALYSIS_PYTHON}" plot_scorep_phases.py --results-input "${FIGURES}/phases/phase_means.csv" --output-dir "${FIGURES}/phases"
# Remove inherited campaign/runtime and sbatch overrides before exporting canonical values.
for name in $(compgen -e); do
    case "${name}" in
        SBATCH_*|SMARTSIM_*|MLCOUPLING_*|PHYDLL_*|SCOREP_*|OMP_*|MKL_*|OPENBLAS_*|KMP_*|GOMP_*|OMPI_*|PMIX_*) unset "${name}" ;;
    esac
done
unset MODEL REP RUN_ROOT LOG_ROOT SSDB CMI_DIR CPP_ML_DIR CAMPAIGN_ROOT
export MODELS="watercnn giant" REPS=5 STEPS=22 MPI_RANKS=96 RANK_GRID_X=12 RANK_GRID_Z=8
export CONFIGS="AIX PHYDLL_CPP SS_DEFAULT SS_TPQ12_I8_B96 SS_TPQ96_I1_B96 SS_TPQ1_I96_B96"
export SOLVER_BIN="${MINI_APP_DIR}/solver_cpp/build_scorep_cmi_fresh/terrain_solver"
export SMARTSIM_PYTHON="${ANALYSIS_PYTHON}" MONITOR_PYTHON="${ANALYSIS_PYTHON}"
export PHYDLL_DL_CLIENT=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/dl_clients/build-scorep/phydll_dl_client
export MEMORY_INTERVAL=1.0 RUN_TIMEOUT=10800 DB_START_TIMEOUT=180 SLURM_EXPORT_ENV=ALL
export OUTPUT_BASE="${MINI_APP_DIR}/results/terrain_nobatch/validated_${SLURM_JOB_ID}"
for executable in "${SOLVER_BIN}" "${PHYDLL_DL_CLIENT}" "${SMARTSIM_PYTHON}"; do
    [[ -x "${executable}" ]] || { printf 'Missing executable: %s\n' "${executable}" >&2; exit 1; }
done
job=$(sbatch --parsable --array=0-9%2 --job-name=terrain_nobatch_full \
    --export=ALL \
    campaign_cpu_terrain.sbatch)
export CAMPAIGN_ROOT="${OUTPUT_BASE}"
analysis=$(sbatch --parsable --dependency="afterok:${job%%;*}" \
    --export=ALL analyze_terrain_nobatch.sbatch)
printf '%s\n' "Production array=${job} analysis=${analysis} output=${OUTPUT_BASE}" | tee "${FIGURES}/production_submission.txt"
