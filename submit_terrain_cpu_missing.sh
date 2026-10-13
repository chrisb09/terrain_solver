#!/bin/bash
# Dry-run by default. Three smoke arrays gate three production arrays; no builds.
set -eo pipefail
mini=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
cmi=${mini%/mini_app}/CPP-ML-Interface
campaign=${TERRAIN_MISSING_CAMPAIGN:-terrain_missing_20261012}
action=${1:---dry-run}
[[ $# -le 1 && $campaign =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
case $action in --dry-run|--submit|--test-only) ;; *) exit 1 ;; esac
[[ $PWD == "$mini" && -d $mini/logs ]] || exit 1
while read -r name; do case $name in SBATCH_*) unset "$name" ;; esac; done < <(compgen -e)
bash "$mini/terrain_cpu_plain_gate.sh" --preflight
# Check the original instrumented memory binaries under the original environment.
(
    unset PYTHONHOME PYTHONPATH PYTHONOPTIMIZE LD_PRELOAD USE_SCOREP
    source "${mini%/mini_app}/set_env_claix23_cuda12.4.sh"
    export CMI_DIR=$cmi SMARTSIM_SCOREP_ROOT=$cmi/tmp/opencode/scorep-8.4-papi72-install
    export SMARTSIM_PAPI_ROOT=$cmi/tmp/opencode/papi-7.2.0-install
    source "$cmi/env_scorep.sh"
    export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64:${LD_LIBRARY_PATH:-}"
    "$cmi/extern/python/smartsim_cuda-12/bin/python" - "$mini" "$cmi" <<'PY'
import hashlib, re, subprocess, sys
from pathlib import Path
mini, cmi = map(Path, sys.argv[1:])
libs = []
for binary in (mini / 'solver_cpp/build_scorep_cmi_fresh/terrain_solver', cmi / 'dl_clients/build-scorep/phydll_dl_client'):
    text = subprocess.check_output(['ldd', '-r', str(binary)], stderr=subprocess.STDOUT, text=True)
    if 'not found' in text or 'undefined symbol' in text or 'libscorep' not in text:
        raise ValueError(f'Invalid memory binary linkage: {binary}')
    matches = re.findall(r'^\s*libphydll\.so\s+=>\s+(\S+)', text, re.M)
    if len(matches) != 1:
        raise ValueError('Missing/ambiguous loaded C library')
    libs.append(Path(matches[0]).resolve())
if libs[0] != libs[1] or hashlib.sha256(libs[0].read_bytes()).hexdigest() != 'dd551169e436d07e367fc5304389eafba3999f43587a20af0ec0fc8b8c87f553':
    raise ValueError('Original memory library hash/matching gate failed')
for name in ('prep_watercnn_4k.h5', 'prep_giant_240x192.h5'):
    if not (mini / 'input' / name).is_file():
        raise ValueError(f'Missing input: {name}')
for model in ('watercnn', 'giant'):
    for suffix in ('', '_split'):
        if not (mini / 'train_models/model_a' / f'{model}{suffix}_cpu.pt').is_file():
            raise ValueError('Missing CPU model')
print('Validated original Score-P memory binaries, library hash, and both model inputs.')
PY
)
queued=$(squeue -u "$USER" --noheader --format='%j')
[[ $queued != *"${campaign}_"* ]] || { printf 'Duplicate queued campaign: %s\n' "$campaign" >&2; exit 1; }
manifest=$mini/terrain_cpu_missing_${campaign}_jobs.csv
[[ ! -e $manifest ]] || { printf 'Already submitted: %s\n' "$manifest" >&2; exit 1; }
for dataset in "${campaign}_other" "${campaign}_readiness_false" "${campaign}_readiness_true" "${campaign}_smoke_other" "${campaign}_smoke_readiness_false" "${campaign}_smoke_readiness_true"; do
    [[ ! -e $mini/results/terrain_cpu_final/$dataset ]] || exit 1
done
submit() {
    local stage=$1 mode=$2 group=$3 array=$4 steps=$5 dependency=${6:-} dataset=$campaign id
    [[ $stage != smoke ]] || dataset=${campaign}_smoke
    local -a command=(sbatch --parsable --job-name="${campaign}_${stage}_${mode}_${group}"
        --array="$array" --export="MODE=$mode,TERRAIN_GROUP=$group,STEPS=$steps,TERRAIN_MISSING_CAMPAIGN=$dataset")
    [[ -z $dependency ]] || command+=(--dependency="afterok:$dependency")
    if [[ $action == --submit ]]; then
        id=$("${command[@]}" "$mini/terrain_cpu_missing.sbatch")
        id=${id%%;*}
        [[ $id =~ ^[0-9]+$ ]] || exit 1
        printf '%s,%s,%s,%s,"%s",%s,%s,%s\n' "$id" "$stage" "$mode" "$group" "$array" "$steps" "$dependency" "$dataset" >> "$manifest"
        printf '%s\n' "$id"
    elif [[ $action == --test-only ]]; then
        # Scheduler resource/array check only; dependencies are not real jobs yet.
        sbatch --test-only --job-name="${campaign}_${stage}_${mode}_${group}" --array="$array" \
            --export="MODE=$mode,TERRAIN_GROUP=$group,STEPS=$steps,TERRAIN_MISSING_CAMPAIGN=$dataset" "$mini/terrain_cpu_missing.sbatch" >&2
        printf 'SMOKE_%s_%s\n' "$mode" "$group"
    else
        printf '%q ' "${command[@]}" "$mini/terrain_cpu_missing.sbatch" >&2
        printf '\n' >&2
        printf 'SMOKE_%s_%s\n' "$mode" "$group"
    fi
}
if [[ $action == --submit ]]; then
    (set -o noclobber; printf 'job_id,stage,mode,group,array,steps,afterok,campaign\n' > "$manifest")
fi
memory_smoke=$(submit smoke memory other '0,5%2' 6)
other_smoke=$(submit smoke timing other '0,5%2' 6)
phy_smoke=$(submit smoke timing phydll '0,5,10,15%2' 6)
memory=$(submit production memory other '0-9%2' 6 "$memory_smoke")
other=$(submit production timing other '0-9%2' 22 "$other_smoke")
phy=$(submit production timing phydll '0-19%2' 22 "$phy_smoke")
printf 'campaign=%s action=%s memory=%s timing_other=%s timing_phydll=%s\n' "$campaign" "$action" "$memory" "$other" "$phy"
