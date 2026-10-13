#!/bin/zsh
############################
# Super-high-res AIx wrapper (full 3x3) -- HET SURROGATE.
#
# thes2181 (the only account with c23mm) is out of CPU quota, so the CPU-side
# het group runs on a second c23g node under rwth0792 with --gres=none (GPUs not
# allocated; the rank sees GPUS_PER_NODE=0 and gets the CUDA stub via the
# campaign solver wrapper). c23mm and c23g have identical CPU/NUMA topology
# (8x12, 2 sockets, same SLIT), so this is a faithful stand-in.
#
# Target 7680x4320, all layouts, comms coll/p2p1/p2p24, timelines off.
############################
#SBATCH --job-name=matrix5r___CFG__
#SBATCH --account=__ACCOUNT__
#SBATCH --time=__TIME__
#SBATCH --exclusive
#SBATCH --output=logs/mini_app_output_5r_het_%j.txt

############################
# Component 0: CPU-side surrogate (c23g node; GPU requested because c23g
# requires it, then hidden from AIx via CUDA_VISIBLE_DEVICES=NoDevFiles)
############################
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=__HET_MM__
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:__GPUS__
#SBATCH --mem-per-cpu=5G

#SBATCH hetjob

############################
# Component 1: GPU (c23g)
############################
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=__HET_G__
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:__GPUS__
#SBATCH --mem-per-cpu=5G

set -uo pipefail

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}" || exit 1
AIX_ROOT="${MINI_APP_DIR}/../CPP-ML-Interface/extern/AIxeleratorService"
DRIVER="${MINI_APP_DIR}/5r_driver_hetsurrogate.zsh"
REPEAT="${REPEAT_ENV:?REPEAT_ENV must be set (1..5)}"
CFG="${CFG_ENV:?CFG_ENV must be set}"
MANIFEST_DIR="${MINI_APP_DIR}/results/aix_layout_matrix/${CFG}"
MANIFEST="${MANIFEST_DIR}/runs_5r_${CFG}_r${REPEAT}.csv"
mkdir -p "${MANIFEST_DIR}" "${MINI_APP_DIR}/logs"

########## campaign constants ##########
export USE_SCOREP_ENV=0
export SKIP_PAPI_METRICS=1
export SCOREP_METRIC_PAPI=
export MLCOUPLING_MEM_LOG_VERBOSE=0
export TOTAL_STEPS_ENV=202
export SAVE_EVERY_ENV=202
export DB_NODES_ENV=1
export ML_INFERENCE_CPU_CORES_ENV=24
export FORCE_FRESH_RUN_ENV=1
export OVERWRITE_OUTPUT_ENV=1
export SKIP_RENDERING_ENV=1
export USE_LOCAL_RUNTIME_STAGE_ENV=0
export USE_LOCAL_MODEL_CACHE_ENV=1
export USE_SMARTSIM=0
export USE_CPP_ML_INTERFACE=1
export ML_INTERFACE_ENV=cpp
export CPP_ML_INTERFACE_PROVIDER_ENV=AIX
export SOLVER_CPU_BIND_ENV=cores
export AIX_P2P_TIMELINE_ENV=0
export NIC_SAMPLING_ENV=0
export OVERWRITE_JOB_NAME_ENV=1
export APPLY_CUDA_STUBS_ENV=1
export JOB_NAME_ENV="5r_${CFG}_r${REPEAT}"

echo "config: CFG=${CFG} (HET SURROGATE c23g+c23g) HET_G_SIDE_TASKS=${HET_G_SIDE_TASKS_ENV:-auto} GPUS_PER_NODE=${GPUS_PER_NODE_ENV:-auto} TARGET=${TARGET_WIDTH_ENV}x${TARGET_HEIGHT_ENV} MODEL=${MODEL_NAME_ENV} BATCH=${ML_BATCH_SIZE_ENV}"
echo "run_id_cell,jobid,repeat,layout,comm,rc" > "${MANIFEST}"

run_cell() {
  local layout="$1" comm="$2"
  local install build config credits
  case "${layout}" in
    default) install="${AIX_ROOT}/INSTALL";            build="build_het_matrix_default" ;;
    affine)  install="${AIX_ROOT}/INSTALL-AFFINE";     build="build_het_matrix_affine" ;;
    numa)    install="${AIX_ROOT}/INSTALL-NUMA";       build="build_het_matrix_numa_v3" ;;
    *) echo "unknown layout ${layout}"; return 0 ;;
  esac
  case "${comm}" in
    coll)  config="config_aix.toml";           credits="" ;;
    p2p1)  config="config_aix_pipelined.toml"; credits="1" ;;
    p2p24) config="config_aix_pipelined.toml"; credits="24" ;;
    *) echo "unknown comm ${comm}"; return 0 ;;
  esac
  export AIX_INSTALL_PREFIX_ENV="${install}"
  export COMPILE_OUTPUT_PATH_ENV="${build}"
  export SKIP_COMPILE_ENV=1
  export CPP_ML_CONFIG_ENV="${config}"
  if [[ -n "${credits}" ]]; then
    export AIX_P2P_INITIAL_CREDITS_ENV="${credits}"
  else
    unset AIX_P2P_INITIAL_CREDITS_ENV
  fi
  export RUN_ID_ENV="5r_${CFG}_${layout}_${comm}_r${REPEAT}"
  echo "=================================================================="
  echo "[$(date '+%F %T')] cell layout=${layout} comm=${comm} run_id=${RUN_ID_ENV}"
  echo "=================================================================="
  zsh "${DRIVER}"
  local rc=$?
  echo "CELL_RESULT layout=${layout} comm=${comm} repeat=${REPEAT} rc=${rc} run_id=${RUN_ID_ENV}"
  echo "${RUN_ID_ENV},${SLURM_JOB_ID:-NA},${REPEAT},${layout},${comm},${rc}" >> "${MANIFEST}"
  return 0
}

echo "Campaign ${CFG} repeat ${REPEAT} starting on $(hostname) at $(date '+%F %T')"
for layout in default affine numa; do
  for comm in coll p2p1 p2p24; do
    run_cell "${layout}" "${comm}"
  done
done
echo "Campaign ${CFG} repeat ${REPEAT} done at $(date '+%F %T')"
