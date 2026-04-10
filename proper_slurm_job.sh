#!/bin/zsh


############################
# Global job options
############################
#SBATCH --job-name=terrain_solver_coupled
#SBATCH --account=thes2181
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=logs/mini_app_output_%j.txt

############################
# Component 0: CPU (c23mm)
############################
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G

#SBATCH hetjob

############################
# Component 1a: GPU (c23g)
############################
#SBATCH --partition=c23g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G

#############################
# Component 1b: CPU (c23mm) for ML inference if not using GPU
#############################
##SBATCH --partition=c23mm
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=72
##SBATCH --mem-per-cpu=5G



DB_HOSTNAME_FILE="db_hostname_${SLURM_JOB_ID}.txt"
#SOLVER_STEP_LOG="logs/mini_app_solver_steps_${SLURM_JOB_ID}.log"

# instead, lets use logs/mini_app_output_%j.txt as the SOLVER_STEP_LOG since it will contain both the solver output and the SmartSim controller output
SOLVER_STEP_LOG="logs/mini_app_output_${SLURM_JOB_ID}.txt"


########## SmartSim Timout Envs ##########

export SR_MODEL_TIMEOUT=300000
export SR_CMD_TIMEOUT=300000
export SR_SOCKET_TIMEOUT=300000

########## SmartSim/ML Parameters ##########

USE_SMARTSIM=1
SOLVER_HET_GROUP=0
DB_HET_GROUP=1

#MODEL_PATH="train_models/model_a/real_function_jit.pt"
#MODEL_PATH="train_models/model_a/best_model_jit_transformer_mlp.pt"
#MODEL_PATH="train_models/model_a/best_model_jit_watercnn.pt"
MODEL_PATH="train_models/model_a/best_model_jit_benchmark_giant_mlp.pt"
MODEL_BACKEND="TORCH"
ML_BATCH_SIZE=50000
MODEL_STAGE_MAX_RETRIES=2
MODEL_STAGE_FALLBACK_TO_SHARED=1
MODEL_STAGE_DB_GROUP=1
DB_NODE_PREFLIGHT=1
MODEL_ARTIFACT_MANIFEST="train_models/model_a/artifact_manifest_benchmark_giant_mlp.json"
#MODEL_ARTIFACT_MANIFEST="train_models/model_a/artifact_manifest_perfect_model.json"
MODEL_INPUTS=""
MODEL_OUTPUTS=""
MODEL_PATH_SOURCE=""
MODEL_PATH_FOR_SOLVER=""
USE_LOCAL_MODEL_CACHE=1 # if 1, we copy the model to /tmp on the component 1 nodes, to bypass the shared filesystem which is slow for model loading at high GPU/node counts, and set MODEL_PATH_FOR_SOLVER to the local path; if 0, we use the original MODEL_PATH_SOURCE which is on the shared filesystem and can be accessed by all nodes but may have slower load times at high GPU/node counts
TORCH_CPU_MODEL_CONVERT=1 # if 1 and running CPU+TORCH, rewrite TorchScript model with map_location=cpu before staging/loading
MODEL_STAGE_LOG=""
MODEL_STAGE_DURATION_SOLVER=0
MODEL_STAGE_DURATION_DB=0
MODEL_STAGE_TOTAL_DURATION=0
LOCAL_FAST_ROOT=""
CONTROLLER_START_MAX_RETRIES=2
SMARTSIM_RUNTIME_ROOT="/home/thes2181/python"
USE_LOCAL_RUNTIME_STAGE=0
RUNTIME_STAGE_MAX_RETRIES=2
RUNTIME_STAGE_LOG=""
RUNTIME_STAGE_DURATION=0

#MODEL_NAME="perfect_model"
#MODEL_NAME="transformer_mlp"
#MODEL_NAME="watercnn_a"
MODEL_NAME="benchmark_giant_mlp"

# Derive DB/ML settings from Slurm het-group component selected by DB_HET_GROUP.
_db_nodes_var="SLURM_JOB_NUM_NODES_HET_GROUP_${DB_HET_GROUP}"
_db_gpus_per_node_var="SLURM_GPUS_PER_NODE_HET_GROUP_${DB_HET_GROUP}"
_db_gpus_per_task_var="SLURM_GPUS_PER_TASK_HET_GROUP_${DB_HET_GROUP}"
_db_ntasks_per_node_var="SLURM_NTASKS_PER_NODE_HET_GROUP_${DB_HET_GROUP}"
_db_cpus_per_task_var="SLURM_CPUS_PER_TASK_HET_GROUP_${DB_HET_GROUP}"
_db_gpus_on_node_var="SLURM_GPUS_ON_NODE_HET_GROUP_${DB_HET_GROUP}"
_db_tres_per_task_var="SLURM_TRES_PER_TASK_HET_GROUP_${DB_HET_GROUP}"
_db_tres_per_node_var="SLURM_TRES_PER_NODE_HET_GROUP_${DB_HET_GROUP}"

DB_NODES="${(P)_db_nodes_var:-${SLURM_JOB_NUM_NODES:-1}}"
ML_INFERENCE_CPU_CORES="${(P)_db_cpus_per_task_var:-${SLURM_CPUS_PER_TASK:-1}}"

_gpus_per_node_raw="${(P)_db_gpus_per_node_var:-${SLURM_GPUS_PER_NODE:-}}"
_gpus_per_task_raw="${(P)_db_gpus_per_task_var:-${SLURM_GPUS_PER_TASK:-}}"
_db_ntasks_per_node_raw="${(P)_db_ntasks_per_node_var:-${SLURM_NTASKS_PER_NODE:-1}}"
_gpus_on_node_raw="${(P)_db_gpus_on_node_var:-${SLURM_GPUS_ON_NODE:-}}"
_db_tres_per_task_raw="${(P)_db_tres_per_task_var:-${SLURM_TRES_PER_TASK:-}}"
_db_tres_per_node_raw="${(P)_db_tres_per_node_var:-${SLURM_TRES_PER_NODE:-}}"

_gpus_per_node_raw="${_gpus_per_node_raw%%\(*}"
_gpus_per_node_raw="${_gpus_per_node_raw%%,*}"
if [[ "${_gpus_per_node_raw}" == *":"* ]]; then
  _gpus_per_node_raw="${_gpus_per_node_raw##*:}"
fi

_gpus_per_task_raw="${_gpus_per_task_raw%%\(*}"
_gpus_per_task_raw="${_gpus_per_task_raw%%,*}"
if [[ "${_gpus_per_task_raw}" == *":"* ]]; then
  _gpus_per_task_raw="${_gpus_per_task_raw##*:}"
fi

_db_ntasks_per_node_raw="${_db_ntasks_per_node_raw%%\(*}"
_db_ntasks_per_node_raw="${_db_ntasks_per_node_raw%%,*}"

_gpus_on_node_raw="${_gpus_on_node_raw%%\(*}"
_gpus_on_node_raw="${_gpus_on_node_raw%%,*}"

if [[ "${_gpus_per_task_raw}" != <-> ]] && [[ "${_db_tres_per_task_raw}" =~ 'gres/gpu[:=]([0-9]+)' ]]; then
  _gpus_per_task_raw="${match[1]}"
fi
if [[ "${_gpus_per_node_raw}" != <-> ]] && [[ "${_db_tres_per_node_raw}" =~ 'gres/gpu[:=]([0-9]+)' ]]; then
  _gpus_per_node_raw="${match[1]}"
fi

if [[ "${_gpus_per_task_raw}" != <-> ]] && [[ "${_gpus_per_node_raw}" != <-> ]] && command -v scontrol >/dev/null 2>&1; then
  _db_component_job_id="${SLURM_JOB_ID}+${DB_HET_GROUP}"
  _db_scontrol_job="$(scontrol show job "${_db_component_job_id}" 2>/dev/null || true)"
  if [[ -n "${_db_scontrol_job}" ]]; then
    if [[ "${_gpus_per_task_raw}" != <-> ]] && [[ "${_db_scontrol_job}" =~ 'TresPerTask=[^[:space:]]*gres/gpu[:=]([0-9]+)' ]]; then
      _gpus_per_task_raw="${match[1]}"
    fi
    if [[ "${_gpus_per_node_raw}" != <-> ]] && [[ "${_db_scontrol_job}" =~ 'TresPerNode=[^[:space:]]*gres/gpu[:=]([0-9]+)' ]]; then
      _gpus_per_node_raw="${match[1]}"
    fi
    if [[ "${_gpus_on_node_raw}" != <-> ]] && [[ "${_db_scontrol_job}" =~ 'gres/gpu=([0-9]+)' ]]; then
      _gpus_on_node_raw="${match[1]}"
    fi
  fi
fi

if [[ "${_gpus_per_node_raw}" == <-> ]]; then
  GPUS_PER_NODE="${_gpus_per_node_raw}"
elif [[ "${_gpus_per_task_raw}" == <-> ]] && [[ "${_db_ntasks_per_node_raw}" == <-> ]]; then
  GPUS_PER_NODE=$(( _gpus_per_task_raw * _db_ntasks_per_node_raw ))
elif [[ "${_gpus_on_node_raw}" == <-> ]]; then
  GPUS_PER_NODE="${_gpus_on_node_raw}"
else
  GPUS_PER_NODE=0
fi

if [[ "${DB_NODES}" != <-> ]] || [[ "${DB_NODES}" -lt 1 ]]; then
  DB_NODES=1
fi
if [[ "${ML_INFERENCE_CPU_CORES}" != <-> ]] || [[ "${ML_INFERENCE_CPU_CORES}" -lt 1 ]]; then
  ML_INFERENCE_CPU_CORES=1
fi
if [[ "${GPUS_PER_NODE}" != <-> ]] || [[ "${GPUS_PER_NODE}" -lt 0 ]]; then
  GPUS_PER_NODE=0
fi

TOTAL_GPU_COUNT=0

if (( GPUS_PER_NODE > 0 )); then
  TOTAL_GPU_COUNT=$(( GPUS_PER_NODE * DB_NODES ))
fi

echo "DB_HET_GROUP=${DB_HET_GROUP} Slurm raw vars: ${_db_gpus_per_node_var}='${(P)_db_gpus_per_node_var:-${SLURM_GPUS_PER_NODE:-}}', ${_db_gpus_per_task_var}='${(P)_db_gpus_per_task_var:-${SLURM_GPUS_PER_TASK:-}}', ${_db_tres_per_task_var}='${(P)_db_tres_per_task_var:-${SLURM_TRES_PER_TASK:-}}', ${_db_ntasks_per_node_var}='${(P)_db_ntasks_per_node_var:-${SLURM_NTASKS_PER_NODE:-}}', ${_db_gpus_on_node_var}='${(P)_db_gpus_on_node_var:-${SLURM_GPUS_ON_NODE:-}}' -> parsed: gpus_per_task=${_gpus_per_task_raw}, gpus_per_node=${_gpus_per_node_raw}, gpus_on_node=${_gpus_on_node_raw}; computed GPUS_PER_NODE=${GPUS_PER_NODE}, DB_NODES=${DB_NODES}, ML_INFERENCE_CPU_CORES=${ML_INFERENCE_CPU_CORES}; TOTAL_GPU_COUNT=${TOTAL_GPU_COUNT}"

USE_GPU=$(( GPUS_PER_NODE > 0 ? 1 : 0 ))

if (( USE_SMARTSIM == 1 )); then
  if (( USE_GPU == 1 )); then
    echo "Configuring for GPU-based ML inference with ${GPUS_PER_NODE} GPUs per node."
    export CUSTOM_JOB_NAME_SUFFIX_ENV="_${GPUS_PER_NODE}gpu_${MODEL_NAME}_revamped_prepare"
  else
    echo "Configuring for CPU-based ML inference with ${ML_INFERENCE_CPU_CORES} CPU cores per task."
    export CUSTOM_JOB_NAME_SUFFIX_ENV="_${ML_INFERENCE_CPU_CORES}cpu_${MODEL_NAME}_revamped_prepare"
  fi
fi


export OVERWRITE_JOB_NAME_ENV=1



########### Solver Specific Parameters ############


CUSTOM_JOB_NAME_SUFFIX="" # optional suffix to add to the job name for easier identification in job queues and output files, e.g. "_test" or "_render_only"

job_name_template='circle_r${RADIUS}_d${INIT_DEPTH}_s${TOTAL_STEPS}_${SAVE_MODE}_${TARGET_WIDTH}x${TARGET_HEIGHT}_${_partition}_${_nodes}n_${_ntasks_per_node}t_${_cpus_per_task}c-${DB_NODES}n_${DB}__${IO_MODE}_${MPI_SYNC_MODE}${CUSTOM_JOB_NAME_SUFFIX}'


INIT_MODE="circle" # circle, square, uniform
RADIUS=500 # radius for circle and square initial conditions
INIT_DEPTH=100 # initial depth of water
#TOTAL_STEPS=10000000
TOTAL_STEPS=10
#TOTAL_STEPS=10000
SAVE_EVERY=1
#SAVE_MODE="triangular"        # periodic, triangular
SAVE_MODE="periodic"          # periodic, triangular
TRIANGULAR_SCALE=1          # scale factor for triangular save schedule (>=1, only used when SAVE_MODE=triangular)
#TARGET_WIDTH=21600
#TARGET_HEIGHT=10800
TARGET_WIDTH=8640
TARGET_HEIGHT=4320
#TARGET_WIDTH=2160
#TARGET_HEIGHT=1080
#TARGET_WIDTH=216
#TARGET_HEIGHT=108
CHUNK_SIZE=60
#CHUNK_SIZE=36
#CHUNK_SIZE=18
# We also include some slurm parameters as part of the job name so running different configurations in parallel is easier to manage.
_partition="${SLURM_JOB_PARTITION:-}"
_solver_nodes_var="SLURM_JOB_NUM_NODES_HET_GROUP_${SOLVER_HET_GROUP}"
_solver_ntasks_var="SLURM_NTASKS_HET_GROUP_${SOLVER_HET_GROUP}"
_solver_ntasks_per_node_var="SLURM_NTASKS_PER_NODE_HET_GROUP_${SOLVER_HET_GROUP}"

_nodes="${(P)_solver_nodes_var:-${SLURM_JOB_NUM_NODES:-}}"
_ntasks_per_node="${(P)_solver_ntasks_per_node_var:-${SLURM_NTASKS_PER_NODE:-}}"
_cpus_per_task="${SLURM_CPUS_PER_TASK:-}"

# SLURM_NTASKS_PER_NODE can be formatted like "24(x2)" or comma-separated in hetjobs.
_ntasks_per_node_num="${_ntasks_per_node%%\(*}"
_ntasks_per_node_num="${_ntasks_per_node_num%%,*}"
if [[ -z "${_ntasks_per_node_num}" ]]; then
  _ntasks_per_node_num=1
fi

# Prefer Slurm's already-computed task count for the selected solver het-group.
if [[ -n "${(P)_solver_ntasks_var:-}" ]]; then
  MPI_RANKS="${(P)_solver_ntasks_var}"
elif [[ -n "${SLURM_NTASKS:-}" ]] && [[ -z "${SLURM_HET_SIZE:-}" ]]; then
  MPI_RANKS="${SLURM_NTASKS}"
else
  MPI_RANKS=$(( ${_nodes:-1} * ${_ntasks_per_node_num} ))
fi
echo "Calculated MPI_RANKS=${MPI_RANKS} from _nodes=${_nodes}, _ntasks_per_node=${_ntasks_per_node}, SLURM_NTASKS=${SLURM_NTASKS}, and _solver_ntasks_var=${(P)_solver_ntasks_var:-}"
IO_MODE="rank0_gather"     # parallel_hdf5, rank0_gather
MPI_SYNC_MODE="none"        # none, step, report
HDF5_XFER_MODE="independent" # collective, independent (independent is more robust on this stack at high ranks)

#JOB_NAME="${INIT_MODE}_r${RADIUS}_d${INIT_DEPTH}_s${TOTAL_STEPS}_${SAVE_MODE}_${TARGET_WIDTH}x${TARGET_HEIGHT}_${_partition}_${_nodes}n_${_ntasks_per_node}t_${_cpus_per_task}c__${IO_MODE}_${MPI_SYNC_MODE}"
JOB_NAME=$(eval echo "$job_name_template")
if [[ "${TRIANGULAR_SCALE}" -gt 1 ]]; then
  JOB_NAME="${JOB_NAME}_ts${TRIANGULAR_SCALE}"
fi
echo "Using preliminary job name: ${JOB_NAME}"

RANK_GRID_X=0               # 0 = auto
RANK_GRID_Z=0               # 0 = auto
OVERWRITE_OUTPUT=1          # 1 = pass --overwrite-output

SKIP_COMPILE=0
SKIP_RENDERING=0

#VIDEO_FPS=60
VIDEO_FPS=1
#REDUCED_HEIGHT=135 # ensure that TARGET_WIDTH / (TARGET_HEIGHT + REDUCED_HEIGHT) is close to 16:9 aspect ratio for the combined video
REDUCED_HEIGHT=$((TARGET_WIDTH * 9 / 16 - TARGET_HEIGHT)) # automatically calculate the reduced height needed to achieve 16:9 aspect ratio in the final video when combined with the original TARGET_HEIGHT, overrides the default of 135 if set
echo "Calculated REDUCED_HEIGHT=${REDUCED_HEIGHT} to achieve 16:9 aspect ratio in final video with TARGET_WIDTH=${TARGET_WIDTH} and TARGET_HEIGHT=${TARGET_HEIGHT}"
HEIGHT_MIN=40 # minimum height for the slice view, adjust based on expected water height range to ensure good visibility of water in the slice view

# Check Environment Variables, if they are set, overwrite the above defaults
if [[ -n "${JOB_NAME_ENV:-}" ]]; then
  JOB_NAME="${JOB_NAME_ENV}"
  echo "Using JOB_NAME from environment variable: ${JOB_NAME}"
fi
if [[ -n "${CUSTOM_JOB_NAME_SUFFIX_ENV:-}" ]]; then
  CUSTOM_JOB_NAME_SUFFIX="${CUSTOM_JOB_NAME_SUFFIX_ENV}"
  echo "Using CUSTOM_JOB_NAME_SUFFIX from environment variable: ${CUSTOM_JOB_NAME_SUFFIX}"
fi
if [[ -n "${INIT_MODE_ENV:-}" ]]; then
  INIT_MODE="${INIT_MODE_ENV}"
  echo "Using INIT_MODE from environment variable: ${INIT_MODE}"
fi
if [[ -n "${MODEL_PATH_ENV:-}" ]]; then
  MODEL_PATH="${MODEL_PATH_ENV}"
  echo "Using MODEL_PATH from environment variable: ${MODEL_PATH}"
fi
if [[ -n "${MODEL_BACKEND_ENV:-}" ]]; then
  MODEL_BACKEND="${MODEL_BACKEND_ENV}"
  echo "Using MODEL_BACKEND from environment variable: ${MODEL_BACKEND}"
fi
if [[ -n "${MODEL_ARTIFACT_MANIFEST_ENV:-}" ]]; then
  MODEL_ARTIFACT_MANIFEST="${MODEL_ARTIFACT_MANIFEST_ENV}"
  echo "Using MODEL_ARTIFACT_MANIFEST from environment variable: ${MODEL_ARTIFACT_MANIFEST}"
fi
if [[ -n "${MODEL_INPUTS_ENV:-}" ]]; then
  MODEL_INPUTS="${MODEL_INPUTS_ENV}"
  echo "Using MODEL_INPUTS from environment variable: ${MODEL_INPUTS}"
fi
if [[ -n "${MODEL_OUTPUTS_ENV:-}" ]]; then
  MODEL_OUTPUTS="${MODEL_OUTPUTS_ENV}"
  echo "Using MODEL_OUTPUTS from environment variable: ${MODEL_OUTPUTS}"
fi
if [[ -n "${RADIUS_ENV:-}" ]]; then
  RADIUS="${RADIUS_ENV}"
  echo "Using RADIUS from environment variable: ${RADIUS}"
fi
if [[ -n "${INIT_DEPTH_ENV:-}" ]]; then
  INIT_DEPTH="${INIT_DEPTH_ENV}"
  echo "Using INIT_DEPTH from environment variable: ${INIT_DEPTH}"
fi
if [[ -n "${TOTAL_STEPS_ENV:-}" ]]; then
  TOTAL_STEPS="${TOTAL_STEPS_ENV}"
  echo "Using TOTAL_STEPS from environment variable: ${TOTAL_STEPS}"
fi
if [[ -n "${SAVE_EVERY_ENV:-}" ]]; then
  SAVE_EVERY="${SAVE_EVERY_ENV}"
  echo "Using SAVE_EVERY from environment variable: ${SAVE_EVERY}"
fi
if [[ -n "${SAVE_MODE_ENV:-}" ]]; then
  SAVE_MODE="${SAVE_MODE_ENV}"
  echo "Using SAVE_MODE from environment variable: ${SAVE_MODE}"
fi
if [[ -n "${TRIANGULAR_SCALE_ENV:-}" ]]; then
  TRIANGULAR_SCALE="${TRIANGULAR_SCALE_ENV}"
  echo "Using TRIANGULAR_SCALE from environment variable: ${TRIANGULAR_SCALE}"
fi
if [[ -n "${TARGET_WIDTH_ENV:-}" ]]; then
  TARGET_WIDTH="${TARGET_WIDTH_ENV}"
  echo "Using TARGET_WIDTH from environment variable: ${TARGET_WIDTH}"
fi
if [[ -n "${TARGET_HEIGHT_ENV:-}" ]]; then
  TARGET_HEIGHT="${TARGET_HEIGHT_ENV}"
  echo "Using TARGET_HEIGHT from environment variable: ${TARGET_HEIGHT}"
fi
if [[ -n "${CHUNK_SIZE_ENV:-}" ]]; then
  CHUNK_SIZE="${CHUNK_SIZE_ENV}"
  echo "Using CHUNK_SIZE from environment variable: ${CHUNK_SIZE}"
fi
if [[ -n "${MPI_RANKS_ENV:-}" ]]; then
  MPI_RANKS="${MPI_RANKS_ENV}"
  echo "Using MPI_RANKS from environment variable: ${MPI_RANKS}"
fi
if [[ -n "${IO_MODE_ENV:-}" ]]; then
  IO_MODE="${IO_MODE_ENV}"
  echo "Using IO_MODE from environment variable: ${IO_MODE}"
fi
if [[ -n "${MPI_SYNC_MODE_ENV:-}" ]]; then
  MPI_SYNC_MODE="${MPI_SYNC_MODE_ENV}"
  echo "Using MPI_SYNC_MODE from environment variable: ${MPI_SYNC_MODE}"
fi
if [[ -n "${HDF5_XFER_MODE_ENV:-}" ]]; then
  HDF5_XFER_MODE="${HDF5_XFER_MODE_ENV}"
  echo "Using HDF5_XFER_MODE from environment variable: ${HDF5_XFER_MODE}"
fi
if [[ -n "${RANK_GRID_X_ENV:-}" ]]; then
  RANK_GRID_X="${RANK_GRID_X_ENV}"
  echo "Using RANK_GRID_X from environment variable: ${RANK_GRID_X}"
fi
if [[ -n "${RANK_GRID_Z_ENV:-}" ]]; then
  RANK_GRID_Z="${RANK_GRID_Z_ENV}"
  echo "Using RANK_GRID_Z from environment variable: ${RANK_GRID_Z}"
fi
if [[ -n "${OVERWRITE_OUTPUT_ENV:-}" ]]; then
  OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT_ENV}"
  echo "Using OVERWRITE_OUTPUT from environment variable: ${OVERWRITE_OUTPUT}"
fi
if [[ -n "${SKIP_RENDERING_ENV:-}" ]]; then
  SKIP_RENDERING="${SKIP_RENDERING_ENV}"
  echo "Using SKIP_RENDERING from environment variable: ${SKIP_RENDERING}"
fi
if [[ -n "${SKIP_COMPILE_ENV:-}" ]]; then
  SKIP_COMPILE="${SKIP_COMPILE_ENV}"
  echo "Using SKIP_COMPILE from environment variable: ${SKIP_COMPILE}"
fi
if [[ -n "${FFMPEG_THREADS_ENV:-}" ]]; then
  FFMPEG_THREADS="${FFMPEG_THREADS_ENV}"
  echo "Using FFMPEG_THREADS from environment variable: ${FFMPEG_THREADS}"
fi
if [[ -n "${VIDEO_FPS_ENV:-}" ]]; then
  VIDEO_FPS="${VIDEO_FPS_ENV}"
  echo "Using VIDEO_FPS from environment variable: ${VIDEO_FPS}"
fi
if [[ -n "${RENDER_CPUS_ENV:-}" ]]; then
  RENDER_CPUS="${RENDER_CPUS_ENV}"
  echo "Using RENDER_CPUS from environment variable: ${RENDER_CPUS}"
fi
if [[ -n "${REDUCED_HEIGHT_ENV:-}" ]]; then
  REDUCED_HEIGHT="${REDUCED_HEIGHT_ENV}"
  echo "Using REDUCED_HEIGHT from environment variable: ${REDUCED_HEIGHT}"
fi
if [[ -n "${HEIGHT_MIN_ENV:-}" ]]; then
  HEIGHT_MIN="${HEIGHT_MIN_ENV}"
  echo "Using HEIGHT_MIN from environment variable: ${HEIGHT_MIN}"
fi
if [[ -n "${USE_LOCAL_MODEL_CACHE_ENV:-}" ]]; then
  USE_LOCAL_MODEL_CACHE="${USE_LOCAL_MODEL_CACHE_ENV}"
  echo "Using USE_LOCAL_MODEL_CACHE from environment variable: ${USE_LOCAL_MODEL_CACHE}"
fi
if [[ -n "${TORCH_CPU_MODEL_CONVERT_ENV:-}" ]]; then
  TORCH_CPU_MODEL_CONVERT="${TORCH_CPU_MODEL_CONVERT_ENV}"
  echo "Using TORCH_CPU_MODEL_CONVERT from environment variable: ${TORCH_CPU_MODEL_CONVERT}"
fi
if [[ -n "${SMARTSIM_RUNTIME_ROOT_ENV:-}" ]]; then
  SMARTSIM_RUNTIME_ROOT="${SMARTSIM_RUNTIME_ROOT_ENV}"
  echo "Using SMARTSIM_RUNTIME_ROOT from environment variable: ${SMARTSIM_RUNTIME_ROOT}"
fi
if [[ -n "${USE_LOCAL_RUNTIME_STAGE_ENV:-}" ]]; then
  USE_LOCAL_RUNTIME_STAGE="${USE_LOCAL_RUNTIME_STAGE_ENV}"
  echo "Using USE_LOCAL_RUNTIME_STAGE from environment variable: ${USE_LOCAL_RUNTIME_STAGE}"
fi

if [[ -n "${OVERWRITE_JOB_NAME_ENV:-}" ]]; then
  if [[ "${OVERWRITE_JOB_NAME_ENV}" -eq 1 ]]; then
    # check if JOB_NAME_ENV or JOB_NAME_TEMPLATE_ENV is also set and use name if set, otherwise use the template and if neither is set, use the default naming scheme and print a warning
    if [[ -n "${JOB_NAME_ENV:-}" ]]; then
      JOB_NAME="${JOB_NAME_ENV}"
      echo "OVERWRITE_JOB_NAME_ENV is set to 1, using JOB_NAME from environment variable: ${JOB_NAME}"
    elif [[ -n "${JOB_NAME_TEMPLATE_ENV:-}" ]]; then
      job_name_template="${JOB_NAME_TEMPLATE_ENV}"
      JOB_NAME=$(eval echo "$job_name_template")
      if [[ "${TRIANGULAR_SCALE}" -gt 1 ]]; then
        JOB_NAME="${JOB_NAME}_ts${TRIANGULAR_SCALE}"
      fi
      echo "OVERWRITE_JOB_NAME_ENV is set to 1, using JOB_NAME_TEMPLATE from environment variable: ${JOB_NAME}"
    else
      echo "Warning: OVERWRITE_JOB_NAME_ENV is set to 1 but neither JOB_NAME_ENV nor JOB_NAME_TEMPLATE_ENV is set. Using default naming scheme with template: ${job_name_template}"
      JOB_NAME=$(eval echo "$job_name_template")
      if [[ "${TRIANGULAR_SCALE}" -gt 1 ]]; then
        JOB_NAME="${JOB_NAME}_ts${TRIANGULAR_SCALE}"
      fi
    fi
  fi
fi

# Keep render/video work on a single node with an explicit CPU budget, even when
# the solver allocation spans multiple nodes.
SLURM_CPUS_ON_NODE_FIRST="${SLURM_CPUS_ON_NODE:-}"
if [[ "${SLURM_CPUS_ON_NODE_FIRST}" == *"("* ]]; then
  SLURM_CPUS_ON_NODE_FIRST="${SLURM_CPUS_ON_NODE_FIRST%%(*}"
fi
if [[ -z "${RENDER_CPUS:-}" ]]; then
  if [[ -n "${SLURM_CPUS_ON_NODE_FIRST}" ]]; then
    RENDER_CPUS="${SLURM_CPUS_ON_NODE_FIRST}"
  else
    RENDER_CPUS=16
  fi
fi
if [[ "${RENDER_CPUS}" -lt 1 ]]; then
  RENDER_CPUS=1
fi
MAX_THREADS="${RENDER_CPUS}"
if [[ -z "${FFMPEG_THREADS:-}" ]]; then
  FFMPEG_THREADS=$(( MAX_THREADS < 16 ? MAX_THREADS : 16 ))
fi









############ Environment Setup ############



set -euo pipefail

cd /hpcwork/ro092286/smartsim/ || exit

export SMARTSIM_RUNTIME_ROOT

if (( USE_GPU == 1 )); then
  source /hpcwork/ro092286/smartsim/install.sh cuda-12
else
  source /hpcwork/ro092286/smartsim/install.sh cpu
fi
#source /hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh

# Suppress OpenMPI PMI s1/s2 component probe warnings on systems without libpmi/libpmi2.
export OMPI_MCA_pmix="^s1,s2"
# Work around OpenMPI OMPIO collective I/O instability seen with parallel HDF5 at high rank counts.
# Force ROMIO backend for MPI-IO instead of OMPIO (stack traces showed mca_io_ompio/mca_fcoll_dynamic_gen2).
export OMPI_MCA_io="romio321"
# Also disable the problematic OMPIO collective algorithm if OMPIO gets selected by accident.
export OMPI_MCA_fcoll="^dynamic_gen2"
# Keep SmartRedis logs in Slurm stdout/stderr instead of a separate log file.
export SR_LOG_FILE="stdout"
# Increase SmartRedis verbosity during debugging so backend issues include more context.
export SR_LOG_LEVEL="debug"

module -t list








#########  Input and Output Paths ############


MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
if (( USE_GPU == 1 )); then
  RUNTIME_DEVICE="smartsim_cuda-12"
  PY_ENV="${SMARTSIM_RUNTIME_ROOT}/${RUNTIME_DEVICE}"
else
  RUNTIME_DEVICE="smartsim_cpu"
  PY_ENV="${SMARTSIM_RUNTIME_ROOT}/${RUNTIME_DEVICE}"
fi
EXTERNAL_DIR="/hpcwork/thes2181/mini_app"
#INPUT_IMAGE="${MINI_APP_DIR}/old/Srtm_ramp2.world.21600x10800.jpg"
INPUT_IMAGE="${MINI_APP_DIR}/old/World_elevation_map.png"
PREP_H5="${EXTERNAL_DIR}/${JOB_NAME}/world_init.h5"
TRAJ_H5="${EXTERNAL_DIR}/${JOB_NAME}/world_trajectory.h5"

# check if ${EXTERNAL_DIR} does not exist, if not create it, if it does exist (and is not empty), print a warning
if [[ ! -d "${EXTERNAL_DIR}" ]]; then
  mkdir -p "${EXTERNAL_DIR}"
elif [[ -n "$(ls -A "${EXTERNAL_DIR}")" ]]; then
  echo "Warning: ${EXTERNAL_DIR} is not empty."
fi

# check if ${EXTERNAL_DIR}/${JOB_NAME} does not exist, if not create it, if it does exist (and is not empty), print a warning
if [[ -d "${EXTERNAL_DIR}/${JOB_NAME}" ]]; then
  if [[ -n "$(ls -A "${EXTERNAL_DIR}/${JOB_NAME}")" ]]; then
    echo "Warning: ${EXTERNAL_DIR}/${JOB_NAME} is not empty."
  fi
else
  mkdir -p "${EXTERNAL_DIR}/${JOB_NAME}"
fi


if [[ "${USE_SMARTSIM}" -eq 1 ]] && [[ "${USE_LOCAL_RUNTIME_STAGE}" -eq 1 ]]; then
  RUNTIME_TAR_PATH="${SMARTSIM_RUNTIME_ROOT}/${RUNTIME_DEVICE}.tar"
  LOCAL_RUNTIME_BASE="/tmp/${USER}/smartsim_runtime"
  LOCAL_RUNTIME_TAR="/tmp/${USER}_${SLURM_JOB_ID}_${RUNTIME_DEVICE}.tar"
  LOCAL_RUNTIME_ENV="${LOCAL_RUNTIME_BASE}/${RUNTIME_DEVICE}"
  RUNTIME_STAGE_LOG="logs/runtime_stage_${SLURM_JOB_ID}.log"

  if [[ ! -f "${RUNTIME_TAR_PATH}" ]]; then
    echo "Error: Runtime tarball not found at ${RUNTIME_TAR_PATH}. Run install.sh first to create it."
    exit 1
  fi

  mkdir -p logs
  : > "${RUNTIME_STAGE_LOG}"

  echo "Staging SmartSim runtime tar to local storage. Source: ${RUNTIME_TAR_PATH}"
  runtime_stage_start=$(date +%s)

  copy_runtime_tar_to_group() {
    local het_group="$1"
    local node_count="$2"
    local label="$3"
    local cp_rc

    setopt pipefail
    srun --export=ALL --het-group="${het_group}" --nodes "${node_count}" --ntasks-per-node 1 --cpus-per-task 1 \
      $([[ "${node_count}" -gt 1 ]] && echo "--distribution=block") \
      /bin/zsh -lc "set -euo pipefail; cp -f '${RUNTIME_TAR_PATH}' '${LOCAL_RUNTIME_TAR}'; test -s '${LOCAL_RUNTIME_TAR}'; echo RUNTIME_TAR_COPY_PER_NODE label=${label} het_group=${het_group} host=\$(hostname) tar='${LOCAL_RUNTIME_TAR}'" \
      2>&1 | tee -a "${RUNTIME_STAGE_LOG}"
    cp_rc=${pipestatus[1]}
    unsetopt pipefail
    return "${cp_rc}"
  }

  sbcast_ok=0
  if command -v sbcast >/dev/null 2>&1; then
    set +e
    sbcast --force "${RUNTIME_TAR_PATH}" "${LOCAL_RUNTIME_TAR}" 2>&1 | tee -a "${RUNTIME_STAGE_LOG}"
    sbcast_rc=${pipestatus[1]}
    set -e
    if [[ "${sbcast_rc}" -eq 0 ]]; then
      sbcast_ok=1
    else
      echo "Warning: sbcast failed with rc=${sbcast_rc}; falling back to per-group srun copy." | tee -a "${RUNTIME_STAGE_LOG}"
    fi
  else
    echo "Warning: sbcast not found; falling back to per-group srun copy." | tee -a "${RUNTIME_STAGE_LOG}"
  fi

  if [[ "${sbcast_ok}" -ne 1 ]]; then
    if ! copy_runtime_tar_to_group "${SOLVER_HET_GROUP}" "${_nodes:-1}" "solver"; then
      echo "Error: Runtime tar copy failed on solver group ${SOLVER_HET_GROUP}." | tee -a "${RUNTIME_STAGE_LOG}"
      exit 1
    fi
    if [[ "${DB_HET_GROUP}" -ne "${SOLVER_HET_GROUP}" ]]; then
      if ! copy_runtime_tar_to_group "${DB_HET_GROUP}" "${DB_NODES}" "db"; then
        echo "Error: Runtime tar copy failed on db group ${DB_HET_GROUP}." | tee -a "${RUNTIME_STAGE_LOG}"
        exit 1
      fi
    fi
  fi

  unpack_runtime_on_group() {
    local het_group="$1"
    local node_count="$2"
    local label="$3"
    local stage_rc

    setopt pipefail
    srun --export=ALL --het-group="${het_group}" --nodes "${node_count}" --ntasks-per-node 1 --cpus-per-task 1 \
      $([[ "${node_count}" -gt 1 ]] && echo "--distribution=block") \
      /bin/zsh -lc "set -euo pipefail; mkdir -p '${LOCAL_RUNTIME_BASE}'; tar -xf '${LOCAL_RUNTIME_TAR}' -C '${LOCAL_RUNTIME_BASE}'; test -x '${LOCAL_RUNTIME_ENV}/bin/python3'; echo RUNTIME_STAGE_PER_NODE label=${label} het_group=${het_group} host=\$(hostname) runtime='${LOCAL_RUNTIME_ENV}'" \
      2>&1 | tee -a "${RUNTIME_STAGE_LOG}"
    stage_rc=${pipestatus[1]}
    unsetopt pipefail
    return "${stage_rc}"
  }

  if ! unpack_runtime_on_group "${SOLVER_HET_GROUP}" "${_nodes:-1}" "solver"; then
    echo "Error: Runtime staging/unpack failed on solver group ${SOLVER_HET_GROUP}." | tee -a "${RUNTIME_STAGE_LOG}"
    exit 1
  fi

  if [[ "${DB_HET_GROUP}" -ne "${SOLVER_HET_GROUP}" ]]; then
    if ! unpack_runtime_on_group "${DB_HET_GROUP}" "${DB_NODES}" "db"; then
      echo "Error: Runtime staging/unpack failed on db group ${DB_HET_GROUP}." | tee -a "${RUNTIME_STAGE_LOG}"
      exit 1
    fi
  fi

  runtime_stage_end=$(date +%s)
  RUNTIME_STAGE_DURATION=$((runtime_stage_end - runtime_stage_start))
  echo "Runtime staged to local path ${LOCAL_RUNTIME_ENV}; duration: ${RUNTIME_STAGE_DURATION}s"

  PY_ENV="${LOCAL_RUNTIME_ENV}"
fi

if [[ -f "${PY_ENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${PY_ENV}/bin/activate" || { echo "Failed to activate Python environment at ${PY_ENV}"; exit 1; }
else
  echo "Python environment not found at ${PY_ENV}"
  exit 1
fi

RUNTIME_EXTRA_LIB_DIR="${PY_ENV}/runtime_libs"
if [[ -d "${RUNTIME_EXTRA_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${RUNTIME_EXTRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  echo "Using runtime extra libs from ${RUNTIME_EXTRA_LIB_DIR}"
else
  echo "Warning: runtime extra lib directory not found: ${RUNTIME_EXTRA_LIB_DIR}"
fi

if [[ "${USE_SMARTSIM}" -eq 1 ]]; then
  _backend_req="${MODEL_BACKEND:u}"
  _smart_info_out="$(smart info 2>/dev/null || true)"
  if [[ "${_backend_req}" == "TORCH" ]]; then
    if ! echo "${_smart_info_out}" | grep -Eiq "Torch.*True"; then
      echo "Error: SmartSim runtime at ${PY_ENV} does not report Torch backend support in 'smart info'."
      echo "Refusing to run with MODEL_BACKEND=${MODEL_BACKEND} to avoid runtime RedisAI backend-load failure."
      exit 1
    fi
  elif [[ "${_backend_req}" == "ONNX" || "${_backend_req}" == "ONNXRUNTIME" ]]; then
    if ! echo "${_smart_info_out}" | grep -Eiq "ONNX.*True|ONNXRuntime.*True"; then
      echo "Warning: Could not confirm ONNX backend support from 'smart info' output at ${PY_ENV}."
    fi
  elif [[ "${_backend_req}" == "TF" || "${_backend_req}" == "TENSORFLOW" || "${_backend_req}" == "TFLITE" ]]; then
    if ! echo "${_smart_info_out}" | grep -Eiq "Tensorflow.*True"; then
      echo "Error: SmartSim runtime at ${PY_ENV} does not report Tensorflow backend support in 'smart info'."
      exit 1
    fi
  fi
fi

cd "${MINI_APP_DIR}" || exit

if [[ -n "${MODEL_ARTIFACT_MANIFEST:-}" ]] && [[ -f "${MODEL_ARTIFACT_MANIFEST}" ]]; then
  echo "Resolving model artifact from manifest ${MODEL_ARTIFACT_MANIFEST} for model=${MODEL_NAME} backend=${MODEL_BACKEND}"
  MODEL_RESOLUTION=("${(@f)$(python3 - "${MODEL_ARTIFACT_MANIFEST}" "${MODEL_NAME}" "${MODEL_BACKEND}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
model_name = sys.argv[2]
backend = sys.argv[3].upper()
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
for artifact in payload.get("artifacts", []):
    if artifact.get("model_name") == model_name and str(artifact.get("backend", "")).upper() == backend:
        print(artifact["path"])
        print(",".join(artifact.get("inputs", [])))
        print(",".join(artifact.get("outputs", [])))
        sys.exit(0)
raise SystemExit(f"Artifact for model={model_name} backend={backend} not found in {manifest_path}")
PY
  )}")
  if [[ "${#MODEL_RESOLUTION[@]}" -ge 1 ]] && [[ -n "${MODEL_RESOLUTION[1]}" ]]; then
    MODEL_PATH="${MODEL_RESOLUTION[1]}"
  fi
  if [[ "${#MODEL_RESOLUTION[@]}" -ge 2 ]] && [[ -n "${MODEL_RESOLUTION[2]}" ]]; then
    MODEL_INPUTS="${MODEL_RESOLUTION[2]}"
  fi
  if [[ "${#MODEL_RESOLUTION[@]}" -ge 3 ]] && [[ -n "${MODEL_RESOLUTION[3]}" ]]; then
    MODEL_OUTPUTS="${MODEL_RESOLUTION[3]}"
  fi
  echo "Resolved MODEL_PATH=${MODEL_PATH}"
  if [[ -n "${MODEL_INPUTS}" ]]; then
    echo "Resolved MODEL_INPUTS=${MODEL_INPUTS}"
  fi
  if [[ -n "${MODEL_OUTPUTS}" ]]; then
    echo "Resolved MODEL_OUTPUTS=${MODEL_OUTPUTS}"
  fi
fi

if [[ "${MODEL_PATH}" = /* ]]; then
  MODEL_PATH_SOURCE="${MODEL_PATH}"
else
  MODEL_PATH_SOURCE="${MINI_APP_DIR}/${MODEL_PATH}"
fi
if [[ ! -f "${MODEL_PATH_SOURCE}" ]]; then
  echo "Error: Model file not found at ${MODEL_PATH_SOURCE}"
  exit 1
fi

if [[ "${USE_GPU}" -eq 0 ]] && [[ "${MODEL_BACKEND:u}" == "TORCH" ]] && [[ "${TORCH_CPU_MODEL_CONVERT}" -eq 1 ]]; then
  MODEL_CPU_CONVERT_DIR="/tmp/${USER}/model_cpu_converted_${SLURM_JOB_ID}"
  MODEL_CPU_CONVERT_PATH="${MODEL_CPU_CONVERT_DIR}/$(basename "${MODEL_PATH_SOURCE%.*}")_cpu.pt"
  mkdir -p "${MODEL_CPU_CONVERT_DIR}"
  echo "Converting Torch model to CPU-compatible artifact: ${MODEL_PATH_SOURCE} -> ${MODEL_CPU_CONVERT_PATH}"
  if ! python3 - "${MODEL_PATH_SOURCE}" "${MODEL_CPU_CONVERT_PATH}" <<'PY'
import sys
import torch

src = sys.argv[1]
dst = sys.argv[2]

model = torch.jit.load(src, map_location="cpu")
model = model.eval()
torch.jit.save(model, dst)
print(f"MODEL_CPU_CONVERT_OK src={src} dst={dst}")
PY
  then
    echo "Error: CPU conversion failed for Torch model at ${MODEL_PATH_SOURCE}."
    exit 1
  fi
  MODEL_PATH_SOURCE="${MODEL_CPU_CONVERT_PATH}"
fi

MODEL_PATH_FOR_SOLVER="${MODEL_PATH_SOURCE}"
MODEL_STAGE_LOG="logs/model_stage_${SLURM_JOB_ID}.log"

if [[ "${USE_LOCAL_MODEL_CACHE}" -eq 1 ]]; then
  LOCAL_FAST_ROOT="/tmp/${USER}/model"
  fs_type="$(stat -f -c %T "/tmp" 2>/dev/null || true)"
  MODEL_LOCAL_DIR="${LOCAL_FAST_ROOT}"
  MODEL_LOCAL_PATH="${MODEL_LOCAL_DIR}/$(basename "${MODEL_PATH_SOURCE}")"
  MODEL_PATH_FOR_SOLVER="${MODEL_LOCAL_PATH}"

  mkdir -p logs
  : > "${MODEL_STAGE_LOG}"

  echo "Staging model file to fast volatile storage. Source: ${MODEL_PATH_SOURCE}"
  echo "Model cache root (fixed local path): ${LOCAL_FAST_ROOT} (fs_type=${fs_type:-unknown})"

  get_nodes_for_het_group() {
    local het_group="$1"
    local nodelist_var="SLURM_NODELIST_HET_GROUP_${het_group}"
    local raw_nodelist=""
    local component_job_id="${SLURM_JOB_ID}+${het_group}"
    local component_job_info=""

    if command -v scontrol >/dev/null 2>&1; then
      component_job_info="$(scontrol show job "${component_job_id}" 2>/dev/null || true)"
      if [[ -n "${component_job_info}" ]] && [[ "${component_job_info}" =~ 'NodeList=([^[:space:]]+)' ]]; then
        raw_nodelist="${match[1]}"
      fi
    fi

    if [[ "${raw_nodelist:l}" == "(null)" || "${raw_nodelist:l}" == "null" || "${raw_nodelist:l}" == "none" || "${raw_nodelist:l}" == "n/a" ]]; then
      raw_nodelist=""
    fi

    if [[ -z "${raw_nodelist}" ]]; then
      raw_nodelist="${(P)nodelist_var:-}"
    fi

    if [[ "${raw_nodelist:l}" == "(null)" || "${raw_nodelist:l}" == "null" || "${raw_nodelist:l}" == "none" || "${raw_nodelist:l}" == "n/a" ]]; then
      raw_nodelist=""
    fi

    if [[ -z "${raw_nodelist}" ]]; then
      if [[ "${het_group}" -eq "${SOLVER_HET_GROUP}" ]]; then
        raw_nodelist="${SLURM_JOB_NODELIST:-}"
      else
        return 1
      fi
    fi

    if [[ -z "${raw_nodelist}" ]]; then
      return 1
    fi

    scontrol show hostnames "${raw_nodelist}" 2>/dev/null
  }

  stage_model_to_group() {
    local het_group="$1"
    local node_count="$2"
    local label="$3"
    local stage_start stage_end
    local rc=0
    local total_failed=0
    local node
    local attempt
    local node_ok
    local stage_rc

    local -a group_nodes
    local nodes_out
    nodes_out="$(get_nodes_for_het_group "${het_group}" || true)"
    if [[ -n "${nodes_out}" ]]; then
      group_nodes=("${(@f)nodes_out}")
    fi

    if [[ "${#group_nodes[@]}" -eq 0 ]]; then
      echo "Warning: Could not resolve explicit node list for het-group ${het_group}; falling back to single srun staging." | tee -a "${MODEL_STAGE_LOG}"
      setopt pipefail
      srun --export=ALL --het-group="${het_group}" --nodes "${node_count}" --ntasks-per-node 1 --cpus-per-task 1 \
        $([[ "${node_count}" -gt 1 ]] && echo "--distribution=block") \
        /bin/zsh -lc "set -euo pipefail; _t0=\$(date +%s); mkdir -p \"${MODEL_LOCAL_DIR}\"; cp -f \"${MODEL_PATH_SOURCE}\" \"${MODEL_LOCAL_PATH}\"; test -s \"${MODEL_LOCAL_PATH}\"; _t1=\$(date +%s); echo MODEL_STAGE_PER_NODE label=${label} het_group=${het_group} host=\$(hostname) duration_s=\$((_t1-_t0)) path=${MODEL_LOCAL_PATH}" \
        2>&1 | tee -a "${MODEL_STAGE_LOG}"
      stage_rc=${pipestatus[1]}
      unsetopt pipefail
      if [[ "${stage_rc}" -ne 0 ]]; then
        echo "Error: model staging failed for het-group ${het_group} (${label}) with exit code ${stage_rc}" | tee -a "${MODEL_STAGE_LOG}"
        return "${stage_rc}"
      fi
      stage_start=$(date +%s)
      stage_end=$(date +%s)
    else
      stage_start=$(date +%s)
      for node in "${group_nodes[@]}"; do
        node_ok=0
        for attempt in {1..${MODEL_STAGE_MAX_RETRIES}}; do
          set +e
          srun --export=ALL --het-group="${het_group}" --nodes 1 --ntasks 1 --cpus-per-task 1 --nodelist "${node}" \
            /bin/zsh -lc "set -euo pipefail; _t0=\$(date +%s); mkdir -p \"${MODEL_LOCAL_DIR}\"; cp -f \"${MODEL_PATH_SOURCE}\" \"${MODEL_LOCAL_PATH}\"; test -s \"${MODEL_LOCAL_PATH}\"; _t1=\$(date +%s); echo MODEL_STAGE_PER_NODE label=${label} het_group=${het_group} host=\$(hostname) duration_s=\$((_t1-_t0)) path=${MODEL_LOCAL_PATH}" \
            >> "${MODEL_STAGE_LOG}" 2>&1
          stage_rc=$?
          set -e

          if [[ "${stage_rc}" -eq 0 ]]; then
            node_ok=1
            break
          fi

          echo "MODEL_STAGE_ERROR label=${label} het_group=${het_group} host=${node} attempt=${attempt}/${MODEL_STAGE_MAX_RETRIES} rc=${stage_rc}" | tee -a "${MODEL_STAGE_LOG}"

          set +e
          srun --export=ALL --het-group="${het_group}" --nodes 1 --ntasks 1 --cpus-per-task 1 --nodelist "${node}" \
            /bin/zsh -lc "set +e; echo MODEL_STAGE_DIAG host=\$(hostname); echo MODEL_STAGE_DIAG pwd=\$(pwd); ls -ld \"${LOCAL_FAST_ROOT}\" \"${MODEL_LOCAL_DIR}\" 2>/dev/null || true; df -h \"${LOCAL_FAST_ROOT}\" 2>/dev/null || true; df -i \"${LOCAL_FAST_ROOT}\" 2>/dev/null || true; test -r \"${MODEL_PATH_SOURCE}\" && echo MODEL_STAGE_DIAG model_source_readable=1 || echo MODEL_STAGE_DIAG model_source_readable=0" \
            >> "${MODEL_STAGE_LOG}" 2>&1
          set -e
          sleep 2
        done

        if [[ "${node_ok}" -ne 1 ]]; then
          total_failed=$((total_failed + 1))
          rc=1
        fi
      done
      stage_end=$(date +%s)
    fi

    if [[ "${rc}" -ne 0 ]]; then
      echo "Error: model staging failed for het-group ${het_group} (${label}); failed_nodes=${total_failed}" | tee -a "${MODEL_STAGE_LOG}"
      return 1
    fi

    if [[ "${label}" == "solver" ]]; then
      MODEL_STAGE_DURATION_SOLVER=$((stage_end - stage_start))
    elif [[ "${label}" == "db" ]]; then
      MODEL_STAGE_DURATION_DB=$((stage_end - stage_start))
    fi

    return 0
  }

  local_cache_ok=1
  if ! stage_model_to_group "${SOLVER_HET_GROUP}" "${_nodes:-1}" "solver"; then
    local_cache_ok=0
  fi

  if [[ "${local_cache_ok}" -eq 1 ]] && [[ "${MODEL_STAGE_DB_GROUP}" -eq 1 ]] && [[ "${DB_HET_GROUP}" -ne "${SOLVER_HET_GROUP}" ]]; then
    if ! stage_model_to_group "${DB_HET_GROUP}" "${DB_NODES}" "db"; then
      local_cache_ok=0
    fi
  fi

  if [[ "${local_cache_ok}" -eq 0 ]]; then
    if [[ "${MODEL_STAGE_FALLBACK_TO_SHARED}" -eq 1 ]]; then
      echo "Warning: local model staging failed on at least one node. Falling back to shared model path: ${MODEL_PATH_SOURCE}" | tee -a "${MODEL_STAGE_LOG}"
      MODEL_PATH_FOR_SOLVER="${MODEL_PATH_SOURCE}"
      MODEL_LOCAL_DIR=""
      MODEL_LOCAL_PATH=""
      MODEL_STAGE_DURATION_SOLVER=0
      MODEL_STAGE_DURATION_DB=0
      MODEL_STAGE_TOTAL_DURATION=0
    else
      echo "Error: local model staging failed and fallback is disabled (MODEL_STAGE_FALLBACK_TO_SHARED=0)." | tee -a "${MODEL_STAGE_LOG}"
      exit 1
    fi
  else
    MODEL_STAGE_TOTAL_DURATION=$((MODEL_STAGE_DURATION_SOLVER + MODEL_STAGE_DURATION_DB))
    if [[ "${MODEL_STAGE_DB_GROUP}" -eq 1 ]] && [[ "${DB_HET_GROUP}" -ne "${SOLVER_HET_GROUP}" ]]; then
      echo "Model staged on het-groups ${SOLVER_HET_GROUP} and ${DB_HET_GROUP}; total staging time: ${MODEL_STAGE_TOTAL_DURATION}s"
    else
      echo "Model staged on solver het-group ${SOLVER_HET_GROUP}; total staging time: ${MODEL_STAGE_TOTAL_DURATION}s"
    fi
  fi
else
  echo "USE_LOCAL_MODEL_CACHE=${USE_LOCAL_MODEL_CACHE}; using shared filesystem model path: ${MODEL_PATH_SOURCE}"
fi

CREATE_NEW_H5=1
RUN_SOLVER=1







########### Check and verify existing trajectory file ############


# Check if the trajectory file already exists and how far it has been simulated
LAST_STEP_IN_TRAJ=-1
if [[ -f "${TRAJ_H5}" ]]; then
  LAST_STEP_IN_TRAJ=$(python3 -c "
import h5py, sys
try:
    with h5py.File('${TRAJ_H5}', 'r') as f:
        idx = f.get('step_index')
        if idx is not None and len(idx) > 0:
            print(int(idx[-1]))
        else:
            # Fallback for interrupted/legacy files where step_index is absent or empty:
            # infer from water time dimension and save_every metadata.
            water = f.get('water')
            if water is not None and getattr(water, 'shape', None) and len(water.shape) == 3 and water.shape[0] > 0:
                save_every_attr = f.attrs.get('save_every', ${SAVE_EVERY})
                save_every = int(save_every_attr) if save_every_attr is not None else ${SAVE_EVERY}
                print(int((water.shape[0] - 1) * save_every))
            else:
                print(-1)
except Exception:
    print(-1)
" 2>/dev/null || echo "-1")
  echo "Trajectory file exists. Last saved step: ${LAST_STEP_IN_TRAJ} (target: ${TOTAL_STEPS})"
fi

SOLVE_DURATION=0
if [[ "${LAST_STEP_IN_TRAJ}" -ge "${TOTAL_STEPS}" ]]; then
  echo "Skipping solver: trajectory already complete at step ${LAST_STEP_IN_TRAJ}."
  CREATE_NEW_H5=0
  RUN_SOLVER=0
elif [[ "${LAST_STEP_IN_TRAJ}" -ge 0 ]]; then
  echo "Trajectory is incomplete at step ${LAST_STEP_IN_TRAJ}; solver resume is not implemented yet, restarting fresh."
  CREATE_NEW_H5=1
else
  echo "No existing trajectory found. Starting fresh..."
fi




########### Start rudimentary time measurement ############

START_TIME=$(date +%s)








############ Create initial condition H5 file if needed ############


if [[ "${CREATE_NEW_H5}" -eq 1 ]]; then
  python3 prepare.py \
    --input-image "${INPUT_IMAGE}" \
    --output-hdf5 "${PREP_H5}" \
    --target-width "${TARGET_WIDTH}" \
    --target-height "${TARGET_HEIGHT}" \
    --chunk-size "${CHUNK_SIZE}" \
    --init-mode "${INIT_MODE}" \
    --init-depth "${INIT_DEPTH}" \
    --radius "${RADIUS}"

  PREP_END_TIME=$(date +%s)
  PREP_DURATION=$((PREP_END_TIME - START_TIME))
  echo "Preparation time: ${PREP_DURATION} seconds"
else
  echo "Skipping preparation since trajectory file already exists."
  PREP_DURATION=0
fi









############# Run solver ############


if [[ "${RUN_SOLVER}" -eq 1 ]]; then

  # Safety check: if SKIP_COMPILE is true but the binary may be stale, rebuild anyway.
  # This can happen if modules were loaded differently between job submissions.
  FORCE_RECOMPILE=0
  if [[ "${SKIP_COMPILE}" -eq 1 ]] && [[ -f "solver_cpp/build/terrain_solver" ]]; then
    # Quick sanity check: run binary with -h to see if it crashes with HDF5 version mismatch.
    # If it does, the error message will contain "version mismatch" or "HDF5 header files".
    if ./solver_cpp/build/terrain_solver --help 2>&1 | grep -qi "hdf5.*version\|version.*mismatch"; then
      echo "WARNING: Detected HDF5 version mismatch in binary; forcing recompile despite SKIP_COMPILE=1"
      FORCE_RECOMPILE=1
    fi
  fi

  if [[ "${SKIP_COMPILE}" -eq 1 ]] && [[ "${FORCE_RECOMPILE}" -eq 0 ]]; then
    echo "Skipping compilation as requested."
    COMPILE_DURATION=0
  else
    echo "Compiling solver..."

    COMPILE_START_TIME=$(date +%s)

    # check if the solver_cpp/build directory exists, if it does, remove it to ensure a clean build
    if [[ -d "solver_cpp/build" ]]; then
      rm -rf "solver_cpp/build"
    fi
    mkdir -p "solver_cpp/build"
    cmake -S solver_cpp -B solver_cpp/build
    cmake --build solver_cpp/build -j

    COMPILE_END_TIME=$(date +%s)
    COMPILE_DURATION=$((COMPILE_END_TIME - COMPILE_START_TIME))
    echo "Compilation time: ${COMPILE_DURATION} seconds"

  fi


  SOLVE_START_TIME=$(date +%s)

  DB_HOSTNAME="127.0.0.1:6379"

  if [[ "${USE_SMARTSIM}" -eq 1 ]]; then
    if [[ "${DB_NODE_PREFLIGHT}" -eq 1 ]]; then
      echo "Running DB node preflight checks (ib0, port 6780, local write)..."
      DB_PREFLIGHT_LOG="logs/db_preflight_${SLURM_JOB_ID}.log"
      mkdir -p logs
      : > "${DB_PREFLIGHT_LOG}"
      set +e
      srun --export=ALL --het-group="${DB_HET_GROUP}" --nodes "${DB_NODES}" --ntasks-per-node 1 --cpus-per-task 1 \
        $([[ "${DB_NODES}" -gt 1 ]] && echo "--distribution=block") \
        /bin/zsh -lc 'set +e; _host=$(hostname); _ib=$(ip -o -4 addr show ib0 2>/dev/null | awk "{print \$4}" | head -n1); if [[ -z "${_ib}" ]]; then _ib="missing"; fi; echo "DB_PREFLIGHT host=${_host} ib0=${_ib}"; python3 -c "import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind((\"0.0.0.0\", 6780)); s.close(); print(\"DB_PREFLIGHT port_bind_6780=ok\")" 2>/dev/null || echo "DB_PREFLIGHT port_bind_6780=fail"; _root="${LOCAL_FAST_ROOT:-/tmp}"; _probe="${_root%/}/.db_preflight_${SLURM_JOB_ID}_$$"; mkdir -p "${_probe}" 2>/dev/null && echo "DB_PREFLIGHT fs_write=ok root=${_root}" && rmdir "${_probe}" 2>/dev/null || echo "DB_PREFLIGHT fs_write=fail root=${_root}"' \
        2>&1 | tee -a "${DB_PREFLIGHT_LOG}"
      db_preflight_rc=$?
      set -e
      if [[ "${db_preflight_rc}" -ne 0 ]]; then
        echo "Warning: DB preflight step returned non-zero (${db_preflight_rc}). Continuing; see ${DB_PREFLIGHT_LOG}."
      fi
    fi

    echo "Starting SmartSim controller in the background..."

    # SmartRedis defaults to clustered mode unless SR_DB_TYPE is set.
    # Our mini_app launches a single non-cluster Redis instance.
    
    # Let's check if the het-group 1 node count is larger than 1, in which case we should use "Clustered" mode, otherwise we can use "Standalone" mode which is more lightweight and doesn't require the controller to manage cluster topology.
    if [[ "${DB_NODES}" -gt 1 ]]; then
      echo "DB_NODES=${DB_NODES} > 1, using SmartSim Clustered mode"
      export SR_DB_TYPE="Clustered"
    else
      echo "DB_NODES=${DB_NODES} <= 1, using SmartSim Standalone mode"
      export SR_DB_TYPE="Standalone"
    fi
  
    # pipe the output of the SmartSim controller to a log file for debugging
    python_path="${PY_ENV}/bin/python3"
    echo "Using Python interpreter at ${python_path} for SmartSim controller"
    max_wait_time=300 # seconds
    driver_log="logs/mini_app_driver_${SLURM_JOB_ID}.txt"
    controller_started=0
    controller_attempt=1
    while [[ "${controller_attempt}" -le "${CONTROLLER_START_MAX_RETRIES}" ]]; do
      rm -f "${DB_HOSTNAME_FILE}"
      echo "=== CONTROLLER_ATTEMPT ${controller_attempt}/${CONTROLLER_START_MAX_RETRIES} ===" >> "${driver_log}"
      ${python_path} smartsim_controller.py --db_nodes "${DB_NODES}" --het_group="${DB_HET_GROUP}" --hostname_file="${DB_HOSTNAME_FILE}" $([ "${USE_GPU}" -eq 1 ] && echo "--use_gpu") --cpu_cores_per_node="${ML_INFERENCE_CPU_CORES}" >> "${driver_log}" 2>&1 &
      DRIVER_PID=$!
      echo "Wait until database hostname file ${DB_HOSTNAME_FILE} is created by the SmartSim controller...(attempt ${controller_attempt}/${CONTROLLER_START_MAX_RETRIES}, at most ${max_wait_time}s)"
      wait_time=0
      while [[ ! -f "${DB_HOSTNAME_FILE}" ]]; do
        if ! ps -p "${DRIVER_PID}" > /dev/null 2>&1; then
          echo "Warning: SmartSim controller exited before writing ${DB_HOSTNAME_FILE} (attempt ${controller_attempt}/${CONTROLLER_START_MAX_RETRIES})."
          break
        fi
        sleep 1
        wait_time=$((wait_time + 1))
        if [[ ${wait_time} -ge ${max_wait_time} ]]; then
          echo "Warning: Timeout waiting for ${DB_HOSTNAME_FILE} (attempt ${controller_attempt}/${CONTROLLER_START_MAX_RETRIES})."
          if ps -p "${DRIVER_PID}" > /dev/null 2>&1; then
            kill "${DRIVER_PID}" 2>/dev/null || true
          fi
          break
        fi
      done

      if [[ -f "${DB_HOSTNAME_FILE}" ]]; then
        controller_started=1
        break
      fi

      controller_attempt=$((controller_attempt + 1))
      sleep 2
    done

    if [[ "${controller_started}" -ne 1 ]]; then
      echo "ERROR: SmartSim controller failed to provide ${DB_HOSTNAME_FILE} after ${CONTROLLER_START_MAX_RETRIES} attempts. See ${driver_log}."
      exit 1
    fi

    DB_HOSTNAME=$(cat "${DB_HOSTNAME_FILE}")
    echo "Database hostname obtained from SmartSim controller: ${DB_HOSTNAME}"
    rm "${DB_HOSTNAME_FILE}"
  fi



  RANK_GRID_ARGS=()
  if [[ "${RANK_GRID_X}" -gt 0 ]]; then
    RANK_GRID_ARGS+=(--rank-grid-x "${RANK_GRID_X}")
  fi
  if [[ "${RANK_GRID_Z}" -gt 0 ]]; then
    RANK_GRID_ARGS+=(--rank-grid-z "${RANK_GRID_Z}")
  fi

  OVERWRITE_ARG=()
  if [[ "${OVERWRITE_OUTPUT}" -eq 1 ]] && [[ "${CREATE_NEW_H5}" -eq 1 ]]; then
    OVERWRITE_ARG+=(--overwrite-output)
  fi

  MODEL_IO_ARGS=()
  if [[ -n "${MODEL_INPUTS}" ]]; then
    MODEL_IO_ARGS+=(--model-inputs "${MODEL_INPUTS}")
  fi
  if [[ -n "${MODEL_OUTPUTS}" ]]; then
    MODEL_IO_ARGS+=(--model-outputs "${MODEL_OUTPUTS}")
  fi

  # For multi-node jobs, use --distribution=block to evenly spread tasks across nodes.
  # This prevents task desynchronization at shutdown (especially in rank0_gather I/O mode).
  SRUN_DIST="--distribution=block"
  if [[ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]]; then
    SRUN_DIST="" # Single node: distribution doesn't matter
  fi
  mkdir -p logs
  setopt pipefail
  device="CPU"
  if (( USE_GPU == 1 )); then
    device="GPU"
  fi

  SSDB="${DB_HOSTNAME}" srun --export=ALL --het-group="${SOLVER_HET_GROUP}" --ntasks-per-node "${_ntasks_per_node_num}" \
    --cpus-per-task 1 \
    ${SRUN_DIST} \
    ./solver_cpp/build/terrain_solver \
    --device "${device}" \
    --gpus-per-node "${GPUS_PER_NODE}" \
    --ml-batch-size "${ML_BATCH_SIZE}" \
    --model-path "${MODEL_PATH_FOR_SOLVER}" \
    --model-backend "${MODEL_BACKEND}" \
    "${MODEL_IO_ARGS[@]}" \
    --input-hdf5 "${PREP_H5}" \
    --output-hdf5 "${TRAJ_H5}" \
    --steps "${TOTAL_STEPS}" \
    --save-every "${SAVE_EVERY}" \
    --save-mode "${SAVE_MODE}" \
    --triangular-scale "${TRIANGULAR_SCALE}" \
    --chunk-size "${CHUNK_SIZE}" \
    --io-mode "${IO_MODE}" \
    --mpi-sync-mode "${MPI_SYNC_MODE}" \
    --hdf5-xfer-mode "${HDF5_XFER_MODE}" \
    "${RANK_GRID_ARGS[@]}" \
    "${OVERWRITE_ARG[@]}" \
    --write-surface
    #--write-surface 2>&1 | tee "${SOLVER_STEP_LOG}"
  SRUN_STATUS=${pipestatus[1]}
  unsetopt pipefail
  if [[ "${SRUN_STATUS}" -ne 0 ]]; then
    echo "Solver failed with exit code ${SRUN_STATUS}"
    exit "${SRUN_STATUS}"
  fi

  SOLVE_END_TIME=$(date +%s)
  SOLVE_DURATION=$((SOLVE_END_TIME - SOLVE_START_TIME))
  echo "Solving time: ${SOLVE_DURATION} seconds"
else
  echo "Skipping solver execution since trajectory is already complete."
  COMPILE_DURATION=0
  SOLVE_DURATION=0
fi

if [[ "${USE_SMARTSIM}" -eq 1 ]]; then
  rm -f "${DB_HOSTNAME_FILE}"
  echo "Creating 'close_driver_${SLURM_JOB_ID}.txt' file to signal SmartSim controller to shut down..."
  echo "Done" > "close_driver_${SLURM_JOB_ID}.txt"
  echo "Waiting 10s before killing driver to allow for graceful shutdown..."
  sleep 10
  if [[ -z "${DRIVER_PID:-}" ]]; then
    echo "SmartSim controller PID is not set (controller may not have started in this run). Skipping controller termination."
  else
  # check if process is still running before attempting to kill
  if ps -p "${DRIVER_PID}" > /dev/null 2>&1; then
    echo "Driver process with PID ${DRIVER_PID} is still running, attempting to terminate..."
    if kill -0 "${DRIVER_PID}" 2>/dev/null; then
      echo "Terminating SmartSim controller with PID ${DRIVER_PID}"
      kill "${DRIVER_PID}"
    else
      echo "SmartSim controller process with PID ${DRIVER_PID} is not running."
    fi
  else
  echo "Driver process with PID ${DRIVER_PID} has already exited."
  fi
  fi
fi






############ Post-processing and rendering ############


if [[ "${SKIP_RENDERING}" -eq 1 ]]; then
  echo "Skipping rendering and video generation as requested."

  RENDER_TOP_VIEW_DURATION=0
  RENDER_SLICE_FULL_HEIGHT_DURATION=0
  RENDER_SLICE_REDUCED_HEIGHT_DURATION=0
  TOP_VIEW_VIDEO_DURATION=0
  SLICE_FULL_HEIGHT_VIDEO_DURATION=0
  COMBINED_VIDEO_1_1_DURATION=0
  COMBINED_VIDEO_16_9_DURATION=0

else

  export OMP_NUM_THREADS="${RENDER_CPUS}"
  echo "Post-processing on batch host $(hostname) with RENDER_CPUS=${RENDER_CPUS}, FFMPEG_THREADS=${FFMPEG_THREADS}"

  python3 render.py \
    --input-hdf5 "${TRAJ_H5}" \
    --step -1 \
    --output-png "${EXTERNAL_DIR}/${JOB_NAME}/world_last_step.png"



  rm -rf "${EXTERNAL_DIR}/${JOB_NAME}/rendered_frames"

  RENDER_TOP_VIEW_START_TIME=$(date +%s)

  python3 render.py \
    --all-steps \
    --output-dir "${EXTERNAL_DIR}/${JOB_NAME}/rendered_frames" \
    --input-hdf5 "${TRAJ_H5}" \
    --threads "${MAX_THREADS}"

  RENDER_TOP_VIEW_END_TIME=$(date +%s)
  RENDER_TOP_VIEW_DURATION=$((RENDER_TOP_VIEW_END_TIME - RENDER_TOP_VIEW_START_TIME))
  echo "Top view rendering time: ${RENDER_TOP_VIEW_DURATION} seconds"

  rm -rf "${EXTERNAL_DIR}/${JOB_NAME}/rendered_slices"

  RENDER_SLICE_FULL_HEIGHT_START_TIME=$(date +%s)

  python3 render_slice.py \
    --all-steps \
    --slice-z $((TARGET_HEIGHT / 2)) \
    --output-dir "${EXTERNAL_DIR}/${JOB_NAME}/rendered_slices" \
    --input-hdf5 "${TRAJ_H5}" \
    --threads "${MAX_THREADS}"

  RENDER_SLICE_FULL_HEIGHT_END_TIME=$(date +%s)
  RENDER_SLICE_FULL_HEIGHT_DURATION=$((RENDER_SLICE_FULL_HEIGHT_END_TIME - RENDER_SLICE_FULL_HEIGHT_START_TIME))
  echo "Slice (full height) rendering time: ${RENDER_SLICE_FULL_HEIGHT_DURATION} seconds"

  rm -rf "${EXTERNAL_DIR}/${JOB_NAME}/rendered_slices_reduced_height"

  RENDER_SLICE_REDUCED_HEIGHT_START_TIME=$(date +%s)

  python3 render_slice.py \
    --all-steps \
    --slice-z $((TARGET_HEIGHT / 2)) \
    --height-min ${HEIGHT_MIN} \
    --image-height ${REDUCED_HEIGHT} \
    --output-dir "${EXTERNAL_DIR}/${JOB_NAME}/rendered_slices_reduced_height" \
    --input-hdf5 "${TRAJ_H5}" \
    --threads "${MAX_THREADS}"

  RENDER_SLICE_REDUCED_HEIGHT_END_TIME=$(date +%s)
  RENDER_SLICE_REDUCED_HEIGHT_DURATION=$((RENDER_SLICE_REDUCED_HEIGHT_END_TIME - RENDER_SLICE_REDUCED_HEIGHT_START_TIME))
  echo "Slice (reduced height) rendering time: ${RENDER_SLICE_REDUCED_HEIGHT_DURATION} seconds"

  NORMAL_GCC_MODULE=$(module -t list 2>&1 | grep "^GCC/")

  echo "Currently loaded GCC module: ${NORMAL_GCC_MODULE}"


  FFMPEG_VERSION="FFmpeg/7.1.1"
  FFMPEG_GCC_MODULE="GCCcore/14.2.0"

  echo "Unloading currently loaded GCC module to avoid conflicts with FFmpeg's dependencies..."
  module unload "${NORMAL_GCC_MODULE}" || { echo "Failed to unload GCC module ${NORMAL_GCC_MODULE}"; exit 1; }
  module load "${FFMPEG_GCC_MODULE}" || { echo "Failed to load FFmpeg GCCcore module ${FFMPEG_GCC_MODULE}"; exit 1; }
  module load "${FFMPEG_VERSION}" || { echo "Failed to load FFmpeg module ${FFMPEG_VERSION}"; exit 1; }


  cd "${EXTERNAL_DIR}/${JOB_NAME}" || exit

  # Render 4 videos:
  # 1. Top view
  # 2. Slice with full height
  # 3. Combined video with top view above and slice with full height below (1:1 aspect ratio)
  # 4. Combined video with top view above and slice with reduced height below (16:9 aspect ratio)

  TOP_VIEW_VIDEO_START_TIME=$(date +%s)

  ffmpeg \
    -framerate ${VIDEO_FPS} \
    -pattern_type glob \
    -i 'rendered_frames/frame_*.png' \
    -c:v libx265 \
    -crf 23 \
    -preset slow \
    -threads ${FFMPEG_THREADS} \
    top_view.mp4

  TOP_VIEW_VIDEO_END_TIME=$(date +%s)
  TOP_VIEW_VIDEO_DURATION=$((TOP_VIEW_VIDEO_END_TIME - TOP_VIEW_VIDEO_START_TIME))
  echo "Top view video rendering time: ${TOP_VIEW_VIDEO_DURATION} seconds"

  SLICE_FULL_HEIGHT_VIDEO_START_TIME=$(date +%s)

  ffmpeg \
    -framerate ${VIDEO_FPS} \
    -pattern_type glob \
    -i 'rendered_slices/frame_*.png' \
    -c:v libx265 \
    -crf 23 \
    -preset slow \
    -threads ${FFMPEG_THREADS} \
    slice_full_height.mp4

  SLICE_FULL_HEIGHT_VIDEO_END_TIME=$(date +%s)
  SLICE_FULL_HEIGHT_VIDEO_DURATION=$((SLICE_FULL_HEIGHT_VIDEO_END_TIME - SLICE_FULL_HEIGHT_VIDEO_START_TIME))
  echo "Slice (full height) video rendering time: ${SLICE_FULL_HEIGHT_VIDEO_DURATION} seconds"

  cd "${MINI_APP_DIR}" || exit

  # verify that make_dual_view_video.sh is present and executable
  if [[ ! -x "make_dual_view_video.sh" ]]; then
    echo "Error: make_dual_view_video.sh not found or not executable in ${MINI_APP_DIR}"
    exit 1
  fi


  COMBINED_VIDEO_1_1_START_TIME=$(date +%s)


  ./make_dual_view_video.sh \
    --top-pattern "${EXTERNAL_DIR}/${JOB_NAME}/rendered_frames/frame_*.png" \
    --slice-pattern "${EXTERNAL_DIR}/${JOB_NAME}/rendered_slices/frame_*.png" \
    --output "${EXTERNAL_DIR}/${JOB_NAME}/dual_1x1.mp4" \
    --fps ${VIDEO_FPS} \
    --threads ${FFMPEG_THREADS}

  COMBINED_VIDEO_1_1_END_TIME=$(date +%s)
  COMBINED_VIDEO_1_1_DURATION=$((COMBINED_VIDEO_1_1_END_TIME - COMBINED_VIDEO_1_1_START_TIME))
  echo "Combined video (1:1) rendering time: ${COMBINED_VIDEO_1_1_DURATION} seconds"

  COMBINED_VIDEO_16_9_START_TIME=$(date +%s)


  ./make_dual_view_video.sh \
    --top-pattern "${EXTERNAL_DIR}/${JOB_NAME}/rendered_frames/frame_*.png" \
    --slice-pattern "${EXTERNAL_DIR}/${JOB_NAME}/rendered_slices_reduced_height/frame_*.png" \
    --output "${EXTERNAL_DIR}/${JOB_NAME}/dual_16x9.mp4" \
    --width ${TARGET_WIDTH} \
    --top-height ${TARGET_HEIGHT} \
    --bottom-height ${REDUCED_HEIGHT} \
    --fps ${VIDEO_FPS} \
    --threads ${FFMPEG_THREADS}

  COMBINED_VIDEO_16_9_END_TIME=$(date +%s)
  COMBINED_VIDEO_16_9_DURATION=$((COMBINED_VIDEO_16_9_END_TIME - COMBINED_VIDEO_16_9_START_TIME))
  echo "Combined video (16:9) rendering time: ${COMBINED_VIDEO_16_9_DURATION} seconds"


  module unload "${FFMPEG_VERSION}" "${FFMPEG_GCC_MODULE}" || { echo "Failed to unload FFmpeg/GCCcore modules"; exit 1; }
  module load "${NORMAL_GCC_MODULE}" || { echo "Failed to reload original GCC module ${NORMAL_GCC_MODULE}"; exit 1; }

  NORMAL_GCC_MODULE=$(module -t list 2>&1 | grep "^GCC/")

  echo "Now using ${NORMAL_GCC_MODULE} as the GCC module after loading FFmpeg"

fi







############ Finalize timing and write results to file ############


TIMING_FILE="${EXTERNAL_DIR}/${JOB_NAME}/timing_and_parameters.txt"

{
  echo "Job Name: ${JOB_NAME}"
  echo "Slurm Job ID: ${SLURM_JOB_ID:-N/A}"
  echo "Slurm Nodes: ${SLURM_JOB_NODELIST:-N/A}"
  echo "Slurm Node Count: ${SLURM_JOB_NUM_NODES:-N/A}"
  echo "Slurm NTASKS: ${SLURM_NTASKS:-N/A}"
  echo "Slurm CPUs per Task: ${SLURM_CPUS_PER_TASK:-N/A}"
  echo "Slurm Memory per Node: ${SLURM_MEM_PER_NODE:-N/A}"
  echo "Slurm Time Limit: ${SLURM_TIME_LIMIT:-N/A}"
  echo "Slurm Partition: ${SLURM_JOB_PARTITION:-N/A}"
  echo "Slurm Account: ${SLURM_ACCOUNT:-N/A}"
  echo ""
  echo "Parameters:"
  echo "INIT_MODE: ${INIT_MODE}"
  echo "RADIUS: ${RADIUS}"
  echo "INIT_DEPTH: ${INIT_DEPTH}"
  echo "TOTAL_STEPS: ${TOTAL_STEPS}"
  echo "SAVE_EVERY: ${SAVE_EVERY}"
  echo "SAVE_MODE: ${SAVE_MODE}"
  echo "TRIANGULAR_SCALE: ${TRIANGULAR_SCALE}"
  echo "TARGET_WIDTH: ${TARGET_WIDTH}"
  echo "TARGET_HEIGHT: ${TARGET_HEIGHT}"
  echo "CHUNK_SIZE: ${CHUNK_SIZE}"
  echo "MPI_RANKS: ${MPI_RANKS}"
  echo "IO_MODE: ${IO_MODE}"
  echo "MPI_SYNC_MODE: ${MPI_SYNC_MODE}"
  echo "RANK_GRID_X: ${RANK_GRID_X}"
  echo "RANK_GRID_Z: ${RANK_GRID_Z}"
  echo "OVERWRITE_OUTPUT: ${OVERWRITE_OUTPUT}"
  echo "SKIP_RENDERING: ${SKIP_RENDERING}"
  echo "SKIP_COMPILE: ${SKIP_COMPILE}"
  echo "FFMPEG_THREADS: ${FFMPEG_THREADS}"
  echo "RENDER_CPUS: ${RENDER_CPUS}"
  echo "REDUCED_HEIGHT: ${REDUCED_HEIGHT}"
  echo "HEIGHT_MIN: ${HEIGHT_MIN}"
  echo "USE_LOCAL_MODEL_CACHE: ${USE_LOCAL_MODEL_CACHE}"
  echo "SMARTSIM_RUNTIME_ROOT: ${SMARTSIM_RUNTIME_ROOT}"
  echo "USE_LOCAL_RUNTIME_STAGE: ${USE_LOCAL_RUNTIME_STAGE}"
  echo "RUNTIME_STAGE_LOG: ${RUNTIME_STAGE_LOG:-N/A}"
  echo "RUNTIME_STAGE_DURATION: ${RUNTIME_STAGE_DURATION}"
  echo "LOCAL_FAST_ROOT: ${LOCAL_FAST_ROOT:-N/A}"
  echo "MODEL_STAGE_LOG: ${MODEL_STAGE_LOG:-N/A}"
  echo "MODEL_STAGE_DURATION_SOLVER: ${MODEL_STAGE_DURATION_SOLVER}"
  echo "MODEL_STAGE_DURATION_DB: ${MODEL_STAGE_DURATION_DB}"
  echo "MODEL_STAGE_DURATION_TOTAL: ${MODEL_STAGE_TOTAL_DURATION}"
  echo "MODEL_PATH_SOURCE: ${MODEL_PATH_SOURCE}"
  echo "MODEL_PATH_FOR_SOLVER: ${MODEL_PATH_FOR_SOLVER}"
  echo "SOLVER_STEP_LOG: ${SOLVER_STEP_LOG}"
  echo ""
  echo "Model staging per-node log:"
  if [[ -f "${MODEL_STAGE_LOG}" ]]; then
    grep "^MODEL_STAGE_PER_NODE" "${MODEL_STAGE_LOG}" || echo "  No per-node staging lines found in ${MODEL_STAGE_LOG}"
  else
    echo "  Staging log not found: ${MODEL_STAGE_LOG}"
  fi
  echo ""
  echo "Solver step statistics (parsed from solver log):"
  if [[ -f "${SOLVER_STEP_LOG}" ]]; then
    python3 - "${SOLVER_STEP_LOG}" <<'PY'
import re
import sys

path = sys.argv[1]
step_re = re.compile(r"^Step\s+(\d+),\s*(ML|Regular),.*time:\s*([0-9.eE+-]+)\s*ms")
prepare_re = re.compile(r"^Prepare data time \(seconds\):\s*([0-9.eE+-]+)")
put_re = re.compile(r"^Put tensor time \(seconds\):\s*([0-9.eE+-]+)")
run_re = re.compile(r"^Run model time \(seconds\):\s*([0-9.eE+-]+)")
unpack_re = re.compile(r"^Unpack time \(seconds\):\s*([0-9.eE+-]+)")
acc_re = re.compile(r"^ML timing accounting \(seconds\): total=([0-9.eE+-]+), accounted=([0-9.eE+-]+), cleanup=([0-9.eE+-]+)")

step_stats = {
"ML": {"count": 0, "sum_ms": 0.0, "min_ms": None, "max_ms": None},
"Regular": {"count": 0, "sum_ms": 0.0, "min_ms": None, "max_ms": None},
}
prepare_vals = []
put_vals = []
run_vals = []
unpack_vals = []
ml_total_vals = []
ml_accounted_vals = []
ml_cleanup_vals = []

with open(path, "r", encoding="utf-8", errors="replace") as f:
  for raw in f:
    line = raw.strip()
    m = step_re.match(line)
    if m:
      solver = m.group(2)
      ms = float(m.group(3))
      st = step_stats[solver]
      st["count"] += 1
      st["sum_ms"] += ms
      st["min_ms"] = ms if st["min_ms"] is None else min(st["min_ms"], ms)
      st["max_ms"] = ms if st["max_ms"] is None else max(st["max_ms"], ms)
      continue
    m = prepare_re.match(line)
    if m:
      prepare_vals.append(float(m.group(1)))
      continue
    m = put_re.match(line)
    if m:
      put_vals.append(float(m.group(1)))
      continue
    m = run_re.match(line)
    if m:
      run_vals.append(float(m.group(1)))
      continue
    m = unpack_re.match(line)
    if m:
      unpack_vals.append(float(m.group(1)))
      continue
    m = acc_re.match(line)
    if m:
      ml_total_vals.append(float(m.group(1)))
      ml_accounted_vals.append(float(m.group(2)))
      ml_cleanup_vals.append(float(m.group(3)))

def fmt_step(name):
  st = step_stats[name]
  if st["count"] == 0:
    return f"  {name}: count=0"
  avg = st["sum_ms"] / st["count"]
  return (
    f"  {name}: count={st['count']}, total_ms={st['sum_ms']:.3f}, "
    f"avg_ms={avg:.3f}, min_ms={st['min_ms']:.3f}, max_ms={st['max_ms']:.3f}"
  )

def fmt_series(name, values):
  if not values:
    return f"  {name}: count=0"
  total = sum(values)
  avg = total / len(values)
  return f"  {name}: count={len(values)}, total_s={total:.6f}, avg_s={avg:.6f}"

total_steps = step_stats["ML"]["count"] + step_stats["Regular"]["count"]
print(f"  Parsed step lines: {total_steps}")
print(fmt_step("Regular"))
print(fmt_step("ML"))
print("  ML substeps:")
print(fmt_series("prepare_data", prepare_vals))
print(fmt_series("put_tensor", put_vals))
print(fmt_series("run_model", run_vals))
print(fmt_series("unpack", unpack_vals))
print(fmt_series("ml_total_wall", ml_total_vals))
print(fmt_series("ml_accounted", ml_accounted_vals))
print(fmt_series("ml_cleanup", ml_cleanup_vals))
PY
  else
    echo "  Solver log not found: ${SOLVER_STEP_LOG}"
  fi
  echo ""
  echo "Timings (seconds):"
  echo "Runtime staging total: ${RUNTIME_STAGE_DURATION}"
  echo "Model staging total: ${MODEL_STAGE_TOTAL_DURATION}"
  echo "Model staging solver-group: ${MODEL_STAGE_DURATION_SOLVER}"
  echo "Model staging db-group: ${MODEL_STAGE_DURATION_DB}"
  echo "Preparation: ${PREP_DURATION}"
  echo "Compilation: ${COMPILE_DURATION}"
  echo "Solving: ${SOLVE_DURATION}"
  echo "Top view rendering: ${RENDER_TOP_VIEW_DURATION}"
  echo "Slice full height rendering: ${RENDER_SLICE_FULL_HEIGHT_DURATION}"
  echo "Slice reduced height rendering: ${RENDER_SLICE_REDUCED_HEIGHT_DURATION}"
  echo "Top view video: ${TOP_VIEW_VIDEO_DURATION}"
  echo "Slice full height video: ${SLICE_FULL_HEIGHT_VIDEO_DURATION}"
  echo "Combined video 1:1: ${COMBINED_VIDEO_1_1_DURATION}"
  echo "Combined video 16:9: ${COMBINED_VIDEO_16_9_DURATION}"
  echo "Total: $((RUNTIME_STAGE_DURATION + MODEL_STAGE_TOTAL_DURATION + PREP_DURATION + COMPILE_DURATION + SOLVE_DURATION + RENDER_TOP_VIEW_DURATION + RENDER_SLICE_FULL_HEIGHT_DURATION + RENDER_SLICE_REDUCED_HEIGHT_DURATION + TOP_VIEW_VIDEO_DURATION + SLICE_FULL_HEIGHT_VIDEO_DURATION + COMBINED_VIDEO_1_1_DURATION + COMBINED_VIDEO_16_9_DURATION))"
} > "${TIMING_FILE}"

echo "Timing and parameters saved to: ${TIMING_FILE}"


echo ""
echo "================================"

echo "Runtime staging time: ${RUNTIME_STAGE_DURATION} seconds"
echo "Model staging total time: ${MODEL_STAGE_TOTAL_DURATION} seconds"
echo "Model staging solver-group time: ${MODEL_STAGE_DURATION_SOLVER} seconds"
echo "Model staging db-group time: ${MODEL_STAGE_DURATION_DB} seconds"
echo "Preparation time: ${PREP_DURATION} seconds"
echo "Compilation time: ${COMPILE_DURATION} seconds"
echo "Solving time: ${SOLVE_DURATION} seconds"
echo "Top view rendering time: ${RENDER_TOP_VIEW_DURATION} seconds"
echo "Slice (full height) rendering time: ${RENDER_SLICE_FULL_HEIGHT_DURATION} seconds"
echo "Slice (reduced height) rendering time: ${RENDER_SLICE_REDUCED_HEIGHT_DURATION} seconds"
echo "Top view video rendering time: ${TOP_VIEW_VIDEO_DURATION} seconds"
echo "Slice (full height) video rendering time: ${SLICE_FULL_HEIGHT_VIDEO_DURATION} seconds"
echo "Combined video (1:1) rendering time: ${COMBINED_VIDEO_1_1_DURATION} seconds"
echo "Combined video (16:9) rendering time: ${COMBINED_VIDEO_16_9_DURATION} seconds"
echo "================================"
echo "Total time: $((RUNTIME_STAGE_DURATION + MODEL_STAGE_TOTAL_DURATION + PREP_DURATION + COMPILE_DURATION + SOLVE_DURATION + RENDER_TOP_VIEW_DURATION + RENDER_SLICE_FULL_HEIGHT_DURATION + RENDER_SLICE_REDUCED_HEIGHT_DURATION + TOP_VIEW_VIDEO_DURATION + SLICE_FULL_HEIGHT_VIDEO_DURATION + COMBINED_VIDEO_1_1_DURATION + COMBINED_VIDEO_16_9_DURATION)) seconds"
