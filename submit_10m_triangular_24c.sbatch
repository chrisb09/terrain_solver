#!/bin/zsh
#SBATCH --job-name=circle_10m_triangular_24c
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --output=logs/mini_app_%j.out

SOLVER_STEP_LOG="logs/mini_app_solver_steps_${SLURM_JOB_ID}.log"

CUSTOM_JOB_NAME_SUFFIX=""

job_name_template='circle_r${RADIUS}_d${INIT_DEPTH}_s${TOTAL_STEPS}_${SAVE_MODE}_${TARGET_WIDTH}x${TARGET_HEIGHT}_${_partition}_${_nodes}n_${_ntasks_per_node}t_${_cpus_per_task}c__${IO_MODE}_${MPI_SYNC_MODE}${CUSTOM_JOB_NAME_SUFFIX}'

INIT_MODE="circle"
RADIUS=300
INIT_DEPTH=300
TOTAL_STEPS=10000000
SAVE_EVERY=1
SAVE_MODE="triangular"
TRIANGULAR_SCALE=2
TARGET_WIDTH=2160
TARGET_HEIGHT=1080
CHUNK_SIZE=60

_partition="${SLURM_JOB_PARTITION:-c23ms}"
_nodes="${SLURM_JOB_NUM_NODES:-1}"
_ntasks_per_node="${SLURM_NTASKS_PER_NODE:-24}"
_cpus_per_task="${SLURM_CPUS_PER_TASK:-1}"

MPI_RANKS=24
IO_MODE="rank0_gather"
MPI_SYNC_MODE="none"
HDF5_XFER_MODE="independent"

JOB_NAME=$(eval echo "$job_name_template")
if [[ "${TRIANGULAR_SCALE}" -gt 1 ]]; then
  JOB_NAME="${JOB_NAME}_ts${TRIANGULAR_SCALE}"
fi
echo "Using job name: ${JOB_NAME}"

RANK_GRID_X=0
RANK_GRID_Z=0
OVERWRITE_OUTPUT=1
CREATE_NEW_H5=1
RUN_SOLVER=1

SKIP_COMPILE=1
SKIP_RENDERING=0
VIDEO_FPS=30
REDUCED_HEIGHT=135
HEIGHT_MIN=40

RENDER_CPUS=24
MAX_THREADS=24
FFMPEG_THREADS=16

set -euo pipefail

cd /hpcwork/ro092286/smartsim/ || exit

source /hpcwork/ro092286/smartsim/install.sh
source /hpcwork/ro092286/smartsim/set_env_claix23_cuda12.3.sh

export OMPI_MCA_pmix="^s1,s2"
export OMPI_MCA_io="romio321"
export OMPI_MCA_fcoll="^dynamic_gen2"

module -t list

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
PY_ENV="/hpcwork/ro092286/smartsim/python/smartsim_cpu"
EXTERNAL_DIR="/hpcwork/thes2181/mini_app"
INPUT_IMAGE="${MINI_APP_DIR}/old/World_elevation_map.png"
PREP_H5="${EXTERNAL_DIR}/${JOB_NAME}/world_init.h5"
TRAJ_H5="${EXTERNAL_DIR}/${JOB_NAME}/world_trajectory.h5"

if [[ ! -d "${EXTERNAL_DIR}" ]]; then
  mkdir -p "${EXTERNAL_DIR}"
fi

if [[ ! -d "${EXTERNAL_DIR}/${JOB_NAME}" ]]; then
  mkdir -p "${EXTERNAL_DIR}/${JOB_NAME}"
fi

if [[ -f "${PY_ENV}/bin/activate" ]]; then
  source "${PY_ENV}/bin/activate" || { echo "Failed to activate Python environment at ${PY_ENV}"; exit 1; }
else
  echo "Python environment not found at ${PY_ENV}"
  exit 1
fi

cd "${MINI_APP_DIR}" || exit

START_TIME=$(date +%s)

echo "Preparing initial condition in ${PREP_H5}..."
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

if [[ ! -f "solver_cpp/build/terrain_solver" ]]; then
  echo "Compiling solver..."
  COMPILE_START_TIME=$(date +%s)
  rm -rf "solver_cpp/build"
  mkdir -p "solver_cpp/build"
  cmake -S solver_cpp -B solver_cpp/build -DUSE_CPP_ML_INTERFACE=OFF
  cmake --build solver_cpp/build -j 8
  COMPILE_END_TIME=$(date +%s)
  COMPILE_DURATION=$((COMPILE_END_TIME - COMPILE_START_TIME))
  echo "Compilation time: ${COMPILE_DURATION} seconds"
else
  echo "Solver binary exists. Skipping compilation."
  COMPILE_DURATION=0
fi

SOLVE_START_TIME=$(date +%s)

RANK_GRID_ARGS=()
if [[ "${RANK_GRID_X}" -gt 0 ]]; then
  RANK_GRID_ARGS+=(--rank-grid-x "${RANK_GRID_X}")
fi
if [[ "${RANK_GRID_Z}" -gt 0 ]]; then
  RANK_GRID_ARGS+=(--rank-grid-z "${RANK_GRID_Z}")
fi

OVERWRITE_ARG=(--overwrite-output)

mkdir -p logs
setopt pipefail

echo "Launching solver for ${TOTAL_STEPS} steps (save-mode: ${SAVE_MODE}, scale: ${TRIANGULAR_SCALE})..."

srun --ntasks-per-node "${_ntasks_per_node}" \
  --cpus-per-task 1 \
  ./solver_cpp/build/terrain_solver \
  --ml-interface none \
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
  --write-surface 2>&1 | tee "${SOLVER_STEP_LOG}"

SRUN_STATUS=${pipestatus[1]}
unsetopt pipefail
if [[ "${SRUN_STATUS}" -ne 0 ]]; then
  echo "Solver failed with exit code ${SRUN_STATUS}"
  exit "${SRUN_STATUS}"
fi

SOLVE_END_TIME=$(date +%s)
SOLVE_DURATION=$((SOLVE_END_TIME - SOLVE_START_TIME))
echo "Solving time: ${SOLVE_DURATION} seconds"

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
  echo "Post-processing on batch host $(hostname) with RENDER_CPUS=${RENDER_CPUS}, FFMPEG_THREADS=${FFMPEG_THREADS}, VIDEO_FPS=${VIDEO_FPS}"

  python3 render.py \
    --input-hdf5 "${TRAJ_H5}" \
    --step -1 \
    --output-png "${EXTERNAL_DIR}/${JOB_NAME}/world_last_step.png"

  rm -rf "${EXTERNAL_DIR}/${JOB_NAME}/rendered_frames"
  RENDER_TOP_VIEW_START_TIME=$(date +%s)

  echo "Rendering top view frames..."
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

  echo "Rendering slice full height frames..."
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

  echo "Rendering slice reduced height frames..."
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

  echo "Switching to FFmpeg module environment..."
  module unload "${NORMAL_GCC_MODULE}" || { echo "Failed to unload GCC module ${NORMAL_GCC_MODULE}"; exit 1; }
  module load "${FFMPEG_GCC_MODULE}" || { echo "Failed to load FFmpeg GCCcore module ${FFMPEG_GCC_MODULE}"; exit 1; }
  module load "${FFMPEG_VERSION}" || { echo "Failed to load FFmpeg module ${FFMPEG_VERSION}"; exit 1; }

  cd "${EXTERNAL_DIR}/${JOB_NAME}" || exit

  TOP_VIEW_VIDEO_START_TIME=$(date +%s)
  echo "Encoding top_view.mp4 @ ${VIDEO_FPS} fps..."
  ffmpeg \
    -framerate "${VIDEO_FPS}" \
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
  echo "Encoding slice_full_height.mp4 @ ${VIDEO_FPS} fps..."
  ffmpeg \
    -framerate "${VIDEO_FPS}" \
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

  if [[ ! -x "make_dual_view_video.sh" ]]; then
    echo "Error: make_dual_view_video.sh not found or not executable in ${MINI_APP_DIR}"
    exit 1
  fi

  COMBINED_VIDEO_1_1_START_TIME=$(date +%s)
  echo "Encoding dual_1x1.mp4 @ ${VIDEO_FPS} fps..."
  ./make_dual_view_video.sh \
    --top-pattern "${EXTERNAL_DIR}/${JOB_NAME}/rendered_frames/frame_*.png" \
    --slice-pattern "${EXTERNAL_DIR}/${JOB_NAME}/rendered_slices/frame_*.png" \
    --output "${EXTERNAL_DIR}/${JOB_NAME}/dual_1x1.mp4" \
    --fps "${VIDEO_FPS}" \
    --threads ${FFMPEG_THREADS}

  COMBINED_VIDEO_1_1_END_TIME=$(date +%s)
  COMBINED_VIDEO_1_1_DURATION=$((COMBINED_VIDEO_1_1_END_TIME - COMBINED_VIDEO_1_1_START_TIME))
  echo "Combined video (1:1) rendering time: ${COMBINED_VIDEO_1_1_DURATION} seconds"

  COMBINED_VIDEO_16_9_START_TIME=$(date +%s)
  echo "Encoding dual_16x9.mp4 @ ${VIDEO_FPS} fps..."
  ./make_dual_view_video.sh \
    --top-pattern "${EXTERNAL_DIR}/${JOB_NAME}/rendered_frames/frame_*.png" \
    --slice-pattern "${EXTERNAL_DIR}/${JOB_NAME}/rendered_slices_reduced_height/frame_*.png" \
    --output "${EXTERNAL_DIR}/${JOB_NAME}/dual_16x9.mp4" \
    --width ${TARGET_WIDTH} \
    --top-height ${TARGET_HEIGHT} \
    --bottom-height ${REDUCED_HEIGHT} \
    --fps "${VIDEO_FPS}" \
    --threads ${FFMPEG_THREADS}

  COMBINED_VIDEO_16_9_END_TIME=$(date +%s)
  COMBINED_VIDEO_16_9_DURATION=$((COMBINED_VIDEO_16_9_END_TIME - COMBINED_VIDEO_16_9_START_TIME))
  echo "Combined video (16:9) rendering time: ${COMBINED_VIDEO_16_9_DURATION} seconds"

  module unload "${FFMPEG_VERSION}" "${FFMPEG_GCC_MODULE}" || { echo "Failed to unload FFmpeg/GCCcore modules"; exit 1; }
  module load "${NORMAL_GCC_MODULE}" || { echo "Failed to reload original GCC module ${NORMAL_GCC_MODULE}"; exit 1; }
fi

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
  echo "VIDEO_FPS: ${VIDEO_FPS}"
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
  echo "SOLVER_STEP_LOG: ${SOLVER_STEP_LOG}"
  echo ""
  echo "Timings (seconds):"
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
  TOTAL_DURATION=$(( PREP_DURATION + COMPILE_DURATION + SOLVE_DURATION + RENDER_TOP_VIEW_DURATION + RENDER_SLICE_FULL_HEIGHT_DURATION + RENDER_SLICE_REDUCED_HEIGHT_DURATION + TOP_VIEW_VIDEO_DURATION + SLICE_FULL_HEIGHT_VIDEO_DURATION + COMBINED_VIDEO_1_1_DURATION + COMBINED_VIDEO_16_9_DURATION ))
  echo "Total: ${TOTAL_DURATION}"
} > "${TIMING_FILE}"

echo "Job complete. Timings and parameters saved to ${TIMING_FILE}"
