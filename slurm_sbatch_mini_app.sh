#!/bin/zsh
#SBATCH --job-name=tiny_mini_app_solver
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1

##SBATCH --time=48:00:00
##SBATCH --account=thes2181
##SBATCH --partition=c23mm


#SBATCH --time=01:00:00
#SBATCH --partition=devel

#SBATCH --output=logs/mini_app_%j.out

SOLVER_STEP_LOG="logs/mini_app_solver_steps_${SLURM_JOB_ID}.log"


##SBATCH --exclusive
##SBATCH --mem=20G

# Some parameters for the mini-app, adjust as needed

CUSTOM_JOB_NAME_SUFFIX="" # optional suffix to add to the job name for easier identification in job queues and output files, e.g. "_test" or "_render_only"

job_name_template='circle_r${RADIUS}_d${INIT_DEPTH}_s${TOTAL_STEPS}_${SAVE_MODE}_${TARGET_WIDTH}x${TARGET_HEIGHT}_${_partition}_${_nodes}n_${_ntasks_per_node}t_${_cpus_per_task}c__${IO_MODE}_${MPI_SYNC_MODE}${CUSTOM_JOB_NAME_SUFFIX}'


INIT_MODE="circle" # circle, square, uniform
RADIUS=300 # radius for circle and square initial conditions
INIT_DEPTH=300 # initial depth of water
#TOTAL_STEPS=10000000
TOTAL_STEPS=10000
#TOTAL_STEPS=10000
SAVE_EVERY=1
#SAVE_MODE="triangular"        # periodic, triangular
SAVE_MODE="periodic"          # periodic, triangular
TRIANGULAR_SCALE=1          # scale factor for triangular save schedule (>=1, only used when SAVE_MODE=triangular)
#TARGET_WIDTH=2160
#TARGET_HEIGHT=1080
TARGET_WIDTH=2160
TARGET_HEIGHT=1080
#CHUNK_SIZE=60
CHUNK_SIZE=60
# We also include some slurm parameters as part of the job name so running different configurations in parallel is easier to manage.
_partition="${SLURM_JOB_PARTITION:-}"
_nodes="${SLURM_JOB_NUM_NODES:-}"
_ntasks_per_node="${SLURM_NTASKS_PER_NODE:-}"
_cpus_per_task="${SLURM_CPUS_PER_TASK:-}"

#MPI_RANKS=${SLURM_NTASKS:-1}
MPI_RANKS=$(( $SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE ))
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
REDUCED_HEIGHT=135 # ensure that TARGET_WIDTH / (TARGET_HEIGHT + REDUCED_HEIGHT) is close to 16:9 aspect ratio for the combined video
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

set -euo pipefail

cd /hpcwork/ro092286/smartsim/ || exit

source /hpcwork/ro092286/smartsim/install.sh
source /hpcwork/ro092286/smartsim/set_env_claix23_cuda12.3.sh

# Suppress OpenMPI PMI s1/s2 component probe warnings on systems without libpmi/libpmi2.
export OMPI_MCA_pmix="^s1,s2"
# Work around OpenMPI OMPIO collective I/O instability seen with parallel HDF5 at high rank counts.
# Force ROMIO backend for MPI-IO instead of OMPIO (stack traces showed mca_io_ompio/mca_fcoll_dynamic_gen2).
export OMPI_MCA_io="romio321"
# Also disable the problematic OMPIO collective algorithm if OMPIO gets selected by accident.
export OMPI_MCA_fcoll="^dynamic_gen2"

module -t list

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
PY_ENV="/hpcwork/ro092286/smartsim/python/smartsim_cpu"
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

if [[ -f "${PY_ENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${PY_ENV}/bin/activate" || { echo "Failed to activate Python environment at ${PY_ENV}"; exit 1; }
else
  echo "Python environment not found at ${PY_ENV}"
  exit 1
fi

cd "${MINI_APP_DIR}" || exit

CREATE_NEW_H5=1
RUN_SOLVER=1

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

# Save timestamp to measure time for preparation and solving
START_TIME=$(date +%s)

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

  # For multi-node jobs, use --distribution=block to evenly spread tasks across nodes.
  # This prevents task desynchronization at shutdown (especially in rank0_gather I/O mode).
  SRUN_DIST="--distribution=block"
  if [[ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]]; then
    SRUN_DIST="" # Single node: distribution doesn't matter
  fi

  mkdir -p logs
  setopt pipefail
  srun --ntasks-per-node "${_ntasks_per_node}" \
    --cpus-per-task 1 \
    ${SRUN_DIST} \
    ./solver_cpp/build/terrain_solver \
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
else
  echo "Skipping solver execution since trajectory is already complete."
  COMPILE_DURATION=0
  SOLVE_DURATION=0
fi

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
    -framerate 60 \
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
    -framerate 60 \
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
    --fps 60 \
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
    --fps 60 \
    --threads ${FFMPEG_THREADS}

  COMBINED_VIDEO_16_9_END_TIME=$(date +%s)
  COMBINED_VIDEO_16_9_DURATION=$((COMBINED_VIDEO_16_9_END_TIME - COMBINED_VIDEO_16_9_START_TIME))
  echo "Combined video (16:9) rendering time: ${COMBINED_VIDEO_16_9_DURATION} seconds"


  module unload "${FFMPEG_VERSION}" "${FFMPEG_GCC_MODULE}" || { echo "Failed to unload FFmpeg/GCCcore modules"; exit 1; }
  module load "${NORMAL_GCC_MODULE}" || { echo "Failed to reload original GCC module ${NORMAL_GCC_MODULE}"; exit 1; }

  NORMAL_GCC_MODULE=$(module -t list 2>&1 | grep "^GCC/")

  echo "Now using ${NORMAL_GCC_MODULE} as the GCC module after loading FFmpeg"

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
  echo "Total: $((PREP_DURATION + COMPILE_DURATION + SOLVE_DURATION + RENDER_TOP_VIEW_DURATION + RENDER_SLICE_FULL_HEIGHT_DURATION + RENDER_SLICE_REDUCED_HEIGHT_DURATION + TOP_VIEW_VIDEO_DURATION + SLICE_FULL_HEIGHT_VIDEO_DURATION + COMBINED_VIDEO_1_1_DURATION + COMBINED_VIDEO_16_9_DURATION))"
} > "${TIMING_FILE}"

echo "Timing and parameters saved to: ${TIMING_FILE}"


echo ""
echo "================================"

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
echo "Total time: $((PREP_DURATION + COMPILE_DURATION + SOLVE_DURATION + RENDER_TOP_VIEW_DURATION + RENDER_SLICE_FULL_HEIGHT_DURATION + RENDER_SLICE_REDUCED_HEIGHT_DURATION + TOP_VIEW_VIDEO_DURATION + SLICE_FULL_HEIGHT_VIDEO_DURATION + COMBINED_VIDEO_1_1_DURATION + COMBINED_VIDEO_16_9_DURATION)) seconds"
