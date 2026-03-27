#!/bin/zsh

# Script to submit multiple mini_app jobs with different configurations.
# Uses sbatch CLI overrides and _ENV variables — no script copies needed.

# Configurations to run: each entry is "nodes ntasks_per_node total_steps"
# Format: nodes=N, ntasks_per_node=M gives total_ntasks=N*M (evenly distributed by definition)
CONFIGS=(
#  "1  1  1000000"
#  "1  2  1000000"
#  "1  4  1000000"
#  "1  6  1000000"
#  "1  8  1000000"
#  "1  12 1000000"
#  "1  16 1000000"
#  "1  24 1000000"
#  "1  32 1000000"
#  "1  48 1000000"
#  "1  64 1000000"
#  "1  72 1000000"
#  "1  96 1000000"
## repeat those because time limit
#    "1  1  1000000"    # 1 node × 1 task/node = 1 total ✓
#    "1  2  1000000"    # 1 node × 2 tasks/node = 2 total ✓
## repeat these due to faulty runs
#    "2  8  1000000"    # 2 nodes × 8 tasks/node = 16 total ✓
#    "2  16 1000000"    # 2 nodes × 16 tasks/node = 32 total ✓
#    "2  24 1000000"    # 2 nodes × 24 tasks/node = 48 total ✓
##
#    "3  8  1000000"    # 3 nodes × 8 tasks/node = 24 total ✓
#    "3  16 1000000"    # 3 nodes × 16 tasks/node = 48 total ✓
#    "3  24 1000000"    # 3 nodes × 24 tasks/node = 72 total ✓
#    "4  8  1000000"    # 4 nodes × 8 tasks/node = 32 total ✓
#    "4  16 1000000"    # 4 nodes × 16 tasks/node = 64 total ✓
#    "4  24 1000000"    # 4 nodes × 24 tasks/node = 96 total ✓
#    "8  8  1000000"    # 8 nodes × 8 tasks/node = 64 total ✓
#    "8  16 1000000"    # 8 nodes × 16 tasks/node = 128 total ✓
#    "8  24 1000000"    # 8 nodes × 24 tasks/node = 192 total ✓
#    "12 8  1000000"    # 12 nodes × 8 tasks/node = 96 total ✓
#    "12 16 1000000"    # 12 nodes × 16 tasks/node = 192 total ✓
#    "12 24 1000000"    # 12 nodes × 24 tasks/node = 288 total ✓
##
#  "1  96 1000000"
  "2  96 1000000"
  "3  96 1000000"
  "4  96 1000000"
#  "5  96 1000000"
#  "6  96 1000000"
#  "7  96 1000000"
#  "8  96 1000000"
)

# Two IO_MODE variants to run for each configuration
IO_MODES=("rank0_gather") # "parallel_hdf5","rank0_gather"

# Base script to submit directly
BASE_SCRIPT="slurm_sbatch_mini_app.sh"

# Check if base script exists
if [ ! -f "$BASE_SCRIPT" ]; then
    echo "Error: $BASE_SCRIPT not found!"
    exit 1
fi

#CUSTOM_JOB_NAME_SUFFIX_ENV=

# Loop over all configurations and IO modes
for config in "${CONFIGS[@]}"; do
    nodes=$(echo "$config" | awk '{print $1}')
    ntasks_per_node=$(echo "$config" | awk '{print $2}')
    total_steps=$(echo "$config" | awk '{print $3}')
    
    # Calculate total ntasks (nodes × ntasks_per_node)
    total_ntasks=$((nodes * ntasks_per_node))

    for io_mode in "${IO_MODES[@]}"; do
        # we require at least 4 ntasks to run with parallel HDF5, so skip invalid configs
        if [[ "$io_mode" == "parallel_hdf5" && "$total_ntasks" -lt 4 ]]; then
            echo "Skipping config nodes=${nodes}, ntasks_per_node=${ntasks_per_node} (total=${total_ntasks}) for IO_MODE=${io_mode} (requires at least 4 ntasks)"
            continue
        fi
        echo "Submitting: nodes=${nodes}, ntasks=${ntasks_per_node} per node (total ${total_ntasks}), total_steps=${total_steps}, io_mode=${io_mode}"

        sbatch \
            --nodes="${nodes}" \
            --ntasks-per-node="${ntasks_per_node}" \
            --export="SKIP_COMPILE_ENV=1,TOTAL_STEPS_ENV=${total_steps},IO_MODE_ENV=${io_mode},OVERWRITE_OUTPUT_ENV=1,OVERWRITE_JOB_NAME_ENV=1" \
            "$BASE_SCRIPT"

        echo "Submitted."
    done
done

echo "All jobs submitted!"
