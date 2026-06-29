#!/bin/bash

# if env var SKIP_COMPILE is set to 1, skip compilation and just run the job
if [ "${SKIP_COMPILE:-0}" -ne 0 ]; then
    echo "Skipping compilation as SKIP_COMPILE_ENV is set to 1"
else
    echo "Compiling the application..."
    ./slurm_build.sh
    if [ $? -ne 0 ]; then
        echo "Compilation failed. Exiting."
        exit 1
    fi
fi

MODEL="benchmark_giant_mlp"

# Dependency tracking
PREV_JID=""

# Helper function to submit and attach watcher with dependency
submit_sequential() {
    local job_name="$1"
    local export_vars="$2"
    local further_sbatch_args="${@:3}"  # Capture any additional arguments
    local sbatch_cmd="sbatch --parsable --job-name=${job_name} --export=${export_vars},MODEL_NAME_ENV=${MODEL} ${further_sbatch_args}"
    
    if [ -n "$PREV_JID" ]; then
        sbatch_cmd="${sbatch_cmd} --dependency=afterany:${PREV_JID}"
    fi
    
    # Run sbatch and capture ID
    local jid
    jid=$(eval ${sbatch_cmd} ./proper_slurm_job.sh)
    
    if [ $? -eq 0 ]; then
        echo "Submitted ${job_name}: ${jid}"
        if [ -n "$PREV_JID" ]; then echo "  (Depends on ${PREV_JID})"; fi
        
        # Register the watcher for this job
        ~/scripts/swatch --jobid "${jid}"
        
        # Update dependency for next job
        PREV_JID="${jid}"
    else
        echo "Failed to submit ${job_name}"
    fi
}

# We deactivate some jobs by adding ### to their line
PREFIX="GPU"

# 1. Direct SmartSim
submit_sequential "SMARTSIM_${PREFIX}_terrain_solver" "USE_SMARTSIM=1,FORCE_TERRAIN_UPLOAD_EACH_STEP_ENV=0,SKIP_COMPILE_ENV=1,OVERWRITE_OUTPUT_ENV=1"

# 2. Direct SmartSim with Force Terrain Upload
submit_sequential "SMARTSIM_${PREFIX}_force_terrain" "USE_SMARTSIM=1,FORCE_TERRAIN_UPLOAD_EACH_STEP_ENV=1,SKIP_COMPILE_ENV=1,OVERWRITE_OUTPUT_ENV=1"

# 3. SmartSim via CPP-ML-Interface
submit_sequential "CMI_smartsim_${PREFIX}_terrain_solver" "USE_SMARTSIM=0,SKIP_COMPILE_ENV=1,OVERWRITE_OUTPUT_ENV=1"

# 4. AIX via CPP-ML-Interface
submit_sequential "CMI_aix_${PREFIX}_terrain_solver" "USE_SMARTSIM=0,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,SKIP_COMPILE_ENV=1,OVERWRITE_OUTPUT_ENV=1"

# 5. PhyDLL via CPP-ML-Interface (C++ DL Client)
submit_sequential "CMI_phydll_${PREFIX}_terrain_solver" "USE_SMARTSIM=0,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,SKIP_COMPILE_ENV=1,OVERWRITE_OUTPUT_ENV=1"

# 6. PhyDLL via CPP-ML-Interface (Python DL Client)
submit_sequential "CMI_phydll_py_${PREFIX}_terrain_solver" "USE_SMARTSIM=0,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,USE_PYTHON_DL_CLIENT=1,SKIP_COMPILE_ENV=1,OVERWRITE_OUTPUT_ENV=1"
