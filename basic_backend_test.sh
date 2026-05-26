#!/bin/bash
./build.sh

# Dependency tracking
PREV_JID=""

# Helper function to submit and attach watcher with dependency
submit_sequential() {
    local job_name="$1"
    local export_vars="$2"
    local sbatch_cmd="sbatch --parsable --job-name=${job_name} --export=${export_vars}"
    
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

# 1. Direct SmartSim
submit_sequential "SMARTSIM_terrain_solver" "USE_SMARTSIM=1,SKIP_COMPILE_ENV=1"

# 2. SmartSim via CPP-ML-Interface
submit_sequential "CPP_ML_INTERFACE_smartsim_terrain_solver" "USE_SMARTSIM=0,SKIP_COMPILE_ENV=1"

# 3. SmartSim with Force Terrain Upload
submit_sequential "CPP_ML_INTERFACE_smartsim_force_terrain" "ALL,USE_SMARTSIM=0,FORCE_TERRAIN_UPLOAD_EACH_STEP_ENV=1,SKIP_COMPILE_ENV=1"

# 4. AIX via CPP-ML-Interface
submit_sequential "CPP_ML_INTERFACE_aix_terrain_solver" "USE_SMARTSIM=0,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,SKIP_COMPILE_ENV=1"

# 5. PhyDLL via CPP-ML-Interface
submit_sequential "CPP_ML_INTERFACE_phydll_terrain_solver" "USE_SMARTSIM=0,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,SKIP_COMPILE_ENV=1"
