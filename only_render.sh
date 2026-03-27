#!/bin/sh
sbatch --export=SKIP_RENDERING_ENV=0,SKIP_COMPILE_ENV=1 slurm_sbatch_mini_app.sh

# only render again