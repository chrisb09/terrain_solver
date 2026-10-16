#!/usr/bin/env bash
# Submit with --dependency=afterok:<completed-variant-builder-job>.
#SBATCH --job-name=submit-missing65
#SBATCH --account=thes2181
#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/submit_missing65_%j.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/submit_missing65_%j.err
set -euo pipefail
export GPU_REFRESH_BUILD_JOB_ID_ENV=4938224
export GPU_REFRESH_MINI_ENV=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
export PYTHONPATH="$GPU_REFRESH_MINI_ENV${PYTHONPATH:+:$PYTHONPATH}"
python3 -B -c '
import os
import gpu_refresh_campaign as campaign
campaign.submit("--submit", "missing65_20261015_" + os.environ["SLURM_JOB_ID"],
                campaign.missing65_profile_groups(), notifier="snotify", smoke_first=True)
'
