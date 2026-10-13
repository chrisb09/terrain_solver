#!/bin/bash
#SBATCH --job-name=smartsim-backend-smoke
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:35:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
python3 -B -c 'import os, sys; import gpu_refresh_campaign as r; g = next(g for g in r.groups() if g["id"] == "profile_24c1g_r1"); g["id"] = "smoke_smartsim_backend_24c1g"; g["cases"] = [c for c in g["cases"] if c[1] == "SMARTSIM" and c[2] in ("shared_c1", "shared_c3")]; assert len(g["cases"]) == 4; sys.exit(r.run(g, os.environ["GPU_REFRESH_CAMPAIGN_ENV"]))'
