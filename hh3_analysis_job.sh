#!/bin/zsh
# HH3 final analysis stage 2: config comparison (needs pandas), overview tables
# and scaling plots from the cached metrics. devel job so login-node CPU
# limits / tool timeouts cannot interrupt the post-processing.
#SBATCH --job-name=hh3_analyze2
#SBATCH --partition=devel
#SBATCH --account=default
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/hh3_analysis_%j.txt

set -uo pipefail
cd /hpcwork/ro092286/smartsim/mini_app
mkdir -p logs

# pandas/matplotlib live in the SmartSim cuda-12 venv
source /hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/activate

echo "=== HH3 analysis job start $(date) ==="
python3 -c "import pandas, matplotlib; print('env ok:', pandas.__version__, matplotlib.__version__)"

echo "--- cross-config comparison (pandas) ---"
CPP_ML_DIR=/hpcwork/ro092286/smartsim/CPP-ML-Interface
python3 "${CPP_ML_DIR}/scripts/plot_config_comparison.py" results/hethires3 \
    --output results/hethires3/config_comparison.png || echo "[WARN] config comparison failed"
echo "config comparison rc=$?"

echo "--- overview tables (cached timelines) ---"
python3 build_hh3_overview.py
echo "overview rc=$?"

echo "--- scaling plots ---"
python3 plot_hh3_scaling.py
echo "scaling rc=$?"

echo "=== HH3 analysis job done $(date) ==="
