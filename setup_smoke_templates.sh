#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 1. CPU Template (single allocation on c23mm)
CPU_TEMPLATE="${SCRIPT_DIR}/.smoke_cpu_template.sh"
cat <<'EOF' > "${CPU_TEMPLATE}"
#!/bin/zsh
#SBATCH --account=thes2181
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=00:10:00
#SBATCH --gres=none

source proper_slurm_job.sh
EOF
chmod +x "${CPU_TEMPLATE}"

# 2. GPU Template (hetjob: c23mm solver + c23g GPU inference)
GPU_TEMPLATE="${SCRIPT_DIR}/.smoke_gpu_template.sh"
cat <<'EOF' > "${GPU_TEMPLATE}"
#!/bin/zsh
#SBATCH --account=thes2181
#SBATCH --time=00:15:00

# Component 0: CPU solver
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G

#SBATCH hetjob

# Component 1: GPU inference
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G

source proper_slurm_job.sh
EOF
chmod +x "${GPU_TEMPLATE}"

echo "Templates created successfully."
