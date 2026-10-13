#!/usr/bin/env bash
#SBATCH --job-name=probe_bind
#SBATCH --account=thes2181
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/probe_bind_%j.txt
# Compare Slurm rank->core binding for the default vs explicit block:block
# distribution (16 tasks, one node), and print the NUMA node of each core.
set -u
echo "host=$(hostname)"
for mode in default block; do
  echo "==== mode=${mode} ===="
  if [[ "${mode}" == "block" ]]; then
    srun --distribution=block:block -n16 bash -c 'echo "$SLURM_PROCID $(grep Cpus_allowed_list /proc/self/status | awk "{print \$2}")"'
  else
    srun -n16 bash -c 'echo "$SLURM_PROCID $(grep Cpus_allowed_list /proc/self/status | awk "{print \$2}")"'
  fi
done
echo "==== cpu -> numa map ===="
for n in /sys/devices/system/node/node*; do
  echo "$(basename "$n"): $(cat "$n/cpulist")"
done
