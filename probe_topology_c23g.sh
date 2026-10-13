#!/usr/bin/env bash
#SBATCH --job-name=probe_topo
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00
#SBATCH --output=logs/probe_topo_%j.txt
module load hwloc/2.7.1 >/dev/null 2>&1 || true
echo "host=$(hostname)"
echo "==== lstopo --distances ===="
lstopo --distances 2>&1
echo "==== sysfs node distances ===="
for n in /sys/devices/system/node/node*; do
  echo "$(basename "$n"): $(cat "$n/distance" 2>/dev/null)"
done
echo "==== GPU PCI -> numa_node ===="
for d in /sys/bus/pci/devices/*/; do
  cls=$(cat "$d/class" 2>/dev/null)
  if [[ "$cls" == 0x030000 || "$cls" == 0x030200 ]]; then
    echo "$(basename "$d") numa_node=$(cat "$d/numa_node" 2>/dev/null)"
  fi
done
echo "==== numactl --hardware ===="
numactl --hardware 2>/dev/null | head -30
