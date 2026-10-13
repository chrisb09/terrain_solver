#!/usr/bin/env bash
# Lightweight NIC traffic sampler: appends per-interface cumulative counters
# from /sys/class/net to a CSV at a fixed interval (default 1 s). Runs on each
# allocated node (via srun in het jobs, plain background otherwise). Deltas are
# computed offline in the analysis.
# Usage: nic_sampler.sh <output_csv> [interval_s]
set -u

out="$1"
interval="${2:-1}"
host="$(hostname -s)"

# Ensure the header exists (may race between nodes; duplicates are filtered in
# post-processing).
if [[ ! -f "$out" ]]; then
    echo "timestamp,host,iface,rx_bytes,tx_bytes,rx_packets,tx_packets" >> "$out" 2>/dev/null
fi

while :; do
    ts="$(date +%s.%N)"
    for statdir in /sys/class/net/*/statistics; do
        iface="${statdir%/statistics}"
        iface="${iface##*/}"
        [[ "$iface" == "lo" ]] && continue
        read -r rx  < "$statdir/rx_bytes"   2>/dev/null || continue
        read -r tx  < "$statdir/tx_bytes"   2>/dev/null
        read -r rxp < "$statdir/rx_packets" 2>/dev/null
        read -r txp < "$statdir/tx_packets" 2>/dev/null
        echo "$ts,$host,$iface,$rx,$tx,$rxp,$txp" >> "$out" 2>/dev/null
    done
    sleep "$interval"
done
