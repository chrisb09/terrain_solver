#!/usr/bin/env python3
"""
analyze_traffic_comparison.py
==============================
Compares analytical key-placement traffic predictions against empirical measurements
for SmartSim Clustered (single shared keyspace) vs Standalone (per-node independent DBs).

Usage:
  python3 analyze_traffic_comparison.py \
      --clustered-log logs/clustered_solver.log \
      --standalone-log logs/standalone_solver.log \
      --clustered-net logs/clustered_net.txt \
      --standalone-net logs/standalone_net.txt \
      --clustered-cubex scorep_clustered/profile.cubex \
      --standalone-cubex scorep_standalone/profile.cubex \
      --ranks 96 --shards 4 --width 3840 --height 2160 --steps 10
"""

import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


def crc16_xmodem(data: bytes) -> int:
    """Computes CRC16-XMODEM (CRC-16/CCITT-FALSE) as specified by Redis Cluster."""
    crc = 0x0000
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def redis_slot(key: str) -> int:
    """Extracts hash tag {...} if present and returns hash slot in [0, 16383]."""
    start = key.find("{")
    if start != -1:
        end = key.find("}", start + 1)
        if end != -1 and end > start + 1:
            key = key[start + 1 : end]
    return crc16_xmodem(key.encode("utf-8")) % 16384


def redis_shard(key: str, n_shards: int = 4) -> int:
    """Maps hash slot to shard index [0..n_shards-1] assuming contiguous partition."""
    return redis_slot(key) // (16384 // n_shards)


def compute_analytical_predictions(ranks: int, shards: int, width: int, height: int, steps: int):
    """
    Computes theoretical key placement, copy counts, and predicted byte volumes
    for WaterCNN (2 inputs: water, terrain; 1 output: y) on grid width x height.
    """
    # Domain decomposition
    cells_total = width * height
    cells_per_rank = cells_total // ranks

    # Tensor bytes per rank per ML call
    # Input 0 (water):   cells * 9 floats * 4 bytes
    # Input 1 (terrain): cells * 9 floats * 4 bytes
    # Output 0 (y):      cells * 1 float  * 4 bytes
    in0_bytes_per_rank = cells_per_rank * 9 * 4
    in1_bytes_per_rank = cells_per_rank * 9 * 4
    out0_bytes_per_rank = cells_per_rank * 1 * 4

    in0_mb_step = (in0_bytes_per_rank * ranks) / (1024 * 1024)
    in1_mb_step = (in1_bytes_per_rank * ranks) / (1024 * 1024)
    out0_mb_step = (out0_bytes_per_rank * ranks) / (1024 * 1024)

    # Shard mapping analysis for each rank
    in0_shards = []
    in1_shards = []
    out0_shards = []
    in1_same_as_in0 = 0
    out0_same_as_in0 = 0

    for r in range(ranks):
        k0 = f"input_{r}_0"
        k1 = f"input_{r}_1"
        ko = f"output_{r}_0"

        s0 = redis_shard(k0, shards)
        s1 = redis_shard(k1, shards)
        so = redis_shard(ko, shards)

        in0_shards.append(s0)
        in1_shards.append(s1)
        out0_shards.append(so)

        if s1 == s0:
            in1_same_as_in0 += 1
        if so == s0:
            out0_same_as_in0 += 1

    # Number of ML steps: even steps from 1 to steps
    # E.g. steps=10 -> steps 2, 4, 6, 8, 10 = 5 ML steps (1 warm, 4 steady)
    ml_steps = steps // 2
    warm_ml_steps = 1
    steady_ml_steps = max(0, ml_steps - 1)

    # Base payload per ML step (water input + output; terrain pre-uploaded or uploaded)
    # Case A: Terrain pre-uploaded once at step 0:
    base_step_preloaded_mb = in0_mb_step + out0_mb_step
    # Case B: Terrain uploaded every step:
    base_step_full_mb = in0_mb_step + in1_mb_step + out0_mb_step

    # Extra traffic in Clustered run_model per ML step:
    # 2 * in0 (redundant copy) + 2 * in1 (cross-shard copy) + 2 * out0 (cross-shard copy)
    extra_run_model_mb = 2 * in0_mb_step + 2 * in1_mb_step + 2 * out0_mb_step

    return {
        "ranks": ranks,
        "shards": shards,
        "cells_per_rank": cells_per_rank,
        "ml_steps": ml_steps,
        "warm_ml_steps": warm_ml_steps,
        "steady_ml_steps": steady_ml_steps,
        "target_shard_counts": dict(Counter(in0_shards)),
        "in0_on_target": ranks,
        "in1_on_target": in1_same_as_in0,
        "out0_on_target": out0_same_as_in0,
        "in0_mb_step": in0_mb_step,
        "in1_mb_step": in1_mb_step,
        "out0_mb_step": out0_mb_step,
        "base_step_preloaded_mb": base_step_preloaded_mb,
        "base_step_full_mb": base_step_full_mb,
        "extra_run_model_mb": extra_run_model_mb,
        "clustered_step_preloaded_mb": base_step_preloaded_mb + extra_run_model_mb,
        "clustered_step_full_mb": base_step_full_mb + extra_run_model_mb,
    }


def parse_net_dev_snapshot(net_file: Path, iface: str = "lo"):
    """Parses before/after /proc/net/dev lines recorded in net_file for given interface."""
    if not net_file or not net_file.exists():
        return None
    text = net_file.read_text(encoding="utf-8", errors="replace")
    # Expected format:
    # START <iface>: rx_bytes rx_packets ... tx_bytes tx_packets ...
    # END <iface>: rx_bytes rx_packets ... tx_bytes tx_packets ...
    start_match = re.search(rf"START\s+{iface}:\s*(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)", text)
    end_match = re.search(rf"END\s+{iface}:\s*(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)", text)
    if start_match and end_match:
        s_rx, s_rxp, s_tx, s_txp = map(int, start_match.groups())
        e_rx, e_rxp, e_tx, e_txp = map(int, end_match.groups())
        return {
            "rx_mb": (e_rx - s_rx) / (1024 * 1024),
            "tx_mb": (e_tx - s_tx) / (1024 * 1024),
            "rx_packets": e_rxp - s_rxp,
            "tx_packets": e_txp - s_txp,
        }
    return None


def parse_solver_log(log_path: Path):
    """Extracts step timings, NET_USAGE, and ML_TRAFFIC from solver log."""
    if not log_path or not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")

    # NET_USAGE
    net_usage = {}
    for m in re.finditer(r"NET_USAGE\s+if=(\w+)\s+rx_mb=([0-9.]+)\s+tx_mb=([0-9.]+)", text):
        iface, rx, tx = m.group(1), float(m.group(2)), float(m.group(3))
        net_usage[iface] = {"rx_mb": rx, "tx_mb": tx}

    # ML_TRAFFIC
    ml_traffic = {}
    m_traf = re.search(r"ML_TRAFFIC\s+input_mb=([0-9.]+)\s+output_mb=([0-9.]+)\s+preload_mb=([0-9.]+)", text)
    if m_traf:
        ml_traffic = {
            "input_mb": float(m_traf.group(1)),
            "output_mb": float(m_traf.group(2)),
            "preload_mb": float(m_traf.group(3)),
        }

    # Step timings: STEP_TIMING step=X solver=ML step_ms=Y
    ml_steps = {}
    for m in re.finditer(r"STEP_TIMING\s+step=(\d+)\s+solver=ML\s+step_ms=([0-9.]+)", text):
        step_num = int(m.group(1))
        ms = float(m.group(2))
        ml_steps[step_num] = ms

    # Total run model time
    tot_run_model = 0.0
    m_tot = re.search(r"Total run model time:\s*([0-9.]+)\s*s", text)
    if m_tot:
        tot_run_model = float(m_tot.group(1))

    warm_ms = ml_steps.get(2, 0.0)
    steady_ms_list = [v for k, v in ml_steps.items() if k > 2]
    steady_mean_ms = (sum(steady_ms_list) / len(steady_ms_list)) if steady_ms_list else 0.0

    return {
        "net_usage": net_usage,
        "ml_traffic": ml_traffic,
        "ml_steps": ml_steps,
        "warm_ms": warm_ms,
        "steady_steps_ms": steady_ms_list,
        "steady_mean_ms": steady_mean_ms,
        "total_run_model_s": tot_run_model,
    }


def parse_scorep_cubex(cubex_path: Path):
    """Uses cube_dump to extract region execution times from Score-P profile."""
    if not cubex_path or not cubex_path.exists():
        return None
    cube_dump = "/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode/scorep-8.4-papi72-install/bin/cube_dump"
    if not os.path.exists(cube_dump):
        return None

    try:
        cmd = [cube_dump, "-m", "time", "-r", str(cubex_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        lines = res.stdout.splitlines()
        regions = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            name = parts[0].split("(")[0]
            vals = []
            for p in parts[1:]:
                clean_p = p.replace(".", "", 1).replace("e-", "", 1).replace("e+", "", 1).replace("-", "", 1)
                if clean_p.isdigit():
                    try:
                        vals.append(float(p))
                    except ValueError:
                        pass
            if vals:
                regions[name] = {
                    "mean": sum(vals) / len(vals),
                    "max": max(vals),
                    "min": min(vals),
                    "total": sum(vals),
                }
        return regions
    except Exception as exc:
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Analyze SmartSim Clustered vs Standalone Traffic & Latency")
    parser.add_argument("--clustered-log", type=Path, default=None, help="Solver output log for clustered run")
    parser.add_argument("--standalone-log", type=Path, default=None, help="Solver output log for standalone run")
    parser.add_argument("--clustered-net", type=Path, default=None, help="/proc/net/dev snapshot for clustered run")
    parser.add_argument("--standalone-net", type=Path, default=None, help="/proc/net/dev snapshot for standalone run")
    parser.add_argument("--clustered-cubex", type=Path, default=None, help="Score-P profile.cubex for clustered run")
    parser.add_argument("--standalone-cubex", type=Path, default=None, help="Score-P profile.cubex for standalone run")
    parser.add_argument("--ranks", type=int, default=96, help="Number of solver MPI ranks")
    parser.add_argument("--shards", type=int, default=4, help="Number of DB shards")
    parser.add_argument("--width", type=int, default=3840, help="Grid width")
    parser.add_argument("--height", type=int, default=2160, help="Grid height")
    parser.add_argument("--steps", type=int, default=102, help="Total simulation steps")
    parser.add_argument("--interface", default="lo", help="Network interface to analyze (e.g. 'lo' or 'ib0')")
    args = parser.parse_args()

    # 1. Analytical Predictions
    pred = compute_analytical_predictions(args.ranks, args.shards, args.width, args.height, args.steps)

    print("=" * 80)
    print(" SMARTSIM TRAFFIC & LATENCY COMPARISON ANALYSIS")
    print(f" Domain: {args.width}x{args.height} | Solver Ranks: {args.ranks} | DB Shards: {args.shards}")
    print(f" Simulation Steps: {args.steps} ({pred['ml_steps']} ML steps: {pred['warm_ml_steps']} warm + {pred['steady_ml_steps']} steady)")
    print("=" * 80)

    print("\n--- 1. ANALYTICAL KEY ROUTING & COPY PREDICTION ---")
    print(f"Executing shard distribution (input 0 placement): {pred['target_shard_counts']}")
    print(f"  Input 0 ({pred['in0_on_target']}/{args.ranks}) on target shard: 100.0% -> 0 necessary copies, {args.ranks} actual copies")
    print(f"  Input 1 ({pred['in1_on_target']}/{args.ranks}) on target shard:   0.0% -> {args.ranks} necessary copies, {args.ranks} actual copies")
    print(f"  Output 0 ({pred['out0_on_target']}/{args.ranks}) on target shard:  0.0% -> {args.ranks} necessary copies, {args.ranks} actual copies")
    print("\nPer ML Step Tensor Payloads (across all 96 ranks):")
    print(f"  Input 0 (water):   {pred['in0_mb_step']:>8.2f} MiB")
    print(f"  Input 1 (terrain): {pred['in1_mb_step']:>8.2f} MiB")
    print(f"  Output 0 (y):      {pred['out0_mb_step']:>8.2f} MiB")
    print(f"  Base Payload/Step: {pred['base_step_preloaded_mb']:>8.2f} MiB (with 1-time terrain) / {pred['base_step_full_mb']:>8.2f} MiB (with per-step terrain)")
    print(f"\nPredicted Extra Traffic in run_model (Clustered):")
    print(f"  Extra per ML step: {pred['extra_run_model_mb']:>8.2f} MiB")
    print(f"  Total Clustered/Step: {pred['clustered_step_preloaded_mb']:>8.2f} MiB (1-time terrain) -> {pred['clustered_step_preloaded_mb']/pred['base_step_preloaded_mb']:.2f}x traffic (+{((pred['clustered_step_preloaded_mb']/pred['base_step_preloaded_mb'])-1)*100:.1f}%)")
    print(f"  Total Clustered/Step: {pred['clustered_step_full_mb']:>8.2f} MiB (per-step terrain)  -> {pred['clustered_step_full_mb']/pred['base_step_full_mb']:.2f}x traffic (+{((pred['clustered_step_full_mb']/pred['base_step_full_mb'])-1)*100:.1f}%)")

    # 2. Empirical Parsing
    cl_net = parse_net_dev_snapshot(args.clustered_net, args.interface)
    st_net = parse_net_dev_snapshot(args.standalone_net, args.interface)
    cl_solver = parse_solver_log(args.clustered_log)
    st_solver = parse_solver_log(args.standalone_log)
    cl_scorep = parse_scorep_cubex(args.clustered_cubex)
    st_scorep = parse_scorep_cubex(args.standalone_cubex)

    print("\n--- 2. EMPIRICAL MEASUREMENTS & VALIDATION ---")
    print(f"{'Metric':<35} | {'Clustered (Shared)':<18} | {'Standalone (Per-Node)':<20} | {'Ratio / Delta':<15}")
    print("-" * 95)

    # Traffic from /proc/net/dev snapshot
    cl_rx_str = f"{cl_net['rx_mb']:>15.2f} MiB" if cl_net else f"{'N/A':>15}"
    st_rx_str = f"{st_net['rx_mb']:>17.2f} MiB" if st_net else f"{'N/A':>17}"
    ratio_str = f"{cl_net['rx_mb']/st_net['rx_mb']:.2f}x" if (cl_net and st_net and st_net['rx_mb'] > 0) else "N/A"
    print(f"{f'Network {args.interface} rx_mb (Snapshot)':<35} | {cl_rx_str} | {st_rx_str} | {ratio_str:>15}")

    if cl_net and st_net and (cl_net['tx_mb'] > 0 or st_net['tx_mb'] > 0):
        cl_tx_str = f"{cl_net['tx_mb']:>15.2f} MiB"
        st_tx_str = f"{st_net['tx_mb']:>17.2f} MiB"
        ratio_tx = f"{cl_net['tx_mb']/st_net['tx_mb']:.2f}x" if st_net['tx_mb'] > 0 else "N/A"
        print(f"{f'Network {args.interface} tx_mb (Snapshot)':<35} | {cl_tx_str} | {st_tx_str} | {ratio_tx:>15}")

    # Traffic from Solver NET_USAGE
    cl_lo_val = cl_solver.get("net_usage", {}).get(args.interface, {}).get("rx_mb", None) if cl_solver else None
    st_lo_val = st_solver.get("net_usage", {}).get(args.interface, {}).get("rx_mb", None) if st_solver else None
    cl_lo_str = f"{cl_lo_val:>15.2f} MiB" if cl_lo_val is not None else f"{'N/A':>15}"
    st_lo_str = f"{st_lo_val:>17.2f} MiB" if st_lo_val is not None else f"{'N/A':>17}"
    ratio_lo = f"{cl_lo_val/st_lo_val:.2f}x" if (cl_lo_val is not None and st_lo_val is not None and st_lo_val > 0) else "N/A"
    print(f"{f'Solver NET_USAGE {args.interface} rx_mb':<35} | {cl_lo_str} | {st_lo_str} | {ratio_lo:>15}")

    cl_tx_val = cl_solver.get("net_usage", {}).get(args.interface, {}).get("tx_mb", None) if cl_solver else None
    st_tx_val = st_solver.get("net_usage", {}).get(args.interface, {}).get("tx_mb", None) if st_solver else None
    if cl_tx_val is not None or st_tx_val is not None:
        cl_tx_s = f"{cl_tx_val:>15.2f} MiB" if cl_tx_val is not None else f"{'N/A':>15}"
        st_tx_s = f"{st_tx_val:>17.2f} MiB" if st_tx_val is not None else f"{'N/A':>17}"
        ratio_tx_u = f"{cl_tx_val/st_tx_val:.2f}x" if (cl_tx_val is not None and st_tx_val is not None and st_tx_val > 0) else "N/A"
        print(f"{f'Solver NET_USAGE {args.interface} tx_mb':<35} | {cl_tx_s} | {st_tx_s} | {ratio_tx_u:>15}")

    # Latencies
    cl_warm_val = cl_solver.get("warm_ms", None) if cl_solver else None
    st_warm_val = st_solver.get("warm_ms", None) if st_solver else None
    cl_warm_str = f"{cl_warm_val:>15.2f} ms" if cl_warm_val is not None else f"{'N/A':>15}"
    st_warm_str = f"{st_warm_val:>17.2f} ms" if st_warm_val is not None else f"{'N/A':>17}"
    warm_ovh = f"+{((cl_warm_val/st_warm_val)-1)*100:.1f}%" if (cl_warm_val is not None and st_warm_val is not None and st_warm_val > 0) else "N/A"
    print(f"{'Warm ML Step (Step 2, ms)':<35} | {cl_warm_str} | {st_warm_str} | {warm_ovh:>15}")

    cl_steady_val = cl_solver.get("steady_mean_ms", None) if cl_solver else None
    st_steady_val = st_solver.get("steady_mean_ms", None) if st_solver else None
    cl_steady_str = f"{cl_steady_val:>15.2f} ms" if cl_steady_val is not None else f"{'N/A':>15}"
    st_steady_str = f"{st_steady_val:>17.2f} ms" if st_steady_val is not None else f"{'N/A':>17}"
    steady_ovh = f"+{((cl_steady_val/st_steady_val)-1)*100:.1f}%" if (cl_steady_val is not None and st_steady_val is not None and st_steady_val > 0) else "N/A"
    print(f"{'Steady ML Step (Mean, ms)':<35} | {cl_steady_str} | {st_steady_str} | {steady_ovh:>15}")

    # Score-P Regions
    print(f"\n--- 3. SCORE-P USER REGION TIMINGS (Across all {args.ranks} Ranks) ---")
    print(f"{'Region':<35} | {'Clustered (Per-Step / Total)':<27} | {'Standalone (Per-Step / Total)':<27} | {'Overhead':<10}")
    print("-" * 105)
    for region in ["smartsim_run_model", "smartsim_put_tensor", "smartsim_unpack_tensor", "solver_step_ml_warmup", "MPI_Allreduce"]:
        cl_entry = cl_scorep.get(region, None) if (cl_scorep and "error" not in cl_scorep) else None
        st_entry = st_scorep.get(region, None) if (st_scorep and "error" not in st_scorep) else None
        cl_t = cl_entry["mean"] if isinstance(cl_entry, dict) else None
        st_t = st_entry["mean"] if isinstance(st_entry, dict) else None
        
        div = pred['ml_steps'] if ("smartsim" in region or "ml" in region) else (pred['ml_steps'] * 2)
        if region == "solver_step_ml_warmup":
            div = 1

        cl_step_t = (cl_t / div) if cl_t is not None else None
        st_step_t = (st_t / div) if st_t is not None else None

        cl_str = f"{cl_step_t:.3f} s ({cl_t:.1f} s)" if cl_t is not None else f"{'N/A':>27}"
        st_str = f"{st_step_t:.3f} s ({st_t:.1f} s)" if st_t is not None else f"{'N/A':>27}"
        diff = f"+{((cl_t/st_t)-1)*100:.1f}%" if (cl_t is not None and st_t is not None and st_t > 0) else "N/A"
        print(f"{region:<35} | {cl_str:<27} | {st_str:<27} | {diff:>10}")

    # Bottleneck makespan
    if cl_scorep and st_scorep and "smartsim_run_model" in cl_scorep and "smartsim_run_model" in st_scorep:
        cl_max_model = cl_scorep["smartsim_run_model"]["max"] / pred['ml_steps']
        st_max_model = st_scorep["smartsim_run_model"]["max"] / pred['ml_steps']
        diff_max = f"+{((cl_max_model/st_max_model)-1)*100:.1f}%"
        print(f"{'run_model (Slowest Rank / Step)':<35} | {cl_max_model:>10.3f} s                 | {st_max_model:>10.3f} s                 | {diff_max:>10}")

    print("=" * 80)


if __name__ == "__main__":
    main()
