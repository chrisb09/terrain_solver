#!/usr/bin/env python3
"""
analyze_5node_suite.py
=======================
Aggregates and statistically analyzes the 5-node distributed SmartSim benchmark suite
comparing Clustered vs Standalone deployments across replicate runs.

Usage:
  python3 analyze_5node_suite.py --manifest 5node_traffic_manifest.txt
  python3 analyze_5node_suite.py --jobs 4910131 4910132
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
import numpy as np


def parse_net_dev_snapshot(net_file: Path, iface: str = "ib0"):
    if not net_file or not net_file.exists():
        return None
    text = net_file.read_text(encoding="utf-8", errors="replace")
    start_match = re.search(rf"START\s+{iface}:\s*(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)", text)
    end_match = re.search(rf"END\s+{iface}:\s*(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)", text)
    if start_match and end_match:
        s_rx, s_rxp, s_tx, s_txp = map(int, start_match.groups())
        e_rx, e_rxp, e_tx, e_txp = map(int, end_match.groups())
        return {
            "rx_mb": (e_rx - s_rx) / (1024 * 1024),
            "tx_mb": (e_tx - s_tx) / (1024 * 1024),
            "total_mb": ((e_rx - s_rx) + (e_tx - s_tx)) / (1024 * 1024),
        }
    return None


def parse_solver_log(log_path: Path, iface: str = "ib0"):
    if not log_path or not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")

    net_usage = {}
    m_net = re.search(rf"NET_USAGE\s+if={iface}\s+rx_mb=([0-9.]+)\s+tx_mb=([0-9.]+)", text)
    if m_net:
        rx = float(m_net.group(1))
        tx = float(m_net.group(2))
        net_usage = {"rx_mb": rx, "tx_mb": tx, "total_mb": rx + tx}

    ml_steps = {}
    for m in re.finditer(r"STEP_TIMING\s+step=(\d+)\s+solver=ML\s+step_ms=([0-9.]+)", text):
        ml_steps[int(m.group(1))] = float(m.group(2))

    cpp_ml_steps = [float(m.group(1)) * 1000.0 for m in re.finditer(r"Finished CPP-ML step\.\s+Wall time \(seconds\):\s+total=([0-9.]+)", text)]

    warm_ms = ml_steps.get(2, 0.0)
    steady_ms = [v for k, v in ml_steps.items() if k > 2]
    steady_cpp_ms = cpp_ml_steps[1:] if len(cpp_ml_steps) > 1 else []

    return {
        "net_usage": net_usage,
        "warm_ms": warm_ms,
        "steady_ms": steady_ms,
        "steady_mean_ms": np.mean(steady_ms) if steady_ms else 0.0,
        "steady_median_ms": np.median(steady_ms) if steady_ms else 0.0,
        "steady_cpp_mean_ms": np.mean(steady_cpp_ms) if steady_cpp_ms else 0.0,
    }


def parse_scorep_cubex(cubex_path: Path, ml_steps: int = 51):
    if not cubex_path or not cubex_path.exists():
        return None
    cube_dump = "/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode/scorep-8.4-papi72-install/bin/cube_dump"
    if not os.path.exists(cube_dump):
        return None
    try:
        cmd = [cube_dump, "-m", "time", "-r", str(cubex_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        regions = {}
        for line in res.stdout.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
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
                div = ml_steps if ("smartsim" in name or "ml" in name) else (ml_steps * 2)
                if name == "solver_step_ml_warmup":
                    div = 1
                regions[name] = {
                    "total_s": sum(vals) / len(vals),
                    "per_step_s": (sum(vals) / len(vals)) / div,
                    "max_per_step_s": max(vals) / div,
                }
        return regions
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze 5-Node Distributed SmartSim Suite")
    parser.add_argument("--manifest", type=Path, default=Path("5node_traffic_manifest.txt"), help="Path to manifest text file")
    parser.add_argument("--jobs", nargs="+", help="Specific job IDs to analyze")
    parser.add_argument("--interface", default="ib0", help="Network interface (default: ib0)")
    parser.add_argument("--steps", type=int, default=102, help="Total simulation steps")
    args = parser.parse_args()

    job_entries = []
    if args.jobs:
        for idx, jid in enumerate(args.jobs, 1):
            job_entries.append((idx, jid, "unknown"))
    elif args.manifest.exists():
        for line in args.manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                rep = int(parts[0])
                jid = parts[1]
                order = parts[2] if len(parts) > 2 else "unknown"
                job_entries.append((rep, jid, order))
    else:
        print(f"ERROR: Neither --jobs specified nor manifest {args.manifest} found.", file=sys.stderr)
        sys.exit(1)

    ml_steps = args.steps // 2

    # Data collection across replicates
    reps_data = []

    print("=" * 85)
    print(" 5-NODE SMARTSIM SUITE STATISTICAL ANALYSIS (1 Solver Node + 4 DB Nodes)")
    print(f" Partition: c23ms | Interface: {args.interface} | Steps: {args.steps} ({ml_steps} ML Steps)")
    print(f" Found {len(job_entries)} replicate job(s) to analyze.")
    print("=" * 85)

    for rep, jid, order in job_entries:
        cl_log = Path(f"logs/solver_clustered_{jid}.log")
        st_log = Path(f"logs/solver_standalone_{jid}.log")
        res_dir = Path(f"results/5node_traffic_{jid}")

        cl_net = parse_net_dev_snapshot(res_dir / "clustered" / f"net_{args.interface}_snapshot.txt", args.interface)
        st_net = parse_net_dev_snapshot(res_dir / "standalone" / f"net_{args.interface}_snapshot.txt", args.interface)

        cl_solv = parse_solver_log(cl_log, args.interface)
        st_solv = parse_solver_log(st_log, args.interface)

        cl_cubex = parse_scorep_cubex(res_dir / "clustered" / "scorep" / "profile.cubex", ml_steps)
        st_cubex = parse_scorep_cubex(res_dir / "standalone" / "scorep" / "profile.cubex", ml_steps)

        reps_data.append({
            "rep": rep,
            "job_id": jid,
            "order": order,
            "cl_net": cl_net,
            "st_net": st_net,
            "cl_solv": cl_solv,
            "st_solv": st_solv,
            "cl_cubex": cl_cubex,
            "st_cubex": st_cubex,
        })

    # Summary table header
    print(f"\n{'Rep':<4} | {'Job ID':<9} | {'Order':<16} | {'Clustered ib0':<15} | {'Standalone ib0':<15} | {'Traffic Ratio':<14} | {'Steady Step Mean':<18}")
    print("-" * 105)

    ratios = []
    cl_traffics = []
    st_traffics = []
    cl_steadiest = []
    st_steadiest = []
    cl_models = []
    st_models = []

    for r in reps_data:
        jid = r["job_id"]
        order = r["order"]
        rep = r["rep"]

        cl_t_mb = r["cl_net"]["total_mb"] if r["cl_net"] else (r["cl_solv"]["net_usage"].get("total_mb", 0.0) if r["cl_solv"] else 0.0)
        st_t_mb = r["st_net"]["total_mb"] if r["st_net"] else (r["st_solv"]["net_usage"].get("total_mb", 0.0) if r["st_solv"] else 0.0)

        ratio = (cl_t_mb / st_t_mb) if st_t_mb > 0 else 0.0
        if ratio > 0:
            ratios.append(ratio)
            cl_traffics.append(cl_t_mb)
            st_traffics.append(st_t_mb)

        cl_st_mean = r["cl_solv"]["steady_mean_ms"] if r["cl_solv"] else 0.0
        st_st_mean = r["st_solv"]["steady_mean_ms"] if r["st_solv"] else 0.0
        if cl_st_mean > 0: cl_steadiest.append(cl_st_mean)
        if st_st_mean > 0: st_steadiest.append(st_st_mean)

        if r["cl_cubex"] and "smartsim_run_model" in r["cl_cubex"]:
            cl_models.append(r["cl_cubex"]["smartsim_run_model"]["per_step_s"])
        if r["st_cubex"] and "smartsim_run_model" in r["st_cubex"]:
            st_models.append(r["st_cubex"]["smartsim_run_model"]["per_step_s"])

        cl_str = f"{cl_t_mb:>11.2f} MiB" if cl_t_mb > 0 else "Pending"
        st_str = f"{st_t_mb:>11.2f} MiB" if st_t_mb > 0 else "Pending"
        rat_str = f"{ratio:>10.2f}x" if ratio > 0 else "N/A"
        time_str = f"{cl_st_mean:.1f} vs {st_st_mean:.1f} ms" if (cl_st_mean > 0 and st_st_mean > 0) else "N/A"

        print(f"{rep:<4} | {jid:<9} | {order:<16} | {cl_str:<15} | {st_str:<15} | {rat_str:<14} | {time_str:<18}")

    print("-" * 105)

    if ratios:
        print("\n--- STATISTICAL SUITE SUMMARY ACROSS COMPLETED REPLICATES ---")
        print(f"Number of valid completed paired replicates: {len(ratios)}")
        print(f"InfiniBand Traffic Ratio: Mean = {np.mean(ratios):.3f}x (std = {np.std(ratios):.3f}x, median = {np.median(ratios):.3f}x)")
        print(f"Clustered Traffic Total:  Mean = {np.mean(cl_traffics):.2f} MiB ({np.mean(cl_traffics)/1024:.2f} GiB), std = {np.std(cl_traffics):.2f} MiB")
        print(f"Standalone Traffic Total: Mean = {np.mean(st_traffics):.2f} MiB ({np.mean(st_traffics)/1024:.2f} GiB), std = {np.std(st_traffics):.2f} MiB")
        print(f"Extra run_model Traffic:  Mean = +{(np.mean(cl_traffics)-np.mean(st_traffics)):.2f} MiB (+{(np.mean(cl_traffics)-np.mean(st_traffics))/1024:.2f} GiB)")

        if cl_steadiest and st_steadiest:
            print(f"\nSteady ML Step Latency:")
            print(f"  Clustered:  Mean = {np.mean(cl_steadiest):.2f} ms (std = {np.std(cl_steadiest):.2f} ms, median = {np.median(cl_steadiest):.2f} ms)")
            print(f"  Standalone: Mean = {np.mean(st_steadiest):.2f} ms (std = {np.std(st_steadiest):.2f} ms, median = {np.median(st_steadiest):.2f} ms)")
            print(f"  Overhead:   +{np.mean(cl_steadiest)-np.mean(st_steadiest):.2f} ms (+{((np.mean(cl_steadiest)/np.mean(st_steadiest))-1)*100:.2f}%)")

        if cl_models and st_models:
            print(f"\nScore-P smartsim_run_model Time (per step):")
            print(f"  Clustered:  Mean = {np.mean(cl_models):.3f} s (std = {np.std(cl_models):.3f} s)")
            print(f"  Standalone: Mean = {np.mean(st_models):.3f} s (std = {np.std(st_models):.3f} s)")
            print(f"  Overhead:   +{np.mean(cl_models)-np.mean(st_models):.3f} s (+{((np.mean(cl_models)/np.mean(st_models))-1)*100:.2f}%)")

    print("=" * 85)


if __name__ == "__main__":
    main()
