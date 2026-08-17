#!/usr/bin/env python3
"""Analyze 10-replicate GPU/CPU benchmark output logs or CPU scaling sweep."""

import argparse
import re
import sys
from pathlib import Path
import numpy as np
from scipy import stats

STEP_TIMING_RE = re.compile(r"STEP_TIMING step=(\d+) solver=ML step_ms=([0-9.]+)")

def parse_log_file(log_path: Path):
    """Extract steady-state ML step timings (step > 2) in ms from a log file."""
    if not log_path.exists():
        return []
    timings = []
    text = log_path.read_text(errors="replace")
    for line in text.splitlines():
        match = STEP_TIMING_RE.search(line)
        if match:
            step = int(match.group(1))
            ms = float(match.group(2))
            if step > 2:  # Step 2 is warmup
                timings.append(ms)
    return timings

def analyze_scaling(jobs_info: list, csv_out: Path = None, raw_csv_out: Path = None):
    """Analyze CPU scaling sweep across db_nodes = 1..8 and strategies."""
    dataset = {}  # (db_nodes, strategy) -> list of raw step timings
    job_medians = {}  # (db_nodes, strategy) -> list of per-job medians
    raw_rows = ["job_id,replicate,db_nodes,strategy,tpq,intra,step_ms"]

    for item in jobs_info:
        try:
            db_nodes = int(item['interface'])
        except (ValueError, TypeError):
            db_nodes = int(item['rep']) if str(item['rep']).isdigit() else 1

        strat = item['variant']
        job_id = item['job_id']
        rep = item.get('rep', 1)
        tpq = item.get('tpq', '-')
        intra = item.get('intra', '-')

        key = (db_nodes, strat)
        if key not in dataset:
            dataset[key] = []
            job_medians[key] = []

        log_paths = [
            Path(f"logs/mini_app_output_{job_id}.txt"),
            Path(f"../mini_app/logs/mini_app_output_{job_id}.txt"),
            Path(f"output_{job_id}.txt"),
            Path(f"../mini_app/output_{job_id}.txt"),
        ]
        timings = []
        for p in log_paths:
            t = parse_log_file(p)
            if t:
                timings = t
                break
        if timings:
            dataset[key].extend(timings)
            job_medians[key].append(float(np.median(timings)))
            for ms in timings:
                raw_rows.append(f"{job_id},{rep},{db_nodes},{strat},{tpq},{intra},{ms:.2f}")

    if raw_csv_out:
        raw_csv_out.write_text("\n".join(raw_rows) + "\n")
        print(f"[+] Raw 12-sample dataset written to {raw_csv_out}")

    print("\n" + "="*115)
    print(" 96-CORE EXCLUSIVE CPU ML SCALING BENCHMARK (FORCED IDEAL {...} TAG DISTRIBUTION)")
    print("="*115)

    baseline_samples = dataset.get((1, 'dynamic'), [])
    baseline_median = float(np.median(baseline_samples)) if baseline_samples else None

    for strat in ['dynamic', 'intra_only', 'tpq_only']:
        strat_title = {
            'dynamic': '1. DYNAMIC BALANCED STRATEGY (TPQ * Intra = 96)',
            'intra_only': '2. INTRA-OP-ONLY BASELINE (TPQ = 1, Intra = 96)',
            'tpq_only': '3. TPQ-ONLY BASELINE (TPQ = 96, Intra = 1)',
        }.get(strat, strat.upper())

        print(f"\n--- {strat_title} ---")
        print(f"{'Nodes':<6} | {'Samples':<8} | {'Median ms':<10} | {'Mean ms':<10} | {'StdDev ms':<9} | {'Q1 ms':<9} | {'Q3 ms':<9} | {'Low Whisker':<11} | {'Up Whisker':<11} | {'Welch p-val':<12} | {'Speedup':<8}")
        print("-" * 115)

        for n in range(1, 9):
            key = (n, strat)
            if key not in dataset or not dataset[key]:
                continue
            sm = dataset[key]
            med = float(np.median(sm))
            mean_val = float(np.mean(sm))
            std_val = float(np.std(sm, ddof=1)) if len(sm) > 1 else 0.0
            q1 = float(np.percentile(sm, 25))
            q3 = float(np.percentile(sm, 75))
            iqr = q3 - q1
            low_w = float(max(min(sm), q1 - 1.5 * iqr))
            up_w = float(min(max(sm), q3 + 1.5 * iqr))

            if key == (1, 'dynamic'):
                p_str = "Baseline"
            else:
                _, p_val = stats.ttest_ind(baseline_samples, sm, equal_var=False)
                p_str = f"{p_val:.2e}" if p_val >= 0.0001 else "<0.0001***"

            speedup = (baseline_median / med) if (baseline_median and med > 0) else 1.0
            print(f"{n:<6} | {len(sm):<8} | {med:<10.1f} | {mean_val:<10.1f} | {std_val:<9.1f} | {q1:<9.1f} | {q3:<9.1f} | {low_w:<11.1f} | {up_w:<11.1f} | {p_str:<12} | {speedup:<8.2f}x")

    print("=" * 115)

def analyze_job_mapping(jobs_info: list):
    results = {}
    for item in jobs_info:
        key = (item['interface'], item['variant'])
        if key not in results:
            results[key] = []
        job_id = item['job_id']
        log_paths = [
            Path(f"logs/mini_app_output_{job_id}.txt"),
            Path(f"../mini_app/logs/mini_app_output_{job_id}.txt"),
            Path(f"output_{job_id}.txt"),
            Path(f"../mini_app/output_{job_id}.txt"),
        ]
        timings = []
        for p in log_paths:
            t = parse_log_file(p)
            if t:
                timings = t
                break
        if timings:
            results[key].append({
                'rep': item['rep'],
                'job_id': job_id,
                'timings': timings,
                'run_median': float(np.median(timings)),
                'run_mean': float(np.mean(timings))
            })
    return results

def print_summary_tables(results: dict, title: str = "10-REPLICATE BENCHMARK RESULTS"):
    print("\n" + "="*85)
    print(f" {title.upper()}")
    print("="*85)
    
    conditions = [
        ('direct', 'balanced_control', 'Direct SmartSim', 'Control ({...} tags)'),
        ('direct', 'natural',          'Direct SmartSim', 'Natural (no tags)'),
        ('cpp',    'balanced_control', 'CMI SmartSim',    'Control ({...} tags)'),
        ('cpp',    'natural',          'CMI SmartSim',    'Natural (no tags)'),
    ]

    stats_dict = {}
    for interface, variant, name, mode in conditions:
        key = (interface, variant)
        runs = results.get(key, [])
        if not runs:
            stats_dict[key] = {'count': 0, 'median': 0, 'mean': 0, 'std': 0}
            continue
        medians = [r['run_median'] for r in runs]
        stats_dict[key] = {
            'count': len(medians),
            'median': float(np.median(medians)),
            'mean': float(np.mean(medians)),
            'std': float(np.std(medians, ddof=1)) if len(medians) > 1 else 0.0,
            'medians': medians
        }

    print(f"{'Coupling Path':<20} | {'Mode':<22} | {'Runs':<5} | {'Median ms':<10} | {'Mean ms':<10} | {'StdDev ms':<10}")
    print("-" * 85)
    for interface, variant, name, mode in conditions:
        st = stats_dict[(interface, variant)]
        print(f"{name:<20} | {mode:<22} | {st['count']:<5} | {st['median']:<10.2f} | {st['mean']:<10.2f} | {st['std']:<10.2f}")
    print("=" * 85)

    print("\nMEASURED OVERHEAD (Natural vs Tagged Control Medians):")
    for interface, name in [('direct', 'Direct SmartSim'), ('cpp', 'CMI SmartSim')]:
        c_stat = stats_dict.get((interface, 'balanced_control'), {})
        n_stat = stats_dict.get((interface, 'natural'), {})
        if c_stat.get('median', 0) > 0 and n_stat.get('median', 0) > 0:
            overhead = ((n_stat['median'] / c_stat['median']) - 1.0) * 100.0
            print(f"  {name:20s}: Control = {c_stat['median']:.2f} ms | Natural = {n_stat['median']:.2f} ms | Overhead = {overhead:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Analyze replicate or scaling logs.")
    parser.add_argument("--job-list", type=Path, default=Path("gpu_replicates_jobs.txt"), help="Path to text file containing job submissions mapping")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional output CSV path")
    parser.add_argument("--raw-csv-out", type=Path, default=Path("cpu_96core_raw_samples.csv"), help="Output path for raw samples")
    args = parser.parse_args()

    job_list_path = args.job_list
    if not job_list_path.exists() and (Path("../mini_app") / args.job_list).exists():
        job_list_path = Path("../mini_app") / args.job_list

    if job_list_path.exists():
        filename = job_list_path.name.lower()
        jobs_info = []
        
        if job_list_path.suffix == ".csv":
            lines = job_list_path.read_text().splitlines()
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 6 and parts[5].isdigit():
                    jobs_info.append({
                        'rep': parts[0],
                        'interface': parts[1],  # db_nodes
                        'variant': parts[2],    # strategy
                        'tpq': parts[3],
                        'intra': parts[4],
                        'job_id': int(parts[5])
                    })
        else:
            for line in job_list_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 3:
                    job_id = None
                    for p in parts:
                        if p.isdigit() and len(p) >= 6:
                            job_id = int(p)
                            break
                    if job_id:
                        item = {
                            'rep': parts[0],
                            'interface': parts[1] if len(parts) > 1 else 'direct',
                            'variant': parts[2] if len(parts) > 2 else 'balanced_control',
                            'job_id': job_id
                        }
                        for p in parts[3:]:
                            if '=' in p:
                                k, v = p.split('=', 1)
                                item[k] = v
                        jobs_info.append(item)
        
        if "scaling" in filename or "suite" in filename or "96core" in filename or "pilot" in filename:
            csv_out = args.csv_out or Path("cpu_96core_scaling_summary.csv")
            analyze_scaling(jobs_info, csv_out=csv_out, raw_csv_out=args.raw_csv_out)
        else:
            if "cpu" in filename:
                title = "10-Replicate CPU Benchmark Results (216x144, 24 Ranks, 6 DB Nodes / 24 Cores per Node)"
            else:
                title = "10-Replicate GPU Benchmark Results (1920x1080, 48 Ranks, 6 DB Nodes / 24 GPUs)"
            results = analyze_job_mapping(jobs_info)
            print_summary_tables(results, title=title)
    else:
        print(f"Job list file not found: {args.job_list}", file=sys.stderr)

if __name__ == "__main__":
    main()
