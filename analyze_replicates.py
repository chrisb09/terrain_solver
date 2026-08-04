#!/usr/bin/env zsh
#!/usr/bin/env python3
"""Analyze 10-replicate GPU/CPU benchmark output logs or CPU scaling sweep."""

import argparse
import re
import sys
from pathlib import Path
import numpy as np

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

def analyze_scaling(jobs_info: list, csv_out: Path = None):
    """Analyze CPU scaling sweep across db_nodes = 1..8 and strategies."""
    has_strat = any('strategy' in item or len(item.get('interface', '')) > 4 or item.get('variant') in ('dynamic', 'tpq_only', 'intra_only') for item in jobs_info)
    
    if has_strat:
        # Group by (db_nodes, strategy)
        results = {}  # (db_nodes, strategy) -> list of per-run medians
        for item in jobs_info:
            db_nodes = int(item['rep']) if str(item['rep']).isdigit() else int(item['interface'])
            strat = item['variant']
            job_id = item['job_id']
            key = (db_nodes, strat)
            if key not in results:
                results[key] = []
            
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
                    'job_id': job_id,
                    'run_median': float(np.median(timings)),
                    'tpq': item.get('tpq', '-'),
                    'intra': item.get('intra', '-')
                })

        print("\n" + "="*95)
        print(" 96-CORE EXCLUSIVE CPU ML SCALING BENCHMARK (96 Solver Ranks, 1440x960, Perfect Tags)")
        print("="*95)
        print(f"{'DB Nodes':<10} | {'Strategy':<15} | {'Runs':<5} | {'Median ms':<12} | {'Mean ms':<12} | {'StdDev ms':<10} | {'Speedup vs 1N':<15}")
        print("-" * 95)

        baseline_median = None
        if (1, 'dynamic') in results and results[(1, 'dynamic')]:
            baseline_median = float(np.median([r['run_median'] for r in results[(1, 'dynamic')]]))

        all_keys = sorted(results.keys(), key=lambda x: (x[0], x[1]))
        csv_rows = ["db_nodes,strategy,count,median_ms,mean_ms,std_ms,speedup_vs_1node"]

        for db_nodes, strat in all_keys:
            runs = results[(db_nodes, strat)]
            if not runs:
                continue
            medians = [r['run_median'] for r in runs]
            med = float(np.median(medians))
            mean_val = float(np.mean(medians))
            std_val = float(np.std(medians, ddof=1)) if len(medians) > 1 else 0.0
            speedup = (baseline_median / med) if (baseline_median and med > 0) else 1.0

            print(f"{db_nodes:<10} | {strat:<15} | {len(medians):<5} | {med:<12.2f} | {mean_val:<12.2f} | {std_val:<10.2f} | {speedup:<15.2f}x")
            csv_rows.append(f"{db_nodes},{strat},{len(medians)},{med:.2f},{mean_val:.2f},{std_val:.2f},{speedup:.2f}")

        print("=" * 95)

        if csv_out:
            csv_out.write_text("\n".join(csv_rows) + "\n")
            print(f"[+] Summary CSV written to {csv_out}")
        return

    # Standard simple scaling
    has_tpq = any('tpq' in item for item in jobs_info)
    print("\n" + "="*85)
    if has_tpq:
        print(" CPU DYNAMIC TPQ / INTRA-OP ML SCALING BENCHMARK (24 Ranks, 216x144, Perfect Tags)")
        print("="*85)
        print(f"{'DB Nodes':<10} | {'TPQ':<6} | {'Intra Threads':<13} | {'Job ID':<10} | {'Median ms':<12} | {'Speedup vs 1 Node':<20}")
    else:
        print(" CPU ML SCALING BENCHMARK (24 Solver Ranks, 216x144, Perfect {...} Tags)")
        print("="*85)
        print(f"{'DB Nodes':<10} | {'Job ID':<10} | {'Median ms':<12} | {'Mean ms':<12} | {'Speedup vs 1 Node':<20}")
    print("-" * 85)

    baseline_median = None

    for item in sorted(jobs_info, key=lambda x: int(x['rep'])):
        db_nodes = int(item['rep'])
        job_id = item['job_id']
        tpq_str = item.get('tpq', '-').replace('tpq=', '')
        intra_str = item.get('intra', '-').replace('intra=', '')

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
            med = float(np.median(timings))
            mean_val = float(np.mean(timings))
            if baseline_median is None:
                baseline_median = med
            speedup = (baseline_median / med) if (baseline_median and med > 0) else 1.0
            
            if has_tpq:
                print(f"{db_nodes:<10} | {tpq_str:<6} | {intra_str:<13} | {job_id:<10} | {med:<12.2f} | {speedup:<20.2f}x")
            else:
                print(f"{db_nodes:<10} | {job_id:<10} | {med:<12.2f} | {mean_val:<12.2f} | {speedup:<20.2f}x")
        else:
            if has_tpq:
                print(f"{db_nodes:<10} | {tpq_str:<6} | {intra_str:<13} | {job_id:<10} | {'Pending/No Log':<12} | {'-':<20}")
            else:
                print(f"{db_nodes:<10} | {job_id:<10} | {'Pending/No Log':<12} | {'-':<12} | {'-':<20}")

    print("=" * 85)

def analyze_job_mapping(jobs_info: list):
    results = {}  # (interface, variant) -> list of per-run medians

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
            run_median = float(np.median(timings))
            results[key].append({
                'rep': item['rep'],
                'job_id': job_id,
                'timings': timings,
                'run_median': run_median,
                'run_mean': float(np.mean(timings))
            })
        else:
            print(f"Warning: No valid step timings found for job {job_id} ({item['interface']} {item['variant']} rep {item['rep']})", file=sys.stderr)

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

    stats = {}
    for interface, variant, name, mode in conditions:
        key = (interface, variant)
        runs = results.get(key, [])
        if not runs:
            stats[key] = {'count': 0, 'median': 0, 'mean': 0, 'std': 0}
            continue
        medians = [r['run_median'] for r in runs]
        stats[key] = {
            'count': len(medians),
            'median': float(np.median(medians)),
            'mean': float(np.mean(medians)),
            'std': float(np.std(medians, ddof=1)) if len(medians) > 1 else 0.0,
            'medians': medians
        }

    print(f"{'Coupling Path':<20} | {'Mode':<22} | {'Runs':<5} | {'Median ms':<10} | {'Mean ms':<10} | {'StdDev ms':<10}")
    print("-" * 85)
    for interface, variant, name, mode in conditions:
        st = stats[(interface, variant)]
        print(f"{name:<20} | {mode:<22} | {st['count']:<5} | {st['median']:<10.2f} | {st['mean']:<10.2f} | {st['std']:<10.2f}")
    print("=" * 85)

    print("\nMEASURED OVERHEAD (Natural vs Tagged Control Medians):")
    for interface, name in [('direct', 'Direct SmartSim'), ('cpp', 'CMI SmartSim')]:
        c_stat = stats.get((interface, 'balanced_control'), {})
        n_stat = stats.get((interface, 'natural'), {})
        if c_stat.get('median', 0) > 0 and n_stat.get('median', 0) > 0:
            overhead = ((n_stat['median'] / c_stat['median']) - 1.0) * 100.0
            print(f"  {name:20s}: Control = {c_stat['median']:.2f} ms | Natural = {n_stat['median']:.2f} ms | Overhead = {overhead:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Analyze replicate or scaling logs.")
    parser.add_argument("--job-list", type=Path, default=Path("gpu_replicates_jobs.txt"), help="Path to text file containing job submissions mapping")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    job_list_path = args.job_list
    if not job_list_path.exists() and (Path("../mini_app") / args.job_list).exists():
        job_list_path = Path("../mini_app") / args.job_list

    if job_list_path.exists():
        filename = job_list_path.name.lower()
        jobs_info = []
        
        # Check if CSV manifest
        if job_list_path.suffix == ".csv":
            lines = job_list_path.read_text().splitlines()
            header = lines[0].split(',')
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 6:
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
                    # Find numeric job id
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
            analyze_scaling(jobs_info, csv_out=csv_out)
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
