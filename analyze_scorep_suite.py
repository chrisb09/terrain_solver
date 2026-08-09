#!/usr/bin/env python3
import glob
import os
import re
import sys

def parse_steady_state_timings(suite_job_id=None):
    """
    Parses STEP_TIMING lines for steady state steps (4, 6, 8, 10) from suite log files.
    Groups results by (group_label, chains).
    """
    pattern = r"Step (\d+), ML, local moved: \d+(?:\.\d+)?, time: ([\d\.]+) ms"
    
    if suite_job_id:
        log_files = glob.glob(f"logs/*_{suite_job_id}_*.log")
    else:
        log_files = glob.glob("logs/scorep_chain_suite_*.log") + glob.glob("logs/scorep_reduced_suite_*.log")
    
    if not log_files:
        print(f"No log files found matching job ID / pattern.")
        return

    # Structure: results[group_label][chains] = list of steady-state step timings across all repeats
    results = {}

    for log_path in log_files:
        fname = os.path.basename(log_path)
        
        # Reduced suite pattern: scorep_reduced_suite_<jobid>_<idx>_<model>_<scale>_c<chains>_rep<repeat>.log
        m_reduced = re.search(r"scorep_reduced_suite_\d+_\d+_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_c(\d+)_rep(\d+)\.log$", fname)
        # Chain suite pattern: scorep_chain_suite_<jobid>_<idx>_(50k|600k)_c(\d+)(?:_rep\d+)?\.log
        m_chain = re.search(r"scorep_chain_suite_\d+_\d+_(50k|600k)_c(\d+)(?:_rep(\d+))?\.log$", fname)
        
        if m_reduced:
            model_key = m_reduced.group(1)
            scale_label = m_reduced.group(2)
            chains = int(m_reduced.group(3))
            group_label = f"Model: {model_key} ({scale_label}, bs=600k)"
        elif m_chain:
            batch_label = m_chain.group(1)
            chains = int(m_chain.group(2))
            group_label = f"Model: perfect_model (scale1to1, bs={batch_label})"
        else:
            continue
        
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        step_matches = dict(re.findall(pattern, content))
        steady_steps = ["4", "6", "8", "10"]
        timings = [float(step_matches[s]) for s in steady_steps if s in step_matches]
        
        if group_label not in results:
            results[group_label] = {}
        if chains not in results[group_label]:
            results[group_label][chains] = []
        results[group_label][chains].extend(timings)

    def calc_median(vals):
        if not vals:
            return 0.0
        sorted_v = sorted(vals)
        n = len(sorted_v)
        mid = n // 2
        return (sorted_v[mid] + sorted_v[mid - 1]) / 2.0 if n % 2 == 0 else sorted_v[mid]

    for group_label, chains_dict in sorted(results.items()):
        print(f"\n### Steady-State ML Step Time (Median over 3 Runs x 4 Steps = 12 Points) - {group_label}")
        print("| Chains (c) | Data Points | Median (ms) | Min (ms) | Max (ms) | Speedup vs Default (c=0) | Speedup vs c=1 |")
        print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        c0_vals = chains_dict.get(0, [])
        c0_med = calc_median(c0_vals)
        c1_vals = chains_dict.get(1, [])
        c1_med = calc_median(c1_vals)

        for c in range(0, 7):
            vals = chains_dict.get(c, [])
            if not vals:
                print(f"| {c}{' (default)' if c == 0 else ''} | 0 | N/A | N/A | N/A | - | - |")
                continue
            med = calc_median(vals)
            mn = min(vals)
            mx = max(vals)
            
            sp_c0 = f"{c0_med / med:.2f}x" if c0_med > 0 and med > 0 else "-"
            sp_c1 = f"{c1_med / med:.2f}x" if c1_med > 0 and med > 0 else "-"
            c_str = f"{c} (default)" if c == 0 else f"{c}"
            
            print(f"| {c_str} | {len(vals)} | {med:.1f} | {mn:.1f} | {mx:.1f} | {sp_c0} | {sp_c1} |")

if __name__ == "__main__":
    job_id = sys.argv[1] if len(sys.argv) > 1 else None
    parse_steady_state_timings(job_id)
