#!/usr/bin/env python3
import sys
import re
import glob
from pathlib import Path
import statistics

def parse_log(log_path):
    text = Path(log_path).read_text(errors="replace")
    
    # Extract job name or configuration
    job_name_match = re.search(r"Job Name:\s*(\S+)", text)
    job_name = job_name_match.group(1) if job_name_match else Path(log_path).stem
    
    # Extract ML interface
    ml_iface_match = re.search(r"Resolved ML interface:\s*([^\n]+)", text)
    ml_iface = ml_iface_match.group(1).strip() if ml_iface_match else "unknown"
    
    # Extract step timings
    # STEP_TIMING step=2 solver=ML step_ms=123.456 local_moved=...
    ml_steps = []
    for m in re.finditer(r"STEP_TIMING\s+step=(\d+)\s+solver=ML\s+step_ms=([0-9.]+)", text):
        step_num = int(m.group(1))
        step_ms = float(m.group(2))
        ml_steps.append((step_num, step_ms))
    
    if not ml_steps:
        return None
    
    first_ml_ms = ml_steps[0][1] if ml_steps else float("nan")
    # Steady steps (exclude step 2 warmup if we have multiple steps)
    steady_ms = [ms for s, ms in ml_steps if s > 2] if len(ml_steps) > 1 else [ms for s, ms in ml_steps]
    
    median_ms = statistics.median(steady_ms) if steady_ms else float("nan")
    mean_ms = statistics.mean(steady_ms) if steady_ms else float("nan")
    min_ms = min(steady_ms) if steady_ms else float("nan")
    max_ms = max(steady_ms) if steady_ms else float("nan")
    
    # ML Traffic
    # ML_TRAFFIC input_mb=15.8203 output_mb=0.878906 preload_mb=0 units=MiB
    traffic_match = re.search(r"ML_TRAFFIC\s+input_mb=([0-9.]+)\s+output_mb=([0-9.]+)\s+preload_mb=([0-9.]+)", text)
    input_mb = float(traffic_match.group(1)) if traffic_match else float("nan")
    output_mb = float(traffic_match.group(2)) if traffic_match else float("nan")
    preload_mb = float(traffic_match.group(3)) if traffic_match else float("nan")
    
    # Memory usage
    mem_match = re.search(r"MEM_USAGE_MAX\s+rss_mb=([0-9.]+)", text)
    max_rss_mb = float(mem_match.group(1)) if mem_match else float("nan")
    
    # Solving duration
    solve_time_match = re.search(r"Solving time:\s*(\d+)\s*seconds", text)
    solve_sec = int(solve_time_match.group(1)) if solve_time_match else -1
    
    return {
        "file": log_path,
        "job_name": job_name,
        "ml_iface": ml_iface,
        "step_count": len(ml_steps),
        "first_ml_ms": first_ml_ms,
        "median_ml_ms": median_ms,
        "mean_ml_ms": mean_ms,
        "min_ml_ms": min_ms,
        "max_ml_ms": max_ms,
        "input_mb": input_mb,
        "output_mb": output_mb,
        "preload_mb": preload_mb,
        "max_rss_mb": max_rss_mb,
        "solve_sec": solve_sec
    }

def main():
    logs = sys.argv[1:]
    if not logs:
        logs = sorted(glob.glob("output_3033*.txt") + glob.glob("logs/mini_app_output_3033*.txt"))
    
    results = []
    for log in logs:
        parsed = parse_log(log)
        if parsed:
            results.append(parsed)
            
    if not results:
        print("No valid benchmark results found.")
        return
        
    print(f"{'Log File':<32} {'Interface':<18} {'Steps':<6} {'First ML(ms)':<14} {'Median ML(ms)':<14} {'Min-Max ML(ms)':<18} {'Preload(MB)':<12} {'Solve(s)':<8}")
    print("-" * 130)
    for r in results:
        fname = Path(r['file']).name
        min_max_str = f"{r['min_ml_ms']:.1f} - {r['max_ml_ms']:.1f}"
        print(f"{fname:<32} {r['ml_iface']:<18} {r['step_count']:<6} {r['first_ml_ms']:<14.2f} {r['median_ml_ms']:<14.2f} {min_max_str:<18} {r['preload_mb']:<12.1f} {r['solve_sec']:<8}")

if __name__ == "__main__":
    main()
