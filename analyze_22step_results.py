#!/usr/bin/env python3
import sys
import re
from pathlib import Path
import numpy as np

def parse_log(log_path: Path):
    if not log_path.exists():
        return None
    
    text = log_path.read_text()

    job_name = "?"
    driver_log = log_path.with_name(log_path.name.replace("mini_app_output_", "mini_app_driver_"))
    if driver_log.exists():
        dm = re.search(r"SLURM_JOB_NAME=(\S+)", driver_log.read_text())
        if dm:
            job_name = dm.group(1)
    if job_name == "?":
        pm = re.search(r"CPP_ML_INTERFACE_PROVIDER from environment variable: (\w+)", text)
        sm = re.search(r"CUSTOM_JOB_NAME_SUFFIX from environment variable: _cpp_interface_\w+_\d+gpu_(\w+?)(?:_revamped|_prepare|$)", text)
        if pm and sm:
            provider = pm.group(1)
            model = sm.group(1)
            if model == "benchmark":
                model = "giant_mlp"
            job_name = f"{provider}/{model}"
            if provider == "AIX":
                bm = re.search(r"CPP_ML_CONFIG from environment variable: (\S+)", text)
                variant = "p2p" if bm and "pipelined" in bm.group(1) else "collective"
                job_name = f"{job_name}/{variant}"
            elif provider == "PHYDLL":
                py_m = re.search(r"USE_PYTHON_DL_CLIENT=(\d+)", text)
                if not py_m:
                    py_m = re.search(r"USE_PYTHON_DL_CLIENT from environment variable: (\d+)", text)
                py_client = (py_m and py_m.group(1) == "1") or ("phydll_dl_client.py" in text)
                client_type = "Python" if py_client else "C++"
                job_name = f"{job_name}/{client_type}"
            elif provider == "SMARTSIM":
                lay_m = re.search(r"DB_LAYOUT from environment variable: (\S+)", text)
                seq_m = re.search(r"SMARTSIM_MPI_SEQUENTIAL_PUT from environment variable: (\d+)", text)
                seq_val = seq_m.group(1) if seq_m else "0"
                lay_val = lay_m.group(1) if lay_m else "shared"
                if lay_val == "per-ml-node":
                    job_name = f"{job_name}/per-node-db"
                else:
                    job_name = f"{job_name}/c{seq_val}"

    step_pattern = re.compile(r"STEP_TIMING step=(\d+) solver=(ML|Regular) step_ms=([\d\.]+) local_moved=([\d\.]+)")
    
    ml_steps = []
    regular_steps = []
    
    for match in step_pattern.finditer(text):
        step_num = int(match.group(1))
        solver = match.group(2)
        step_ms = float(match.group(3))
        
        if solver == "ML":
            ml_steps.append((step_num, step_ms))
        else:
            regular_steps.append((step_num, step_ms))
            
    if not ml_steps:
        return None
        
    cold_step = ml_steps[0][1] if ml_steps else 0.0
    warm_steps = [s[1] for s in ml_steps[1:]]
    reg_step_times = [s[1] for s in regular_steps]
    
    return {
        "job_name": job_name,
        "cold_ms": cold_step,
        "warm_ms": warm_steps,
        "regular_ms": reg_step_times
    }

def print_summary(results: dict):
    print("\n" + "="*120)
    print(" CMI BENCHMARK COMPARISON SUMMARY (Uninstrumented USE_SCOREP=0)")
    print("="*120)
    header = f"{'Case / Provider':<38} | {'Cold ms':<10} | {'Warm Median':<12} | {'Warm IQR':<10} | {'Warm Mean':<12} | {'Warm StdDev':<11} | {'Reg Avg':<9}"
    print(header)
    print("-" * 120)
    
    for case_name, data in results.items():
        label = f"{case_name} [{data['job_name']}]" if data else case_name
        if data is None or not data["warm_ms"]:
            print(f"{label:<38} | {'FAILED / RUNNING':<75}")
            continue
            
        cold = data["cold_ms"]
        warm = np.array(data["warm_ms"])
        reg = np.array(data["regular_ms"]) if data["regular_ms"] else np.array([0.0])
        
        med = np.median(warm)
        q25, q75 = np.percentile(warm, [25, 75])
        iqr = q75 - q25
        mean = np.mean(warm)
        std = np.std(warm, ddof=1) if len(warm) > 1 else 0.0
        reg_avg = np.mean(reg)
        
        print(f"{label:<38} | {cold:<10.1f} | {med:<12.2f} | {iqr:<10.2f} | {mean:<12.2f} | {std:<11.2f} | {reg_avg:<9.2f}")
        
    print("="*120 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_22step_results.py <job_id_1> [<job_id_2> ...]")
        # Auto-discover recent logs
        logs = sorted(Path("logs").glob("mini_app_output_2914*.txt"))
        if not logs:
            return
        results = {}
        for p in logs:
            jid = p.stem.replace("mini_app_output_", "")
            res = parse_log(p)
            results[f"Job {jid}"] = res
        print_summary(results)
        return
        
    results = {}
    for jid in sys.argv[1:]:
        p = Path(f"logs/mini_app_output_{jid}.txt")
        results[f"Job {jid}"] = parse_log(p)
    print_summary(results)

if __name__ == "__main__":
    main()
