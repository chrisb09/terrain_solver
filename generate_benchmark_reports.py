#!/usr/bin/env python3
"""
generate_benchmark_reports.py
=============================
Generates comprehensive benchmark summary tables (Markdown + CSV) and publication-quality
comparison figures for CMI multi-framework evaluation across hardware configurations:
  1. 24-Rank Heterogeneous Allocation (1x CPU node c23mm + 1x GPU node c23g)
  2. 96-Core + 4-GPU Single Node Allocation (1x c23g full node)

Configurations compared:
  - SmartSim Parallel (c=0)
  - SmartSim Per-Node Standalone DB
  - SmartSim Sequential Token (c=1)
  - SmartSim Sequential Token (c=3)
  - AIxelerator Collective (MPI_Gatherv / Bcast / Scatterv)
  - AIxelerator P2P / Pipelined
  - PhyDLL C++ DL Client
  - PhyDLL Python DL Client
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import subprocess

def get_job_name_from_sacct(job_id: str) -> str:
    try:
        res = subprocess.run(["sacct", "-j", str(job_id), "--format=JobName%50", "--noheader"], capture_output=True, text=True)
        for line in res.stdout.strip().split("\n"):
            name = line.strip()
            if name and not name.endswith("+") and not name.startswith("batch") and not name.startswith("extern") and not name.startswith("zsh") and not name.startswith("solver") and not name.startswith("orchestrator") and not name.startswith("shard") and not name.startswith("python") and not name.startswith("phydll"):
                return name
    except Exception:
        pass
    return ""

def parse_job_log(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return None
    
    text = log_path.read_text(errors="replace")
    jid = log_path.stem.replace("mini_app_output_", "")
    job_name = get_job_name_from_sacct(jid)
    
    pm = re.search(r"CPP_ML_INTERFACE_PROVIDER from environment variable: (\w+)", text)
    provider = pm.group(1).upper() if pm else "UNKNOWN"

    # Specific configuration label
    cfg_label = "?"
    if "c0" in job_name:
        cfg_label = "SmartSim Parallel (c=0)"
    elif "per_ml_node" in job_name or "per-node" in job_name:
        cfg_label = "SmartSim Per-Node DB"
    elif "c1" in job_name:
        cfg_label = "SmartSim Chain 1 (c=1)"
    elif "c3" in job_name:
        cfg_label = "SmartSim Chain 3 (c=3)"
    elif "p2p" in job_name:
        cfg_label = "AIxelerator P2P"
    elif "coll" in job_name or "collective" in job_name:
        cfg_label = "AIxelerator Collective"
    elif "cpp" in job_name:
        cfg_label = "PhyDLL C++"
    elif "py" in job_name:
        cfg_label = "PhyDLL Python"
    else:
        # Fallback to text parsing
        if provider == "AIX":
            cm = re.search(r"CPP_ML_CONFIG from environment variable: (\S+)", text)
            cfg_label = "AIxelerator P2P" if (cm and "pipelined" in cm.group(1)) else "AIxelerator Collective"
        elif provider == "PHYDLL":
            py_m = re.search(r"USE_PYTHON_DL_CLIENT=(\d+)", text)
            if not py_m:
                py_m = re.search(r"USE_PYTHON_DL_CLIENT from environment variable: (\d+)", text)
            py_client = (py_m and py_m.group(1) == "1") or ("phydll_dl_client.py" in text)
            cfg_label = "PhyDLL Python" if py_client else "PhyDLL C++"
        elif provider == "SMARTSIM":
            lay_m = re.search(r"DB_LAYOUT=per-ml-node", text)
            seq_m = re.search(r"SMARTSIM_MPI_SEQUENTIAL_PUT=(\d+)", text)
            if lay_m:
                cfg_label = "SmartSim Per-Node DB"
            elif seq_m and seq_m.group(1) == "1":
                cfg_label = "SmartSim Chain 1 (c=1)"
            elif seq_m and seq_m.group(1) == "3":
                cfg_label = "SmartSim Chain 3 (c=3)"
            else:
                cfg_label = "SmartSim Parallel (c=0)"
        else:
            cfg_label = provider

    # Extract step timing
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
    all_ml_steps = [s[1] for s in ml_steps]
    reg_steps = [s[1] for s in regular_steps]
    
    # Memory and network metrics
    mem_max = None
    mem_m = re.search(r"MEM_USAGE_MAX rss_mb=([\d\.]+)", text)
    if mem_m:
        mem_max = float(mem_m.group(1))
        
    mem_sum = None
    mem_s_m = re.search(r"MEM_USAGE_SUM_MAX rss_mb=([\d\.]+)", text)
    if mem_s_m:
        mem_sum = float(mem_s_m.group(1))
        
    solve_time = None
    st_m = re.search(r"Solving time: (\d+) seconds", text)
    if st_m:
        solve_time = float(st_m.group(1))

    return {
        "job_id": log_path.stem.replace("mini_app_output_", ""),
        "provider": provider,
        "label": cfg_label,
        "cold_ms": cold_step,
        "warm_ms": warm_steps,
        "all_ml_ms": all_ml_steps,
        "regular_ms": reg_steps,
        "mem_max_mb": mem_max,
        "mem_sum_mb": mem_sum,
        "solve_time_s": solve_time
    }

def generate_suite_report(job_ids: List[int], suite_name: str, hardware_desc: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for jid in job_ids:
        log_path = Path(f"logs/mini_app_output_{jid}.txt")
        data = parse_job_log(log_path)
        if data:
            results.append(data)
        else:
            print(f"Warning: Could not parse log for job {jid}")
            
    if not results:
        print(f"No results found for suite {suite_name}")
        return

    # Create summary metrics table
    rows = []
    for d in results:
        warm = np.array(d["warm_ms"])
        med = np.median(warm) if len(warm) else 0.0
        q25, q75 = np.percentile(warm, [25, 75]) if len(warm) else (0.0, 0.0)
        iqr = q75 - q25
        mean = np.mean(warm) if len(warm) else 0.0
        std = np.std(warm, ddof=1) if len(warm) > 1 else 0.0
        reg_mean = np.mean(d["regular_ms"]) if d["regular_ms"] else 0.0
        
        rows.append({
            "Job ID": d["job_id"],
            "Configuration": d["label"],
            "Cold Step (ms)": d["cold_ms"],
            "Warm Median (ms)": med,
            "Warm IQR (ms)": iqr,
            "Warm Mean (ms)": mean,
            "Warm StdDev (ms)": std,
            "Regular Step Avg (ms)": reg_mean,
            "Total Solve Time (s)": d["solve_time_s"] if d["solve_time_s"] is not None else 0.0,
            "Max Rank RSS (MB)": d["mem_max_mb"] if d["mem_max_mb"] is not None else 0.0,
            "Sum Peak RSS (MB)": d["mem_sum_mb"] if d["mem_sum_mb"] is not None else 0.0,
        })
        
    df = pd.DataFrame(rows)
    
    # Save CSV
    csv_path = output_dir / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to: {csv_path}")

    # Generate Markdown report
    md_path = output_dir / "benchmark_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# CMI Multi-Framework Benchmark Report: {suite_name}\n\n")
        f.write(f"**Hardware Environment**: {hardware_desc}\n\n")
        f.write(f"**Workload**: WaterCNN 22 Timesteps (10 Warm ML Coupled Steps, 11 Regular Numerical Steps, 1 Initial Step) on 1920x1080 Grid.\n\n")
        f.write("## 1. Executive Performance Summary\n\n")
        
        # Format table
        f.write("| Job ID | Framework / Configuration | Warm Median (ms) | Warm IQR (ms) | Warm Mean (ms) | Warm StdDev (ms) | Cold Step (ms) | Total Solve Time (s) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            f.write(f"| {r['Job ID']} | **{r['Configuration']}** | {r['Warm Median (ms)']:.2f} | {r['Warm IQR (ms)']:.2f} | {r['Warm Mean (ms)']:.2f} | {r['Warm StdDev (ms)']:.2f} | {r['Cold Step (ms)']:.1f} | {r['Total Solve Time (s)']:.1f} |\n")
        f.write("\n")
        
        # Memory table
        f.write("## 2. Memory Footprint Summary\n\n")
        f.write("| Configuration | Max Rank RSS (MB) | Aggregate Peak RSS (MB) |\n")
        f.write("|---|---|---|\n")
        for _, r in df.iterrows():
            f.write(f"| {r['Configuration']} | {r['Max Rank RSS (MB)']:.1f} | {r['Sum Peak RSS (MB)']:.1f} |\n")
        f.write("\n")

        f.write("## 3. Key Observations\n\n")
        
        # Find fastest
        fastest_warm = df.sort_values("Warm Median (ms)").iloc[0]
        f.write(f"- **Fastest Warm Step**: `{fastest_warm['Configuration']}` with **{fastest_warm['Warm Median (ms)']:.2f} ms** per coupled step.\n")
        
        smartsim_c0 = df[df["Configuration"].str.contains("Parallel|c=0")]
        if not smartsim_c0.empty:
            ss_med = smartsim_c0.iloc[0]["Warm Median (ms)"]
            f.write(f"- **SmartSim Baseline (Parallel c=0)**: {ss_med:.2f} ms.\n")
            
        aix_p2p = df[df["Configuration"].str.contains("P2P")]
        if not aix_p2p.empty and not smartsim_c0.empty:
            aix_med = aix_p2p.iloc[0]["Warm Median (ms)"]
            f.write(f"- **AIxelerator P2P vs SmartSim c=0**: {ss_med / aix_med:.2f}x speedup ({aix_med:.2f} ms vs {ss_med:.2f} ms).\n")
            
        phydll_py = df[df["Configuration"].str.contains("Python")]
        phydll_cpp = df[df["Configuration"].str.contains("PhyDLL C\+\+")]
        if not phydll_py.empty and not phydll_cpp.empty:
            py_med = phydll_py.iloc[0]["Warm Median (ms)"]
            cpp_med = phydll_cpp.iloc[0]["Warm Median (ms)"]
            f.write(f"- **PhyDLL Python vs PhyDLL C++**: Python client achieved {py_med:.2f} ms vs C++ client {cpp_med:.2f} ms.\n")

        f.write("\n![Benchmark Warm Step Comparison](fig_framework_warm_step_comparison.png)\n")
        f.write("\n![Step-by-Step Progression](fig_step_by_step_progression.png)\n")

    print(f"Saved summary Markdown report to: {md_path}")

    # Generate Publication-Quality Figures
    # 1. Warm Step Bar Chart with Error Bars (IQR)
    plt.figure(figsize=(12, 6.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    configs = [r["label"] for r in results]
    medians = [float(np.median(r["warm_ms"])) for r in results]
    iqrs = [float(np.percentile(r["warm_ms"], 75) - np.percentile(r["warm_ms"], 25)) for r in results]
    
    # Custom color palette by provider
    colors = []
    for c in configs:
        if "SmartSim" in c:
            colors.append("#2b5c8f") # Navy Blue
        elif "AIxelerator" in c:
            colors.append("#d95f02") # Orange / Vermillion
        elif "PhyDLL" in c:
            colors.append("#2ca02c") # Green
        else:
            colors.append("#7570b3")
            
    bars = plt.bar(range(len(configs)), medians, yerr=iqrs, capsize=6, color=colors, edgecolor="black", alpha=0.88, width=0.62)
    
    plt.xticks(range(len(configs)), configs, rotation=25, ha="right", fontsize=11, fontweight="bold")
    plt.ylabel("Warm Coupled Step Duration (ms)", fontsize=13, fontweight="bold")
    plt.title(f"CMI Framework Comparison — Warm ML Step Duration (Median ± IQR)\n{hardware_desc}", fontsize=13, fontweight="bold", pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Annotate bar values
    for bar, med in zip(bars, medians):
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + max(iqrs)*0.1 + 5, f"{med:.1f} ms",
                 ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    plt.tight_layout()
    fig1_path = output_dir / "fig_framework_warm_step_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved bar chart figure to: {fig1_path}")

    # 2. Step-by-Step Timeline (Step 1 to 22)
    plt.figure(figsize=(13, 7), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']
    for i, r in enumerate(results):
        steps = list(range(1, len(r["all_ml_ms"]) + 1))
        # Steps in simulation: ML steps occur at steps 1, 3, 5, ..., 21 or 2, 4, ...
        # Let's plot the ML step index (1 to 11)
        ml_indices = list(range(1, len(r["all_ml_ms"]) + 1))
        plt.plot(ml_indices, r["all_ml_ms"], marker=markers[i % len(markers)], label=r["label"],
                 linewidth=2.2, markersize=7.5, alpha=0.9)
                 
    plt.xlabel("ML Inference Sample (1 = Cold Start, 2..11 = Steady State)", fontsize=12, fontweight="bold")
    plt.ylabel("Step Wall Duration (ms)", fontsize=12, fontweight="bold")
    plt.yscale("log")
    plt.title(f"Step-by-Step ML Coupling Duration (Cold Start to Steady State)\n{hardware_desc}", fontsize=13, fontweight="bold", pad=15)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10.5, loc="upper right")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig2_path = output_dir / "fig_step_by_step_progression.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved step timeline figure to: {fig2_path}")

def main():
    # 1. 24-rank HetJob Suite (1x CPU node + 1x GPU node)
    hetjob_jids = [3449953, 3449957, 3449961, 3449964, 3449970, 3449975, 3449978, 3450422]
    generate_suite_report(
        hetjob_jids,
        suite_name="24-Rank Heterogeneous CPU/GPU Allocation",
        hardware_desc="CLAIX-23 Heterogeneous (1x c23mm CPU Node with 24 Ranks + 1x c23g GPU Node with 1x NVIDIA H100)",
        output_dir=Path("results_24rank_hetjob_comparison")
    )
    
    # 2. 96-core + 4-GPU Single Node Suite
    single_node_jids = [3450490, 3450492, 3450494, 3450496, 3450498, 3450500, 3450503, 3450507]
    generate_suite_report(
        single_node_jids,
        suite_name="96-Core + 4-GPU Single Node Allocation",
        hardware_desc="CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores + 4x NVIDIA H100 GPUs)",
        output_dir=Path("results_96core_4gpu_comparison")
    )

if __name__ == "__main__":
    main()
