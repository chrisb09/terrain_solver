#!/usr/bin/env python3
"""
generate_benchmark_reports.py
=============================
Generates comprehensive benchmark summary tables (Markdown + CSV) and publication-quality
comparison figures for CMI multi-framework evaluation across hardware configurations:
  1. 24-Rank Heterogeneous Allocation (1x CPU node c23mm + 1x GPU node c23g)
  2. 96-Core + 1-GPU Single Node Allocation (1x c23g node, single GPU)
  3. 96-Core + 4-GPU Single Node Allocation (1x c23g node, 4x NVIDIA H100)
  4. Cross-Hardware Grouped Comparison (Scaling & GPU acceleration effects)

Produces both top-level suite summaries and dedicated per-library-configuration directories
with detailed diagnostic plots and timing CSVs.
"""

import os
import sys
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_job_name_from_sacct(job_id: str) -> str:
    try:
        res = subprocess.run(["sacct", "-j", str(job_id), "--format=JobName%50", "--noheader"], capture_output=True, text=True)
        for line in res.stdout.strip().split("\n"):
            parts = line.strip().split()
            if parts:
                name = parts[-1]
                if name and not name.endswith("+") and not name.startswith("batch") and not name.startswith("extern") and not name.startswith("zsh") and not name.startswith("solver") and not name.startswith("orchestrator") and not name.startswith("shard") and not name.startswith("python") and not name.startswith("phydll"):
                    return name
    except Exception:
        pass
    return ""

def get_config_folder_slug(cfg_label: str) -> str:
    slug_map = {
        "SmartSim Parallel (c=0)": "smartsim_c0",
        "SmartSim Per-Node DB": "smartsim_per_node_db",
        "SmartSim Per-GPU DB": "smartsim_per_gpu_db",
        "SmartSim Chain 1 (c=1)": "smartsim_c1",
        "SmartSim Chain 3 (c=3)": "smartsim_c3",
        "AIxelerator Collective": "aix_collective",
        "AIxelerator P2P": "aix_pipelined",
        "PhyDLL C++": "phydll_cpp",
        "PhyDLL Python": "phydll_py",
    }
    return slug_map.get(cfg_label, cfg_label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", ""))

def parse_job_log(log_path: Path) -> Optional[Dict[str, Any]]:
    if not log_path.exists():
        return None
    
    text = log_path.read_text(errors="replace")
    jid = log_path.stem.replace("mini_app_output_", "")
    job_name = get_job_name_from_sacct(jid)
    
    pm = re.search(r"CPP_ML_INTERFACE_PROVIDER from environment variable: (\w+)", text)
    provider = pm.group(1).upper() if pm else "UNKNOWN"

    # Specific configuration label from exact token matches
    cfg_label = "?"
    if "per_gpu_db" in job_name:
        cfg_label = "SmartSim Per-GPU DB"
    elif "per_node_db" in job_name or "per_ml_node" in job_name:
        cfg_label = "SmartSim Per-Node DB"
    elif "_c0_" in job_name or job_name.endswith("_c0"):
        cfg_label = "SmartSim Parallel (c=0)"
    elif "_c1_" in job_name or job_name.endswith("_c1"):
        cfg_label = "SmartSim Chain 1 (c=1)"
    elif "_c3_" in job_name or job_name.endswith("_c3"):
        cfg_label = "SmartSim Chain 3 (c=3)"
    elif "_p2p_" in job_name or job_name.endswith("_p2p"):
        cfg_label = "AIxelerator P2P"
    elif "_coll_" in job_name or "collective" in job_name:
        cfg_label = "AIxelerator Collective"
    elif "_cpp_" in job_name or "phydll_cpp" in job_name:
        cfg_label = "PhyDLL C++"
    elif "_py_" in job_name or "phydll_py" in job_name:
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
            nd_m = re.search(r"DB_NODES from environment variable: (\d+)", text)
            if lay_m and nd_m and int(nd_m.group(1)) > 1:
                cfg_label = "SmartSim Per-GPU DB"
            elif lay_m:
                cfg_label = "SmartSim Per-Node DB"
            elif seq_m and seq_m.group(1) == "1":
                cfg_label = "SmartSim Chain 1 (c=1)"
            elif seq_m and seq_m.group(1) == "3":
                cfg_label = "SmartSim Chain 3 (c=3)"
            else:
                cfg_label = "SmartSim Parallel (c=0)"
        else:
            cfg_label = provider

    # Extract all step timings in chronological order
    step_pattern = re.compile(r"STEP_TIMING step=(\d+) solver=(ML|Regular) step_ms=([\d\.]+) local_moved=([\d\.]+)")
    all_steps = []
    ml_steps = []
    regular_steps = []
    
    for match in step_pattern.finditer(text):
        step_num = int(match.group(1))
        solver = match.group(2)
        step_ms = float(match.group(3))
        local_moved = float(match.group(4))
        
        step_entry = {
            "step": step_num,
            "solver_type": solver,
            "duration_ms": step_ms,
            "local_moved": local_moved
        }
        all_steps.append(step_entry)
        
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
    
    # Memory metrics
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

    # Parse memory progression over steps if logged
    mem_prog_pattern = re.compile(r"MEM_USAGE rank=0 label=after_\w+ step=(\d+) rss_mb=([\d\.]+)")
    mem_steps = {}
    for match in mem_prog_pattern.finditer(text):
        st = int(match.group(1))
        rss = float(match.group(2))
        mem_steps[st] = rss

    # Network metrics
    ib0_rx, ib0_tx = None, None
    ib_m = re.search(r"NET_USAGE if=ib0 rx_mb=([\d\.]+) tx_mb=([\d\.]+)", text)
    if ib_m:
        ib0_rx = float(ib_m.group(1))
        ib0_tx = float(ib_m.group(2))

    return {
        "job_id": jid,
        "provider": provider,
        "label": cfg_label,
        "folder_slug": get_config_folder_slug(cfg_label),
        "cold_ms": cold_step,
        "warm_ms": warm_steps,
        "all_ml_ms": all_ml_steps,
        "regular_ms": reg_steps,
        "all_steps": all_steps,
        "mem_max_mb": mem_max,
        "mem_sum_mb": mem_sum,
        "mem_steps": mem_steps,
        "ib0_rx_mb": ib0_rx,
        "ib0_tx_mb": ib0_tx,
        "solve_time_s": solve_time
    }

def generate_individual_config_artifacts(data: Dict[str, Any], hardware_desc: str, config_dir: Path):
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Step Timings CSV
    df_steps = pd.DataFrame(data["all_steps"])
    csv_path = config_dir / "step_timings.csv"
    df_steps.to_csv(csv_path, index=False)
    
    # 2. Config Summary Markdown Report
    warm = np.array(data["warm_ms"])
    med = np.median(warm) if len(warm) else 0.0
    q25, q75 = np.percentile(warm, [25, 75]) if len(warm) else (0.0, 0.0)
    iqr = q75 - q25
    mean = np.mean(warm) if len(warm) else 0.0
    std = np.std(warm, ddof=1) if len(warm) > 1 else 0.0
    p95 = np.percentile(warm, 95) if len(warm) else 0.0
    reg_mean = np.mean(data["regular_ms"]) if data["regular_ms"] else 0.0
    
    md_path = config_dir / "config_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Configuration Analysis: {data['label']}\n\n")
        f.write(f"- **Framework / Provider**: `{data['provider']}`\n")
        f.write(f"- **Slurm Job ID**: `{data['job_id']}`\n")
        f.write(f"- **Hardware Environment**: {hardware_desc}\n")
        f.write(f"- **Workload**: WaterCNN ($1920 \\times 1080$, 22 timesteps)\n\n")
        
        f.write("## 1. Timestep Timing Statistics\n\n")
        f.write("| Metric | Duration (ms) |\n")
        f.write("|---|---|\n")
        f.write(f"| **Cold Start ML Step (Step 1)** | {data['cold_ms']:.2f} ms |\n")
        f.write(f"| **Warm ML Step Median** | **{med:.2f} ms** |\n")
        f.write(f"| **Warm ML Step IQR** | {iqr:.2f} ms |\n")
        f.write(f"| **Warm ML Step Mean** | {mean:.2f} ms |\n")
        f.write(f"| **Warm ML Step StdDev** | {std:.2f} ms |\n")
        f.write(f"| **Warm ML Step 95th Percentile** | {p95:.2f} ms |\n")
        f.write(f"| **Regular Numerical Step Avg** | {reg_mean:.2f} ms |\n")
        f.write(f"| **Total Simulation Solve Time** | {data['solve_time_s']:.1f} s |\n\n")
        
        f.write("## 2. Resource Utilization\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| **Peak RSS (Max Rank)** | {data['mem_max_mb']:.1f} MB |\n")
        f.write(f"| **Peak RSS (Sum over Ranks)** | {data['mem_sum_mb']:.1f} MB |\n")
        if data["ib0_rx_mb"] is not None:
            f.write(f"| **InfiniBand Traffic (ib0 RX / TX)** | {data['ib0_rx_mb']:.2f} MB / {data['ib0_tx_mb']:.2f} MB |\n")
        f.write("\n")
        
        f.write("## 3. Performance Visualizations\n\n")
        f.write("### Timestep Progression (Cold Start vs Steady State)\n")
        f.write("![Step Progression](fig_step_progression.png)\n\n")
        f.write("### Warm Step Latency Distribution & Boxplot\n")
        f.write("![Latency Distribution](fig_step_latency_distribution.png)\n\n")
        if data["mem_steps"]:
            f.write("### Memory Footprint Evolution\n")
            f.write("![Memory Profile](fig_memory_profile.png)\n\n")

    # 3. Figure: Step Progression (Cold Start vs ML Steady State vs Numerical)
    plt.figure(figsize=(10, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    steps = [s["step"] for s in data["all_steps"]]
    durs = [s["duration_ms"] for s in data["all_steps"]]
    types = [s["solver_type"] for s in data["all_steps"]]
    
    colors = ["#d95f02" if t == "ML" else "#2b5c8f" for t in types]
    plt.plot(steps, durs, color="#888888", linestyle="--", alpha=0.6, zorder=1)
    
    # Scatter points for ML and Regular
    ml_s = [s for s, t in zip(steps, types) if t == "ML"]
    ml_d = [d for d, t in zip(durs, types) if t == "ML"]
    reg_s = [s for s, t in zip(steps, types) if t == "Regular"]
    reg_d = [d for d, t in zip(durs, types) if t == "Regular"]
    
    plt.scatter(ml_s, ml_d, color="#d95f02", s=70, label="ML Coupled Step", zorder=3, edgecolor="black")
    plt.scatter(reg_s, reg_d, color="#2b5c8f", s=45, label="Regular Numerical Step", zorder=2, alpha=0.8)
    
    plt.xlabel("Simulation Timestep", fontsize=11, fontweight="bold")
    plt.ylabel("Step Wall Duration (ms)", fontsize=11, fontweight="bold")
    plt.yscale("log")
    plt.title(f"{data['label']} — Timestep Duration History\n(Job {data['job_id']})", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(config_dir / "fig_step_progression.png", dpi=300)
    plt.close()

    # 4. Figure: Warm Latency Distribution (Histogram + Boxplot)
    if len(warm) > 0:
        fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [0.25, 0.75]}, dpi=300)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        
        # Boxplot on top
        ax_box.boxplot(warm, vert=False, widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor="#2ca02c", alpha=0.7, edgecolor="black"),
                       medianprops=dict(color="black", linewidth=2.0))
        ax_box.set_yticks([])
        ax_box.set_title(f"{data['label']} — Warm ML Step Duration Distribution (N={len(warm)})", fontsize=12, fontweight="bold")
        
        # Histogram on bottom
        ax_hist.hist(warm, bins=min(len(warm), 8), color="#2ca02c", alpha=0.8, edgecolor="black")
        ax_hist.axvline(med, color="red", linestyle="--", linewidth=2.0, label=f"Median: {med:.2f} ms")
        ax_hist.axvline(mean, color="blue", linestyle=":", linewidth=2.0, label=f"Mean: {mean:.2f} ms")
        ax_hist.set_xlabel("Duration (ms)", fontsize=11, fontweight="bold")
        ax_hist.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax_hist.legend(frameon=True, facecolor="white", framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(config_dir / "fig_step_latency_distribution.png", dpi=300)
        plt.close()

    # 5. Figure: Memory Profile
    if data["mem_steps"]:
        plt.figure(figsize=(9, 4.5), dpi=300)
        m_steps = sorted(data["mem_steps"].keys())
        m_rss = [data["mem_steps"][s] for s in m_steps]
        
        plt.plot(m_steps, m_rss, marker="o", color="#7570b3", linewidth=2.2, markersize=6.0)
        plt.xlabel("Simulation Timestep", fontsize=11, fontweight="bold")
        plt.ylabel("Rank 0 Resident Memory (MB)", fontsize=11, fontweight="bold")
        plt.title(f"{data['label']} — Memory Footprint Profile (Rank 0)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(config_dir / "fig_memory_profile.png", dpi=300)
        plt.close()

def generate_suite_report(job_ids: List[int], suite_name: str, hardware_desc: str, suite_dir: Path) -> Optional[pd.DataFrame]:
    suite_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = suite_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for jid in job_ids:
        log_path = Path(f"logs/mini_app_output_{jid}.txt")
        data = parse_job_log(log_path)
        if data:
            results.append(data)
            # Generate detailed per-configuration directory and plots
            cfg_dir = suite_dir / data["folder_slug"]
            generate_individual_config_artifacts(data, hardware_desc, cfg_dir)
        else:
            print(f"Notice: Log for job {jid} is pending or not yet available.")
            
    if not results:
        print(f"No completed results yet for suite {suite_name}")
        return None

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
            "Folder": d["folder_slug"],
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
    csv_path = summary_dir / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to: {csv_path}")

    # Generate Markdown report
    md_path = summary_dir / "benchmark_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# CMI Multi-Framework Benchmark Report: {suite_name}\n\n")
        f.write(f"**Hardware Environment**: {hardware_desc}\n\n")
        f.write(f"**Workload**: WaterCNN 22 Timesteps (10 Warm ML Coupled Steps, 11 Regular Numerical Steps, 1 Initial Step) on 1920x1080 Grid.\n\n")
        f.write("## 1. Executive Performance Summary\n\n")
        
        # Format table
        f.write("| Job ID | Framework / Configuration | Warm Median (ms) | Warm IQR (ms) | Warm Mean (ms) | Warm StdDev (ms) | Cold Step (ms) | Total Solve Time (s) | Detailed Artifacts |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            f.write(f"| {r['Job ID']} | **{r['Configuration']}** | {r['Warm Median (ms)']:.2f} | {r['Warm IQR (ms)']:.2f} | {r['Warm Mean (ms)']:.2f} | {r['Warm StdDev (ms)']:.2f} | {r['Cold Step (ms)']:.1f} | {r['Total Solve Time (s)']:.1f} | [`../{r['Folder']}/`](../{r['Folder']}/config_summary.md) |\n")
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

    # Generate Figures
    # 1. Warm Step Bar Chart with Error Bars (IQR)
    plt.figure(figsize=(12, 6.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    configs = [r["label"] for r in results]
    medians = [float(np.median(r["warm_ms"])) for r in results]
    iqrs = [float(np.percentile(r["warm_ms"], 75) - np.percentile(r["warm_ms"], 25)) for r in results]
    
    colors = []
    for c in configs:
        if "SmartSim" in c:
            colors.append("#2b5c8f")
        elif "AIxelerator" in c:
            colors.append("#d95f02")
        elif "PhyDLL" in c:
            colors.append("#2ca02c")
        else:
            colors.append("#7570b3")
            
    bars = plt.bar(range(len(configs)), medians, yerr=iqrs, capsize=6, color=colors, edgecolor="black", alpha=0.88, width=0.62)
    
    plt.xticks(range(len(configs)), configs, rotation=25, ha="right", fontsize=11, fontweight="bold")
    plt.ylabel("Warm Coupled Step Duration (ms)", fontsize=13, fontweight="bold")
    plt.title(f"CMI Framework Comparison — Warm ML Step Duration (Median ± IQR)\n{hardware_desc}", fontsize=13, fontweight="bold", pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    for bar, med in zip(bars, medians):
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + max(iqrs)*0.08 + 4, f"{med:.1f} ms",
                 ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    plt.tight_layout()
    fig1_path = summary_dir / "fig_framework_warm_step_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # 2. Step-by-Step Timeline
    plt.figure(figsize=(13, 7), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'h']
    for i, r in enumerate(results):
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
    fig2_path = summary_dir / "fig_step_by_step_progression.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    
    return df

def generate_cross_hardware_comparison(df_het: pd.DataFrame, df_1g: pd.DataFrame, df_4g: pd.DataFrame, output_dir: Path):
    if df_het is None or df_1g is None or df_4g is None:
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Common configurations present in all suites
    common_cfgs = [
        "SmartSim Parallel (c=0)",
        "SmartSim Per-Node DB",
        "SmartSim Chain 1 (c=1)",
        "SmartSim Chain 3 (c=3)",
        "AIxelerator Collective",
        "AIxelerator P2P",
        "PhyDLL C++",
        "PhyDLL Python"
    ]
    
    rows = []
    for cfg in common_cfgs:
        r_het = df_het[df_het["Configuration"] == cfg]
        r_1g = df_1g[df_1g["Configuration"] == cfg]
        r_4g = df_4g[df_4g["Configuration"] == cfg]
        
        m_het = r_het.iloc[0]["Warm Median (ms)"] if not r_het.empty else np.nan
        m_1g = r_1g.iloc[0]["Warm Median (ms)"] if not r_1g.empty else np.nan
        m_4g = r_4g.iloc[0]["Warm Median (ms)"] if not r_4g.empty else np.nan
        
        rows.append({
            "Configuration": cfg,
            "24 Ranks (1 CPU + 1 GPU) [ms]": m_het,
            "96 Ranks (1 Node, 1 GPU) [ms]": m_1g,
            "96 Ranks (1 Node, 4 GPUs) [ms]": m_4g
        })
        
    # Also add SmartSim Per-GPU DB from 4G if present
    r_4g_gpu = df_4g[df_4g["Configuration"] == "SmartSim Per-GPU DB"]
    if not r_4g_gpu.empty:
        rows.append({
            "Configuration": "SmartSim Per-GPU DB (4 DBs)",
            "24 Ranks (1 CPU + 1 GPU) [ms]": np.nan,
            "96 Ranks (1 Node, 1 GPU) [ms]": np.nan,
            "96 Ranks (1 Node, 4 GPUs) [ms]": r_4g_gpu.iloc[0]["Warm Median (ms)"]
        })
        
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(output_dir / "cross_hardware_summary.csv", index=False)
    
    # Write Markdown
    md_path = output_dir / "cross_hardware_comparison.md"
    with open(md_path, "w") as f:
        f.write("# CMI Cross-Hardware Scaling & GPU Acceleration Report\n\n")
        f.write("Comparison of CMI coupling frameworks across:\n")
        f.write("- **24 Ranks HetJob**: 1x `c23mm` CPU node (24 ranks) + 1x `c23g` GPU node (1x H100)\n")
        f.write("- **96 Ranks 1-GPU**: 1x `c23g` node (96 ranks, 1x H100 active)\n")
        f.write("- **96 Ranks 4-GPU**: 1x `c23g` node (96 ranks, 4x H100 active)\n\n")
        f.write("## 1. Summary Comparison Table (Warm Step Median Duration in ms)\n\n")
        f.write("| Framework / Configuration | 24 Ranks (1 CPU + 1 GPU) | 96 Ranks (1 Node, 1 GPU) | 96 Ranks (1 Node, 4 GPUs) |\n")
        f.write("|---|---|---|---|\n")
        for _, r in comp_df.iterrows():
            h_str = f"{r['24 Ranks (1 CPU + 1 GPU) [ms]']:.2f} ms" if not np.isnan(r['24 Ranks (1 CPU + 1 GPU) [ms]']) else "—"
            g1_str = f"{r['96 Ranks (1 Node, 1 GPU) [ms]']:.2f} ms" if not np.isnan(r['96 Ranks (1 Node, 1 GPU) [ms]']) else "—"
            g4_str = f"{r['96 Ranks (1 Node, 4 GPUs) [ms]']:.2f} ms" if not np.isnan(r['96 Ranks (1 Node, 4 GPUs) [ms]']) else "—"
            f.write(f"| **{r['Configuration']}** | {h_str} | {g1_str} | {g4_str} |\n")
            
        f.write("\n![Cross Hardware Grouped Comparison](fig_cross_hardware_comparison.png)\n")

    # Grouped Bar Chart
    plt.figure(figsize=(14, 7), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    labels = [r["Configuration"] for r in rows if "Per-GPU" not in r["Configuration"]]
    het_vals = [r["24 Ranks (1 CPU + 1 GPU) [ms]"] for r in rows if "Per-GPU" not in r["Configuration"]]
    g1_vals = [r["96 Ranks (1 Node, 1 GPU) [ms]"] for r in rows if "Per-GPU" not in r["Configuration"]]
    g4_vals = [r["96 Ranks (1 Node, 4 GPUs) [ms]"] for r in rows if "Per-GPU" not in r["Configuration"]]
    
    x = np.arange(len(labels))
    width = 0.26
    
    plt.bar(x - width, het_vals, width, label="24 Ranks (1 CPU + 1 GPU)", color="#2b5c8f", edgecolor="black", alpha=0.9)
    plt.bar(x, g1_vals, width, label="96 Ranks (1 Node, 1 GPU)", color="#d95f02", edgecolor="black", alpha=0.9)
    plt.bar(x + width, g4_vals, width, label="96 Ranks (1 Node, 4 GPUs)", color="#2ca02c", edgecolor="black", alpha=0.9)
    
    plt.xticks(x, labels, rotation=25, ha="right", fontsize=11, fontweight="bold")
    plt.ylabel("Warm ML Step Duration (ms)", fontsize=13, fontweight="bold")
    plt.title("CMI Framework Scaling & Multi-GPU Acceleration Comparison\n(WaterCNN 1920x1080 Grid, 22 Timesteps)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(frameon=True, facecolor="white", framealpha=0.95, fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    fig_path = output_dir / "fig_cross_hardware_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved cross-hardware comparison report to: {output_dir}")

def main():
    base_results_dir = Path("results")
    
    # 1. 24-rank HetJob Suite (1x CPU node + 1x GPU node)
    hetjob_jids = [3449953, 3449957, 3449961, 3449964, 3449970, 3449975, 3449978, 3450422]
    df_het = generate_suite_report(
        hetjob_jids,
        suite_name="24-Rank Heterogeneous CPU/GPU Allocation",
        hardware_desc="CLAIX-23 Heterogeneous (1x c23mm CPU Node with 24 Ranks + 1x c23g GPU Node with 1x NVIDIA H100)",
        suite_dir=base_results_dir / "24rank_hetjob"
    )
    
    # 2. 96-core + 1-GPU Single Node Suite
    single_node_1g_jids = [3504523, 3504529, 3504532, 3504535, 3504538, 3504541, 3504547, 3504551]
    df_1g = generate_suite_report(
        single_node_1g_jids,
        suite_name="96-Core + 1-GPU Single Node Allocation",
        hardware_desc="CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores, 1x NVIDIA H100 Active)",
        suite_dir=base_results_dir / "96core_1gpu"
    )

    # 3. 96-core + 4-GPU Single Node Suite
    single_node_4g_jids = [3504561, 3504563, 3504565, 3504567, 3504573, 3504577, 3504580, 3504584, 3504590]
    df_4g = generate_suite_report(
        single_node_4g_jids,
        suite_name="96-Core + 4-GPU Single Node Allocation",
        hardware_desc="CLAIX-23 Single GPU Node (1x c23g Node with 96 CPU Cores + 4x NVIDIA H100 GPUs)",
        suite_dir=base_results_dir / "96core_4gpu"
    )

    # 4. Cross-Hardware Comparison
    generate_cross_hardware_comparison(df_het, df_1g, df_4g, output_dir=base_results_dir / "cross_hardware_comparison")

if __name__ == "__main__":
    main()
