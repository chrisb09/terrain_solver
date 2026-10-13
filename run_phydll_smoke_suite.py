#!/usr/bin/env python3
"""
PhyDLL Smoke Scaling Suite:
  - CPU cases: 1:1, 2:1, 4:1 (Default & Readiness) under account rwth2150 on partition c23ms.
  - GPU cases: 1:1 (Default & Readiness) under account rwth0792 on partition c23g (surrogate).
  - Non-exclusive, 24 CPUs per node, severely reduced steps (6 steps).
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

MINI_APP_DIR = Path("/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app")
TMP_DIR = MINI_APP_DIR / "tmp_phydll_smoke"
LOGS_DIR = MINI_APP_DIR / "logs"
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

COMMON_EXPORTS = [
    "ALL",
    "USE_SCOREP_ENV=0",
    "SKIP_PAPI_METRICS=1",
    "SCOREP_METRIC_PAPI=",
    "MLCOUPLING_MEM_LOG_VERBOSE=0",
    "MODEL_NAME_ENV=watercnn",
    "MODEL_BACKEND_ENV=TORCH",
    "TOTAL_STEPS_ENV=6",
    "SAVE_EVERY_ENV=6",
    "TARGET_WIDTH_ENV=216",
    "TARGET_HEIGHT_ENV=144",
    "CHUNK_SIZE_ENV=12",
    "ML_BATCH_SIZE_ENV=500000",
    "SKIP_COMPILE_ENV=1",
    "FORCE_FRESH_RUN_ENV=1",
    "OVERWRITE_OUTPUT_ENV=1",
    "SKIP_RENDERING_ENV=1",
    "USE_LOCAL_RUNTIME_STAGE_ENV=0",
    "USE_LOCAL_MODEL_CACHE_ENV=1",
    "USE_SMARTSIM=0",
    "USE_CPP_ML_INTERFACE=1",
    "ML_INTERFACE_ENV=cpp",
    "CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL",
    "USE_PYTHON_DL_CLIENT=0",
    "OVERWRITE_JOB_NAME_ENV=1",
    "COMPILE_OUTPUT_PATH_ENV=build_gpu_refresh_plain",
    f"PHYDLL_DL_BUILD_DIR={MINI_APP_DIR}/../CPP-ML-Interface/dl_clients/build-gpu-refresh-plain",
    f"PHYDLL_DL_CLIENT={MINI_APP_DIR}/../CPP-ML-Interface/dl_clients/build-gpu-refresh-plain/phydll_dl_client",
    f"PHYDLL_LIB_DIR_ENV={MINI_APP_DIR}/../CPP-ML-Interface/extern/phydll/build-gpu-refresh/lib",
    "PHYDLL_NP_DL_ENV=1",
    "PHYDLL_DL_NODES_ENV=1",
    "PHYDLL_SAFE_MPI_ENV=1",
    "PHYDLL_MPMD_SHUTDOWN_BARRIER=1",
    "ML_INFERENCE_CPU_CORES_ENV=24",
]

CASES = [
    # 1. CPU 1:1 Default
    {
        "name": "phydll_smoke_cpu_1to1_def",
        "account": "rwth2150",
        "partition": "c23ms",
        "solver_nodes": 1,
        "solver_ranks": 24,
        "grid_x": 6,
        "grid_z": 4,
        "device": "cpu",
        "model_path": "train_models/model_a/watercnn_cpu.pt",
        "config": "config_phydll_smoke_cpu_default.toml",
        "gpus": 0,
        "is_gpu": False,
    },
    # 2. CPU 1:1 Readiness
    {
        "name": "phydll_smoke_cpu_1to1_ready",
        "account": "rwth2150",
        "partition": "c23ms",
        "solver_nodes": 1,
        "solver_ranks": 24,
        "grid_x": 6,
        "grid_z": 4,
        "device": "cpu",
        "model_path": "train_models/model_a/watercnn_cpu.pt",
        "config": "config_phydll_smoke_cpu_readiness.toml",
        "gpus": 0,
        "is_gpu": False,
    },
    # 3. CPU 2:1 Default
    {
        "name": "phydll_smoke_cpu_2to1_def",
        "account": "rwth2150",
        "partition": "c23ms",
        "solver_nodes": 2,
        "solver_ranks": 48,
        "grid_x": 8,
        "grid_z": 6,
        "device": "cpu",
        "model_path": "train_models/model_a/watercnn_cpu.pt",
        "config": "config_phydll_smoke_cpu_default.toml",
        "gpus": 0,
        "is_gpu": False,
    },
    # 4. CPU 2:1 Readiness
    {
        "name": "phydll_smoke_cpu_2to1_ready",
        "account": "rwth2150",
        "partition": "c23ms",
        "solver_nodes": 2,
        "solver_ranks": 48,
        "grid_x": 8,
        "grid_z": 6,
        "device": "cpu",
        "model_path": "train_models/model_a/watercnn_cpu.pt",
        "config": "config_phydll_smoke_cpu_readiness.toml",
        "gpus": 0,
        "is_gpu": False,
    },
    # 5. CPU 4:1 Default
    {
        "name": "phydll_smoke_cpu_4to1_def",
        "account": "rwth2150",
        "partition": "c23ms",
        "solver_nodes": 4,
        "solver_ranks": 96,
        "grid_x": 12,
        "grid_z": 8,
        "device": "cpu",
        "model_path": "train_models/model_a/watercnn_cpu.pt",
        "config": "config_phydll_smoke_cpu_default.toml",
        "gpus": 0,
        "is_gpu": False,
    },
    # 6. CPU 4:1 Readiness
    {
        "name": "phydll_smoke_cpu_4to1_ready",
        "account": "rwth2150",
        "partition": "c23ms",
        "solver_nodes": 4,
        "solver_ranks": 96,
        "grid_x": 12,
        "grid_z": 8,
        "device": "cpu",
        "model_path": "train_models/model_a/watercnn_cpu.pt",
        "config": "config_phydll_smoke_cpu_readiness.toml",
        "gpus": 0,
        "is_gpu": False,
    },
    # 7. GPU 1:1 Default (Surrogate)
    {
        "name": "phydll_smoke_gpu_1to1_def",
        "account": "rwth0792",
        "partition": "c23g",
        "solver_nodes": 1,
        "solver_ranks": 24,
        "grid_x": 6,
        "grid_z": 4,
        "device": "gpu",
        "model_path": "train_models/model_a/watercnn_cuda.pt",
        "config": "config_phydll_smoke_gpu_default.toml",
        "gpus": 1,
        "is_gpu": True,
    },
    # 8. GPU 1:1 Readiness (Surrogate)
    {
        "name": "phydll_smoke_gpu_1to1_ready",
        "account": "rwth0792",
        "partition": "c23g",
        "solver_nodes": 1,
        "solver_ranks": 24,
        "grid_x": 6,
        "grid_z": 4,
        "device": "gpu",
        "model_path": "train_models/model_a/watercnn_cuda.pt",
        "config": "config_phydll_smoke_gpu_readiness.toml",
        "gpus": 1,
        "is_gpu": True,
    },
]

def make_sbatch_script(c: dict) -> Path:
    sbatch_file = TMP_DIR / f"{c['name']}.sbatch"
    
    if not c["is_gpu"]:
        # Pure CPU HetJob on c23ms
        lines = [
            "#!/bin/zsh",
            f"#SBATCH --job-name={c['name']}",
            f"#SBATCH --account={c['account']}",
            "#SBATCH --time=00:05:00",
            f"#SBATCH --output={LOGS_DIR}/{c['name']}_%j.log",
            "#SBATCH --partition=c23ms",
            f"#SBATCH --nodes={c['solver_nodes']}",
            "#SBATCH --ntasks-per-node=24",
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --mem-per-cpu=2400M",
            "",
            "#SBATCH hetjob",
            "",
            f"#SBATCH --partition={c['partition']}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=24",
            "#SBATCH --mem-per-cpu=2400M",
            "",
            f'SCRIPT_DIR="{MINI_APP_DIR}"',
            'cd "${SCRIPT_DIR}"',
            'exec /bin/zsh "${SCRIPT_DIR}/proper_slurm_job.sh"',
        ]
    else:
        # Surrogate GPU HetJob on c23g under rwth0792
        lines = [
            "#!/bin/zsh",
            f"#SBATCH --job-name={c['name']}",
            f"#SBATCH --account={c['account']}",
            "#SBATCH --time=00:05:00",
            f"#SBATCH --output={LOGS_DIR}/{c['name']}_%j.log",
            "# Component 0: Solver on c23g (GPU hidden)",
            f"#SBATCH --partition={c['partition']}",
            f"#SBATCH --nodes={c['solver_nodes']}",
            "#SBATCH --ntasks-per-node=24",
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --mem-per-cpu=4G",
            "",
            "#SBATCH hetjob",
            "",
            "# Component 1: DL client on c23g (GPU active)",
            f"#SBATCH --partition={c['partition']}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=24",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --mem-per-cpu=4G",
            "",
            f'SCRIPT_DIR="{MINI_APP_DIR}"',
            'cd "${SCRIPT_DIR}"',
            'exec /bin/zsh "${SCRIPT_DIR}/proper_slurm_job.sh"',
        ]

    sbatch_file.write_text("\n".join(lines) + "\n")
    sbatch_file.chmod(0o755)
    return sbatch_file

def submit_case(c: dict) -> str:
    script_path = make_sbatch_script(c)
    
    exports = list(COMMON_EXPORTS)
    exports.extend([
        f"JOB_NAME_ENV={c['name']}",
        f"CPP_ML_CONFIG_ENV={c['config']}",
        f"MPI_RANKS_ENV={c['solver_ranks']}",
        f"RANK_GRID_X_ENV={c['grid_x']}",
        f"RANK_GRID_Z_ENV={c['grid_z']}",
        f"MODEL_PATH_ENV={c['model_path']}",
    ])
    
    if not c["is_gpu"]:
        exports.append("APPLY_CUDA_STUBS_ENV=1")
    else:
        exports.append("HET_SURROGATE_HIDE_GPU=1")
        exports.append(f"GPUS_PER_NODE_ENV={c['gpus']}")

    export_str = ",".join(exports)
    cmd = ["sbatch", "--parsable", f"--export={export_str}", str(script_path)]
    
    res = subprocess.run(cmd, cwd=MINI_APP_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[FAIL] {c['name']} submission failed: {res.stderr.strip()}", file=sys.stderr)
        return ""
    
    jid = res.stdout.strip()
    print(f"[SUBMIT] {c['name']:<28} -> Job ID {jid}")
    return jid

def parse_log(name: str, jid: str) -> dict:
    log_files = list(LOGS_DIR.glob(f"{name}_*.log"))
    if not log_files:
        return {"status": "NO_LOG", "step4": None, "step6": None, "median": None}
    
    # Pick the most recent
    latest = max(log_files, key=os.path.getmtime)
    text = latest.read_text(errors="replace")
    
    step_times = {}
    for line in text.splitlines():
        # Match STEP_TIMING step=4 solver=ML step_ms=...
        m = re.search(r"STEP_TIMING\s+step=(\d+)\s+solver=ML\s+step_ms=([0-9.]+)", line)
        if m:
            st_num = int(m.group(1))
            val = float(m.group(2))
            step_times[st_num] = val
        else:
            # Fallback format: Step 4, ML, local moved: ..., time: 61.7586 ms
            m2 = re.search(r"Step\s+(\d+),\s*ML.*time:\s*([0-9.]+)\s*ms", line)
            if m2:
                st_num = int(m2.group(1))
                val = float(m2.group(2))
                step_times[st_num] = val

    s4 = step_times.get(4)
    s6 = step_times.get(6)
    
    has_error = "ERROR:" in text or "Traceback" in text or "failed with exit code" in text
    passed = (s4 is not None and not has_error)
    
    # Calculate steady median (over steps 4 and 6, excluding step 2)
    steady_vals = [v for k, v in step_times.items() if k > 2]
    med = float(np.median(steady_vals)) if steady_vals else None

    return {
        "status": "COMPLETED" if passed else "FAILED",
        "step4": s4,
        "step6": s6,
        "median": med,
        "log": latest.name
    }

def main():
    target_cases = CASES
    if len(sys.argv) > 1:
        requested = sys.argv[1:]
        target_cases = [c for c in CASES if c["name"] in requested or any(r in c["name"] for r in requested)]
        if not target_cases:
            print(f"No cases matched requested: {requested}", file=sys.stderr)
            return

    print("=" * 75)
    print(f"Submitting {len(target_cases)} PhyDLL Smoke job(s)")
    print("=" * 75)

    submitted = {}
    for c in target_cases:
        jid = submit_case(c)
        if jid:
            submitted[c["name"]] = jid

    if not submitted:
        print("No jobs submitted.", file=sys.stderr)
        return

    print("\nWaiting for jobs to run and complete...")
    remaining = set(submitted.keys())
    
    start_wait = time.time()
    max_wait = 360  # 6 minutes timeout

    while remaining and (time.time() - start_wait < max_wait):
        # Query squeue
        res = subprocess.run(["squeue", "-u", os.environ.get("USER", "ro092286"), "-h", "-o", "%i %j %T"],
                             capture_output=True, text=True)
        active_names = set()
        for line in res.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                for cname in remaining:
                    if cname in parts[1]:
                        active_names.add(cname)

        done = remaining - active_names
        for d in done:
            print(f"[DONE] {d} finished execution.")
            remaining.remove(d)

        if remaining:
            time.sleep(5)

    print("\n" + "=" * 75)
    print("=== SUMMARY RESULTS TABLE ===")
    print("=" * 75)
    print(f"{'Case Name':<28} {'Ratio':<6} {'Dev':<4} {'Status':<10} {'Step 4 (ms)':<12} {'Step 6 (ms)':<12} {'Median (ms)':<12}")
    print("-" * 90)

    for c in target_cases:
        name = c["name"]
        jid = submitted.get(name, "")
        info = parse_log(name, jid)
        ratio = f"{c['solver_nodes']}:1"
        dev = c["device"].upper()
        s4_str = f"{info['step4']:.2f}" if info['step4'] is not None else "N/A"
        s6_str = f"{info['step6']:.2f}" if info['step6'] is not None else "N/A"
        med_str = f"{info['median']:.2f}" if info['median'] is not None else "N/A"
        print(f"{name:<28} {ratio:<6} {dev:<4} {info['status']:<10} {s4_str:<12} {s6_str:<12} {med_str:<12}")

if __name__ == "__main__":
    main()
