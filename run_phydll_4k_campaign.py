#!/usr/bin/env python3
"""
Harness to generate, submit, monitor, and harvest separate 4K PhyDLL scaling runs on c23ms + c23g:
Each ratio (1:1, 2:1, 4:1, 8:1, 16:1) is submitted as an independent, right-sized job:
  - s7: 1 CPU node + 1 GPU node (192 solver ranks + 4 DL)
  - s8: 2 CPU nodes + 1 GPU node (288 solver ranks + 4 DL)
  - s9: 4 CPU nodes + 1 GPU node (480 solver ranks + 4 DL)
  - s10: 8 CPU nodes + 1 GPU node (864 solver ranks + 4 DL)
  - s11: 16 CPU nodes + 1 GPU node (1632 solver ranks + 4 DL)
Variants: 'default' and 'readiness'.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

MINI_APP_DIR = Path("/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app")
LOGS_DIR = MINI_APP_DIR / "logs"
SCRIPTS_DIR = MINI_APP_DIR / "tmp_phydll_4k"

SCENARIOS = {
    7: {"cpu_nodes": 1, "mm": 96, "ratio": "1:1"},
    8: {"cpu_nodes": 2, "mm": 192, "ratio": "2:1"},
    9: {"cpu_nodes": 4, "mm": 384, "ratio": "4:1"},
    10: {"cpu_nodes": 8, "mm": 768, "ratio": "8:1"},
    11: {"cpu_nodes": 16, "mm": 1536, "ratio": "16:1"},
}

VARIANTS = ["default", "readiness"]

def generate_script(sid: int, variant: str) -> Path:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = SCENARIOS[sid]
    cpu_nodes = cfg["cpu_nodes"]
    mm = cfg["mm"]
    config_toml = f"config_phydll_4k_{variant}.toml"
    config_tag = f"phydll_{variant}"
    job_name = f"phydll_4k_s{sid}_{variant}"
    script_path = SCRIPTS_DIR / f"{job_name}.sbatch"

    time_limit = "00:18:00" if sid <= 9 else "00:20:00"
    script_content = f"""#!/bin/zsh
#SBATCH --job-name={job_name}
#SBATCH --account=thes2181
#SBATCH --partition=c23ms
#SBATCH --nodes={cpu_nodes}
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time={time_limit}
#SBATCH --output=logs/phydll_4k_s{sid}_{variant}_%j.txt

#SBATCH hetjob

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time={time_limit}

set -uo pipefail

CONFIG_TAG="{config_tag}"
CONFIG_TOML="{config_toml}"
SUITE="hethires3"

MINI_APP_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
cd "${{MINI_APP_DIR}}"
mkdir -p logs

SUMMARY="logs/phydll_summary_${{SLURM_JOB_ID}}.txt"
echo "PhyDLL 4K s{sid} {variant} start=$(date)" > "${{SUMMARY}}"

export BASE_EXPORTS="ALL"
export USE_SCOREP="1"
export USE_SCOREP_ENV="1"
export SCOREP_ENABLE_PROFILING="true"
export SCOREP_ENABLE_PROFILING_ENV="true"
export SCOREP_ENABLE_TRACING="true"
export SCOREP_ENABLE_TRACING_ENV="true"
export SKIP_PAPI_METRICS="1"
export SCOREP_METRIC_PAPI=""
export MLCOUPLING_MEM_LOG_VERBOSE="1"
export MODEL_NAME="watercnn"
export MODEL_NAME_ENV="watercnn"
export TOTAL_STEPS="202"
export TOTAL_STEPS_ENV="202"
export SAVE_EVERY="202"
export SAVE_EVERY_ENV="202"
export TARGET_WIDTH="3840"
export TARGET_WIDTH_ENV="3840"
export TARGET_HEIGHT="2160"
export TARGET_HEIGHT_ENV="2160"
export CHUNK_SIZE="60"
export CHUNK_SIZE_ENV="60"
export ML_BATCH_SIZE="20000000"
export ML_BATCH_SIZE_ENV="20000000"
export SKIP_COMPILE="1"
export SKIP_COMPILE_ENV="1"
export FORCE_FRESH_RUN="1"
export FORCE_FRESH_RUN_ENV="1"
export OVERWRITE_OUTPUT="1"
export OVERWRITE_OUTPUT_ENV="1"
export SKIP_RENDERING="1"
export SKIP_RENDERING_ENV="1"
export USE_LOCAL_RUNTIME_STAGE="0"
export USE_LOCAL_RUNTIME_STAGE_ENV="0"
export USE_LOCAL_MODEL_CACHE="1"
export USE_LOCAL_MODEL_CACHE_ENV="1"
export USE_SMARTSIM="0"
export USE_CPP_ML_INTERFACE="1"
export ML_INTERFACE_ENV="cpp"
export CPP_ML_INTERFACE_PROVIDER_ENV="PHYDLL"
export CPP_ML_CONFIG_ENV="${{CONFIG_TOML}}"
export USE_PYTHON_DL_CLIENT="0"
export PHYDLL_DL_CLIENT="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/dl_clients/build-scorep-pilots/phydll_dl_client"
export COMPILE_OUTPUT_PATH_ENV="build_scorep_cmi_default"
export PHYDLL_LIB_DIR_ENV="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/phydll/build/lib"

sid={sid}
mm={mm}
mm_nodes={cpu_nodes}
gside=96
ngpus=4

export HET_MM_NODES_ENV="${{mm_nodes}}"
export MPI_RANKS_ENV="${{mm}}"
export HET_G_SIDE_TASKS_ENV="${{gside}}"
export GPUS_PER_NODE_ENV="${{ngpus}}"
export RUN_ID_ENV="${{SUITE}}_s${{sid}}_${{CONFIG_TAG}}"
export JOB_NAME_ENV="${{SUITE}}_s${{sid}}_${{CONFIG_TAG}}"
export SCOREP_DIR_TAG_ENV="${{SUITE}}_s${{sid}}_${{CONFIG_TAG}}"
export SMARTSIM_PORT_OFFSET_ENV="${{sid}}"
export PHYDLL_NP_DL_ENV="${{ngpus}}"
export PHYDLL_DL_OVERSUBSCRIBE_ENV="1"

echo "=== HH3 s${{sid}}_${{CONFIG_TAG}} (mm=${{mm}}, mm_nodes=${{mm_nodes}}, gside=${{gside}}, gpus=${{ngpus}}) start $(date) ===" >> "${{SUMMARY}}"
zsh "${{MINI_APP_DIR}}/het_hires_job.sh" >> "${{SUMMARY}}" 2>&1
rc=$?
echo "=== HH3 s${{sid}}_${{CONFIG_TAG}} finished rc=${{rc}} $(date) ===" >> "${{SUMMARY}}"
exit ${{rc}}
"""
    script_path.write_text(script_content)
    script_path.chmod(0o775)
    return script_path

def submit_case(sid: int, variant: str) -> str:
    script_path = generate_script(sid, variant)
    cmd = ["sbatch", "--parsable", str(script_path)]
    cfg = SCENARIOS[sid]
    print(f"[SUBMIT] s{sid} ({cfg['ratio']}, {cfg['cpu_nodes']} CPU + 1 GPU node) - {variant}...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    raw_jid = res.stdout.strip()
    jid = raw_jid.split(";")[0] if ";" in raw_jid else raw_jid
    print(f"  -> Slurm Job ID: {jid}")
    return jid

def parse_scenario_log(scenario: int, variant: str):
    target_tag = f"hethires3_s{scenario}_phydll_{variant}"
    candidates = [LOGS_DIR / f"mini_app_output_{target_tag}.txt"]
    for p in LOGS_DIR.glob(f"*{target_tag}*.txt"):
        if p not in candidates:
            candidates.append(p)

    for p in candidates:
        if p.is_file():
            content = p.read_text(errors="ignore")
            pattern = re.compile(r"^Step (\d+), ML, local moved: [^,]+, time: ([0-9.eE+-]+) ms", re.MULTILINE)
            steps = {}
            for match in pattern.finditer(content):
                st_num = int(match.group(1))
                st_time = float(match.group(2))
                steps[st_num] = st_time

            if steps:
                steady_steps = {k: v for k, v in steps.items() if k > 2}
                med = float(np.median(list(steady_steps.values()))) if steady_steps else None
                return {
                    "log_path": str(p),
                    "total_steps": len(steps),
                    "warmup_step2": steps.get(2),
                    "steady_count": len(steady_steps),
                    "median_ms": med,
                    "steps": steps,
                }
    return None

def print_summary():
    print("=" * 82)
    print("PHYDLL 4K SCALING RESULTS SUMMARY (SEPARATE JOBS)")
    print("=" * 82)
    print(f"| Scenario | Ratio | Nodes | Variant   | Warmup (ms) | Steady Med (ms) | Samples |")
    print(f"|:--------:|:-----:|:-----:|:---------:|:-----------:|:---------------:|:-------:|")
    
    for sid in sorted(SCENARIOS.keys()):
        cfg = SCENARIOS[sid]
        for var in VARIANTS:
            res = parse_scenario_log(sid, var)
            ratio = cfg["ratio"]
            nodes = cfg["cpu_nodes"]
            if res and res["median_ms"] is not None:
                w2 = f"{res['warmup_step2']:.2f}" if res['warmup_step2'] else "N/A"
                med = f"{res['median_ms']:.2f}"
                cnt = res['steady_count']
                print(f"| s{sid:<7d} | {ratio:<5s} | {nodes:<5d} | {var:<9s} | {w2:<11s} | {med:<15s} | {cnt:<7d} |")
            else:
                print(f"| s{sid:<7d} | {ratio:<5s} | {nodes:<5d} | {var:<9s} | {'PENDING':<11s} | {'N/A':<15s} | {0:<7d} |")
    print("=" * 82)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        print_summary()
        return

    # Check if user specified scenarios
    targets = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            # E.g. s8_def, s8, s8_ready, all
            if arg == "all":
                for sid in SCENARIOS:
                    for var in VARIANTS:
                        targets.append((sid, var))
            elif arg in ("s8", "s9", "s10", "s11", "s7"):
                sid = int(arg[1:])
                for var in VARIANTS:
                    targets.append((sid, var))
            else:
                m = re.match(r"^s(\d+)_(def|ready|default|readiness)$", arg)
                if m:
                    sid = int(m.group(1))
                    v = "default" if "def" in m.group(2) else "readiness"
                    targets.append((sid, v))
                else:
                    print(f"Unrecognized target: {arg}")
    else:
        # Default: submit s8, s9, s10, s11 (s7 is already queued!)
        for sid in [8, 9, 10, 11]:
            for var in VARIANTS:
                targets.append((sid, var))

    submitted = {}
    for sid, var in targets:
        key = f"s{sid}_{var}"
        jid = submit_case(sid, var)
        submitted[key] = jid

    print(f"\nSubmitted {len(submitted)} separate scenario jobs.")
    print("Current queue status:")
    cmd = ["squeue", "-u", os.environ.get("USER", ""), "-o", "%.10i %.8P %.25j %.8u %.2t %.10M %.6D %R"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
