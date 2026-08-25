#!/usr/bin/env python3
import subprocess
import time
import sys
import os
import re
from pathlib import Path

MINI_APP_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = Path("/hpcwork/thes2181/mini_app")
LOGS_DIR = MINI_APP_DIR / "logs"
SCOREP_DIR = MINI_APP_DIR / "scorep_runs"
ACCOUNT = "thes2181"

CASES = [
    # 1. CPU Native
    {
        "name": "smoke_cpu_native_smartsim",
        "device": "cpu",
        "scorep": False,
        "provider": "SMARTSIM",
        "config": "config.toml",
        "model": "watercnn",
        "py_dl": 0,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    {
        "name": "smoke_cpu_native_aix",
        "device": "cpu",
        "scorep": False,
        "provider": "AIX",
        "config": "config_aix.toml",
        "model": "perfect_model",
        "py_dl": 0,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    {
        "name": "smoke_cpu_native_phydll_cpp",
        "device": "cpu",
        "scorep": False,
        "provider": "PHYDLL",
        "config": "config_phydll.toml",
        "model": "perfect_model",
        "py_dl": 0,
        "ntasks": 7,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    {
        "name": "smoke_cpu_native_phydll_py",
        "device": "cpu",
        "scorep": False,
        "provider": "PHYDLL",
        "config": "config_phydll.toml",
        "model": "perfect_model",
        "py_dl": 1,
        "ntasks": 7,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    # 2. CPU Score-P
    {
        "name": "smoke_cpu_scorep_smartsim",
        "device": "cpu",
        "scorep": True,
        "provider": "SMARTSIM",
        "config": "config.toml",
        "model": "watercnn",
        "py_dl": 0,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    {
        "name": "smoke_cpu_scorep_aix",
        "device": "cpu",
        "scorep": True,
        "provider": "AIX",
        "config": "config_aix.toml",
        "model": "perfect_model",
        "py_dl": 0,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    {
        "name": "smoke_cpu_scorep_phydll_cpp",
        "device": "cpu",
        "scorep": True,
        "provider": "PHYDLL",
        "config": "config_phydll.toml",
        "model": "perfect_model",
        "py_dl": 0,
        "ntasks": 7,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    {
        "name": "smoke_cpu_scorep_phydll_py",
        "device": "cpu",
        "scorep": True,
        "provider": "PHYDLL",
        "config": "config_phydll.toml",
        "model": "perfect_model",
        "py_dl": 1,
        "ntasks": 7,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    # 3. GPU Native
    {
        "name": "smoke_gpu_native_smartsim",
        "device": "gpu",
        "scorep": False,
        "provider": "SMARTSIM",
        "config": "config.toml",
        "model": "watercnn",
        "py_dl": 0,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    {
        "name": "smoke_gpu_native_aix",
        "device": "gpu",
        "scorep": False,
        "provider": "AIX",
        "config": "config_aix.toml",
        "model": "perfect_model",
        "py_dl": 0,
        "ntasks": 24,
        "mpi_ranks": 24,
        "grid_x": 5,
        "grid_z": 5,
        "width": 60,
        "height": 60,
        "chunk": 12,
    },
    {
        "name": "smoke_gpu_native_phydll_cpp",
        "device": "gpu",
        "scorep": False,
        "provider": "PHYDLL",
        "config": "config_phydll.toml",
        "model": "perfect_model",
        "py_dl": 0,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    {
        "name": "smoke_gpu_native_phydll_py",
        "device": "gpu",
        "scorep": False,
        "provider": "PHYDLL",
        "config": "config_phydll.toml",
        "model": "perfect_model",
        "py_dl": 1,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    # 4. GPU Score-P
    {
        "name": "smoke_gpu_scorep_smartsim",
        "device": "gpu",
        "scorep": True,
        "provider": "SMARTSIM",
        "config": "config.toml",
        "model": "watercnn",
        "py_dl": 0,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    {
        "name": "smoke_gpu_scorep_aix",
        "device": "gpu",
        "scorep": True,
        "provider": "AIX",
        "config": "config_aix.toml",
        "model": "perfect_model",
        "py_dl": 0,
        "ntasks": 24,
        "mpi_ranks": 24,
        "grid_x": 5,
        "grid_z": 5,
        "width": 60,
        "height": 60,
        "chunk": 12,
    },
    {
        "name": "smoke_gpu_scorep_phydll_cpp",
        "device": "gpu",
        "scorep": True,
        "provider": "PHYDLL",
        "config": "config_phydll.toml",
        "model": "perfect_model",
        "py_dl": 0,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    {
        "name": "smoke_gpu_scorep_phydll_py",
        "device": "gpu",
        "scorep": True,
        "provider": "PHYDLL",
        "config": "config_phydll.toml",
        "model": "perfect_model",
        "py_dl": 1,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
    },
    # 5. SmartSim Sharded (per-ml-node)
    {
        "name": "smoke_cpu_native_smartsim_sharded",
        "device": "cpu",
        "scorep": False,
        "provider": "SMARTSIM",
        "config": "config.toml",
        "model": "watercnn",
        "py_dl": 0,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
        "db_layout": "per-ml-node",
        "db_nodes": 2,
    },
    {
        "name": "smoke_cpu_scorep_smartsim_sharded",
        "device": "cpu",
        "scorep": True,
        "provider": "SMARTSIM",
        "config": "config.toml",
        "model": "watercnn",
        "py_dl": 0,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
        "db_layout": "per-ml-node",
        "db_nodes": 2,
    },
    {
        "name": "smoke_gpu_native_smartsim_sharded",
        "device": "gpu",
        "scorep": False,
        "provider": "SMARTSIM",
        "config": "config.toml",
        "model": "watercnn",
        "py_dl": 0,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
        "db_layout": "per-ml-node",
        "db_nodes": 2,
        "template": ".smoke_gpu_template_2ml.sh",
    },
    {
        "name": "smoke_gpu_scorep_smartsim_sharded",
        "device": "gpu",
        "scorep": True,
        "provider": "SMARTSIM",
        "config": "config.toml",
        "model": "watercnn",
        "py_dl": 0,
        "ntasks": 6,
        "mpi_ranks": 6,
        "grid_x": 3,
        "grid_z": 2,
        "width": 216,
        "height": 144,
        "chunk": 12,
        "db_layout": "per-ml-node",
        "db_nodes": 2,
        "template": ".smoke_gpu_template_2ml.sh",
    },
]

def build_sbatch_command(c):
    job_name = c["name"]
    is_gpu = (c["device"] == "gpu")
    is_scorep = c["scorep"]

    compile_path = "build-smoke-cmi-scorep" if is_scorep else "build-smoke-cmi"
    use_scorep = "1" if is_scorep else "0"

    export_vars = [
        "ALL",
        f"USE_SCOREP_ENV={use_scorep}",
        "SKIP_COMPILE_ENV=1",
        f"COMPILE_OUTPUT_PATH_ENV={compile_path}",
        "USE_SMARTSIM=0",
        "USE_CPP_ML_INTERFACE=1",
        "ML_INTERFACE_ENV=cpp",
        f"CPP_ML_INTERFACE_PROVIDER_ENV={c['provider']}",
        f"CPP_ML_CONFIG_ENV={c['config']}",
        f"MODEL_NAME_ENV={c['model']}",
        f"TARGET_WIDTH_ENV={c['width']}",
        f"TARGET_HEIGHT_ENV={c['height']}",
        f"CHUNK_SIZE_ENV={c['chunk']}",
        f"MPI_RANKS_ENV={c['mpi_ranks']}",
        f"RANK_GRID_X_ENV={c['grid_x']}",
        f"RANK_GRID_Z_ENV={c['grid_z']}",
        "TOTAL_STEPS_ENV=2",
        "SAVE_EVERY_ENV=2",
        "FORCE_FRESH_RUN_ENV=1",
        "OVERWRITE_OUTPUT_ENV=1",
        "SKIP_RENDERING_ENV=1",
        "USE_LOCAL_RUNTIME_STAGE_ENV=0",
        "USE_LOCAL_MODEL_CACHE_ENV=1",
        "OVERWRITE_JOB_NAME_ENV=1",
        f"JOB_NAME_ENV={job_name}",
        f"USE_PYTHON_DL_CLIENT={c['py_dl']}",
        "PHYDLL_DL_FIELD_COUNT=1",
        "PHYDLL_SAFE_MPI_ENV=1",
        "PHYDLL_REBUILD_DL_CLIENT_ENV=0",
    ]

    if not is_gpu:
        export_vars.append("APPLY_CUDA_STUBS_ENV=1")

    if c.get("db_layout"):
        export_vars.append(f"DB_LAYOUT_ENV={c['db_layout']}")

    if c.get("db_nodes"):
        export_vars.append(f"DB_NODES_ENV={c['db_nodes']}")

    if is_scorep:
        export_vars.extend([
            "SCOREP_ENABLE_PROFILING_ENV=true",
            "SCOREP_ENABLE_TRACING_ENV=false",
            "SKIP_PAPI_METRICS=1",
            "SCOREP_METRIC_PAPI=",
            f"SCOREP_DIR_TAG_ENV={job_name}",
            f"PHYDLL_DL_BUILD_DIR={MINI_APP_DIR}/../CPP-ML-Interface/dl_clients/build-scorep",
            f"PHYDLL_DL_CLIENT={MINI_APP_DIR}/../CPP-ML-Interface/dl_clients/build-scorep/phydll_dl_client",
        ])
    else:
        export_vars.extend([
            f"PHYDLL_DL_BUILD_DIR={MINI_APP_DIR}/../CPP-ML-Interface/dl_clients/build",
            f"PHYDLL_DL_CLIENT={MINI_APP_DIR}/../CPP-ML-Interface/dl_clients/build/phydll_dl_client",
        ])

    export_str = ",".join(export_vars)

    cmd = [
        "sbatch",
        "--parsable",
        f"--account={ACCOUNT}",
        f"--job-name={job_name}",
        f"--output=logs/{job_name}_%j.log",
        f"--export={export_str}",
    ]

    if not is_gpu:
        cmd.extend([
            f"--ntasks={c['ntasks']}",
            "--nodes=1",
            c.get("template", ".smoke_cpu_template.sh")
        ])
    else:
        cmd.append(c.get("template", ".smoke_gpu_template.sh"))

    return cmd

def submit_all():
    job_ids = {}
    print(f"=== Submitting {len(CASES)} smoke test jobs under account {ACCOUNT} ===")
    for c in CASES:
        cmd = build_sbatch_command(c)
        res = subprocess.run(cmd, cwd=MINI_APP_DIR, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"FAILED to submit {c['name']}: {res.stderr}")
            sys.exit(1)
        jid = res.stdout.strip()
        primary_jid = jid.split("+")[0].split(";")[0].split(",")[0]
        job_ids[c["name"]] = (jid, primary_jid)
        print(f"  Submitted {c['name']:<30} -> Job ID {jid}")
    return job_ids

def check_queue():
    res = subprocess.run(["squeue", "-u", "ro092286", "-h", "-o", "%i %T %j"], capture_output=True, text=True)
    running = {}
    for line in res.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            running[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return running

def verify_results(job_ids):
    print("\n" + "="*80)
    print("=== Verification & Results Matrix ===")
    print("="*80)
    
    results = []
    all_ok = True

    for c in CASES:
        name = c["name"]
        raw_jid, primary_jid = job_ids[name]
        
        # Check logs
        log_files = [p for p in LOGS_DIR.glob(f"{name}_*.log") if re.match(rf"^{re.escape(name)}_\d+\.log$", p.name)]
        log_content = ""
        if log_files:
            latest_log = max(log_files, key=os.path.getmtime)
            log_content = latest_log.read_text()
        
        # Verify ML step
        ml_step_ok = "Step 2, ML" in log_content
        
        # Verify HDF5 trajectory
        traj_h5 = EXTERNAL_DIR / name / "world_trajectory.h5"
        traj_ok = traj_h5.exists() and traj_h5.stat().st_size > 1000
        
        # Verify Score-P cubex
        scorep_ok = True
        cubex_path = None
        if c["scorep"]:
            cubex_files = list(SCOREP_DIR.glob(f"*{name}*/profile.cubex"))
            scorep_ok = len(cubex_files) > 0 and any(f.stat().st_size > 1000 for f in cubex_files)
            if cubex_files:
                cubex_path = cubex_files[0]
        
        # Check errors in log
        has_error = "ERROR:" in log_content or "Traceback" in log_content or "failed with exit code" in log_content
        
        status = "PASSED" if (ml_step_ok and traj_ok and scorep_ok and not has_error) else "FAILED"
        if status != "PASSED":
            all_ok = False
            
        results.append({
            "name": name,
            "jid": primary_jid,
            "device": c["device"].upper(),
            "scorep": "Score-P" if c["scorep"] else "Native",
            "provider": c["provider"] + (" (Py)" if c["py_dl"] else (" (C++)" if c["provider"] == "PHYDLL" else "")),
            "ml_step": ml_step_ok,
            "h5": traj_ok,
            "scorep_cubex": scorep_ok if c["scorep"] else "N/A",
            "status": status
        })

    # Print Table
    header = f"{'Case Name':<35} | {'JID':<8} | {'Dev':<4} | {'Mode':<7} | {'Provider':<15} | {'ML Step':<7} | {'HDF5':<5} | {'Cubex':<6} | {'Status':<7}"
    print(header)
    print("-" * len(header))
    for r in results:
        ml_str = "OK" if r["ml_step"] else "FAIL"
        h5_str = "OK" if r["h5"] else "FAIL"
        cub_str = "OK" if r["scorep_cubex"] is True else ("FAIL" if r["scorep_cubex"] is False else "N/A")
        print(f"{r['name']:<35} | {r['jid']:<8} | {r['device']:<4} | {r['scorep']:<7} | {r['provider']:<15} | {ml_str:<7} | {h5_str:<5} | {cub_str:<6} | {r['status']:<7}")
    print("="*80)
    return all_ok

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-only":
        jids = {}
        for c in CASES:
            name = c["name"]
            log_files = [p for p in LOGS_DIR.glob(f"{name}_*.log") if re.match(rf"^{re.escape(name)}_\d+\.log$", p.name)]
            if log_files:
                latest = max(log_files, key=os.path.getmtime)
                jid = latest.stem.split("_")[-1]
                jids[c["name"]] = (jid, jid)
            else:
                jids[c["name"]] = ("-", "-")
        verify_results(jids)
        sys.exit(0)

    job_ids = submit_all()
    print("\nWaiting for jobs to finish...")
    
    while True:
        running = check_queue()
        active_ours = [name for name, (raw, prim) in job_ids.items() if any(prim in k for k in running)]
        if not active_ours:
            print("All jobs finished!")
            break
        print(f"[{time.strftime('%H:%M:%S')}] {len(active_ours)} active jobs remaining: {', '.join(active_ours[:6])}...")
        time.sleep(15)

    success = verify_results(job_ids)
    if not success:
        sys.exit(1)
