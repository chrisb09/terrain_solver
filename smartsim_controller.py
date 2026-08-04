#!/bin/python3


import argparse
import shutil
import os
import sys
from os import environ as env
import time
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--db_nodes", type=int, default=1)
parser.add_argument("--use_gpu", action="store_true", help="Use GPU for the experiment")
parser.add_argument("--cpu_cores_per_node", type=int, default=1, help="Number of CPU cores per node to allocate for the database (only relevant if using slurm launcher)")
parser.add_argument("--het_group", default=None, type=str, help="Heterogeneous group to run the database on (if using slurm launcher)")
parser.add_argument("--hostname_file", default=None, type=str, help="File to write the database hostname to")
args = parser.parse_args()

##### Print out the configuration for this run
print("Experiment configuration:")
print(f"  Database nodes: {args.db_nodes}")
print(f"  Use GPU: {args.use_gpu}")
print(f"  CPU cores per node for database: {args.cpu_cores_per_node}")
print(f"  Heterogeneous group: {args.het_group}")
print(f"  Hostname file: {args.hostname_file}")

use_gpu = args.use_gpu


def positive_env_int(name, default):
    value = int(env.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value

device = "GPU" if use_gpu else "CPU"
queue = "c23g" if use_gpu else "c23ms"
intra_op_threads = positive_env_int("SMARTSIM_INTRA_OP_THREADS", 8)
inter_op_threads = positive_env_int("SMARTSIM_INTER_OP_THREADS", 1)
threads_per_queue = positive_env_int(
    "SMARTSIM_THREADS_PER_QUEUE",
    min(8, args.cpu_cores_per_node // 8) if args.cpu_cores_per_node >= 8 else 1,
)

print(f"Using device: {'GPU' if use_gpu else 'CPU'} (device={device}, python_exe={sys.executable}, queue={queue})")
print(
    "Database model execution settings: "
    f"intra_op_threads={intra_op_threads}, inter_op_threads={inter_op_threads}, "
    f"threads_per_queue={threads_per_queue}"
)
db_nodelist = env.get("SMARTSIM_DB_NODELIST", "")
db_exclude_node = env.get("SMARTSIM_DB_EXCLUDE_NODE", "")
if db_nodelist:
    print(f"Database node list: {db_nodelist}")
if db_exclude_node:
    print(f"Database excludes solver node: {db_exclude_node}")

if use_gpu:
    env.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    print("GPU debug environment:")
    print(f"  CUDA_LAUNCH_BLOCKING={env.get('CUDA_LAUNCH_BLOCKING')}")
    print(f"  CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"  SLURM_JOB_GPUS={env.get('SLURM_JOB_GPUS', '<unset>')}")
    print(f"  SLURM_STEP_GPUS={env.get('SLURM_STEP_GPUS', '<unset>')}")
    print(f"  SLURM_GPUS_ON_NODE={env.get('SLURM_GPUS_ON_NODE', '<unset>')}")
    print(f"  SLURM_GPUS_PER_NODE={env.get('SLURM_GPUS_PER_NODE', '<unset>')}")
    print(f"  SLURM_GPUS_PER_TASK={env.get('SLURM_GPUS_PER_TASK', '<unset>')}")

    try:
        import torch
        print("Torch CUDA diagnostics:")
        print(f"  torch.__version__={torch.__version__}")
        print(f"  torch.cuda.is_available()={torch.cuda.is_available()}")
        print(f"  torch.cuda.device_count()={torch.cuda.device_count()}")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            print(f"  torch.cuda.current_device()={torch.cuda.current_device()}")
            for idx in range(torch.cuda.device_count()):
                print(f"  torch.cuda.get_device_name({idx})={torch.cuda.get_device_name(idx)}")
    except Exception as exc:
        print(f"Torch CUDA diagnostics failed: {exc}")

    try:
        result = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True)
        print(f"nvidia-smi -L exit_code={result.returncode}")
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())
    except Exception as exc:
        print(f"nvidia-smi diagnostics failed: {exc}")

env["SMARTSIM_WLM_TRIALS"] = "60"

from smartsim.experiment import Experiment

# Strip Score-P environment variables before creating the experiment.
# The srun command uses `--export ALL` which propagates the full environment;
# Score-P vars in the propagated env can interfere with Slurm step launch.
for key in list(env.keys()):
    if key.startswith("SCOREP_"):
        del env[key]

# Remove Score-P wrapper binaries from PATH so srun's environment is clean.
_scorep_bin = "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/Score-P/8.4-gompi-2022a/bin"
_current_path = env.get("PATH", "")
if _scorep_bin in _current_path:
    env["PATH"] = ":".join(p for p in _current_path.split(":") if p != _scorep_bin)

# Unset Slurm task-count environment variables that could confuse srun when
# SmartSim launches the orchestrator step from within a batch job context.
# SmartSim passes --ntasks=1 explicitly; these env vars must not override it.
for _key in ("SLURM_NTASKS", "SLURM_NPROCS", "SLURM_NTASKS_PER_NODE",
             "SLURM_TASKS_PER_NODE", "SLURM_NPROCS"):
    env.pop(_key, None)

exp_dir = "smartsim_experiments/" + env.get("SLURM_JOB_ID", "local")
if os.path.exists(exp_dir):
    print(f"Cleaning up previous experiment directory: {exp_dir}")
    shutil.rmtree(exp_dir)

exp = Experiment(name=exp_dir, launcher="slurm")


db = exp.create_database(port=6780,
                         interface="ib0",
                         batch=False,
#                         time="00:10:00",
                         single_cmd=False,
                         db_nodes=args.db_nodes,
                          intra_op_threads=intra_op_threads,
                          inter_op_threads=inter_op_threads,
                          threads_per_queue=threads_per_queue
                         )

db.set_run_arg("export", "ALL")
db.set_run_arg("ntasks", str(args.db_nodes))
db.set_run_arg("ntasks-per-node", "1")
if db_nodelist and env.get("SMARTSIM_PIN_DB_NODELIST", "0") == "1":
    db_hosts = db_nodelist.split(",")
    db_settings = [entity.run_settings for entity in db.entities]
    if len(db_hosts) != args.db_nodes or len(db_settings) != args.db_nodes:
        raise RuntimeError(
            f"cannot map {args.db_nodes} database shards to hosts={db_nodelist!r}"
        )
    for host, run_settings in zip(db_hosts, db_settings):
        run_settings.run_args["nodelist"] = host
if db_exclude_node and not (db_nodelist and env.get("SMARTSIM_PIN_DB_NODELIST", "0") == "1"):
    db.set_run_arg("exclude", db_exclude_node)
# Only apply het-group if we are actually in a heterogeneous job allocation.
if args.het_group is not None and (env.get("SLURM_HET_SIZE") or env.get("SLURM_JOB_NUM_NODES_HET_GROUP_0")):
    db.set_run_arg("het-group", args.het_group)

db.set_cpus(max(1, args.cpu_cores_per_node))

import traceback

# DEBUG: print the env we're launching with (excluding too-large values)
print("=== DEBUG: controller environment (key vars) ===", flush=True)
for k in sorted(env.keys()):
    if k.startswith("SLURM_") or k in ("LD_LIBRARY_PATH", "PATH", "PYTHONPATH"):
        v = env[k]
        if len(v) > 500:
            v = v[:500] + "... [truncated]"
        print(f"  {k}={v}", flush=True)

try:
    exp.start(db, block=False, summary=True)
except Exception:
    print("=== DEBUG: exp.start failed ===", flush=True)
    traceback.print_exc()
    # Try sacct to see what steps are registered
    try:
        job_id = env.get("SLURM_JOB_ID", "")
        result = subprocess.run(
            ["sacct", "--noheader", "-p", "--format=jobname,jobid,state,exitcode"],
            capture_output=True, text=True, timeout=15
        )
        print(f"sacct (all) stdout:\n{result.stdout}", flush=True)
        print(f"sacct (all) stderr: {result.stderr}", flush=True)
        if job_id:
            result2 = subprocess.run(
                ["sacct", "-j", job_id, "--format=jobname,jobid,state,exitcode"],
                capture_output=True, text=True, timeout=15
            )
            print(f"sacct (-j {job_id}) stdout:\n{result2.stdout}", flush=True)
            print(f"sacct (-j {job_id}) stderr: {result2.stderr}", flush=True)
    except Exception as sacct_err:
        print(f"sacct diagnostics failed: {sacct_err}", flush=True)
    # Try to run a simple srun to see if srun works at all
    try:
        _srun_cmd = ["srun", "--ntasks=1", "--cpus-per-task=1", "hostname"]
        if args.het_group is not None and env.get("SLURM_HET_SIZE"):
            _srun_cmd = ["srun", f"--het-group={args.het_group}", "--ntasks=1", "--cpus-per-task=1", "hostname"]
        result3 = subprocess.run(_srun_cmd, capture_output=True, text=True, timeout=15)
        print(f"srun hostname stdout: {result3.stdout}", flush=True)
        print(f"srun hostname stderr: {result3.stderr}", flush=True)
        print(f"srun hostname returncode: {result3.returncode}", flush=True)
    except Exception as srun_err:
        print(f"srun diagnostic failed: {srun_err}", flush=True)
    raise

if args.db_nodes > 1:
    # Clustered Redis can temporarily mark peers as failed while TF backend/model
    # initialization stalls event processing on a shard.
    # Apply these settings only after the orchestrator is active.
    timeout_ms = env.get("SMARTSIM_CLUSTER_NODE_TIMEOUT_MS", "120000")
    require_full_coverage = env.get("SMARTSIM_CLUSTER_REQUIRE_FULL_COVERAGE", "no")
    conf_retries = int(env.get("SMARTSIM_DB_CONF_RETRIES", "30"))
    conf_retry_sleep_s = float(env.get("SMARTSIM_DB_CONF_RETRY_SLEEP_S", "1"))

    config_applied = False
    for attempt in range(1, conf_retries + 1):
        try:
            db.set_db_conf("cluster-node-timeout", timeout_ms)
            db.set_db_conf("cluster-require-full-coverage", require_full_coverage)
            print(
                "Applied clustered Redis stability config: "
                f"cluster-node-timeout={timeout_ms}, "
                f"cluster-require-full-coverage={require_full_coverage}"
            )
            config_applied = True
            break
        except Exception as exc:
            print(
                f"Database not ready for set_db_conf yet (attempt {attempt}/{conf_retries}): {exc}",
                flush=True,
            )
            time.sleep(conf_retry_sleep_s)

    if not config_applied:
        print(
            "Warning: proceeding without clustered Redis stability config after retries.",
            flush=True,
        )

time.sleep(5)  # Wait a bit

address = db.get_address()
print(f"DB address: {address}")

if args.hostname_file is not None:
    with open(args.hostname_file, "w") as f:
        f.write(",".join(address))

print(f"Wrote database hostname to file: {args.hostname_file}", flush=True)


# Wait until there's a file indicating the solver is done, then stop the database and clean up the experiment.
run_id = env.get("RUN_ID_ENV") or env.get("SLURM_JOB_ID", "local")
done_file = f"close_driver_{run_id}.txt"

print(f"Waiting for solver to finish (looking for file: {done_file})...", flush=True)
while not os.path.exists(done_file):
    time.sleep(1)
    print(f"Still waiting for solver to finish...", flush=True)
print("Solver finished, stopping database and cleaning up experiment...", flush=True)


exp.stop(db)

os.remove(done_file)
