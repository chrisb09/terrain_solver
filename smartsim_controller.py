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

device = "GPU" if use_gpu else "CPU"
queue = "c23g" if use_gpu else "c23ms"

print(f"Using device: {'GPU' if use_gpu else 'CPU'} (device={device}, python_exe={sys.executable}, queue={queue})")

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

from smartsim.experiment import Experiment


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
#                         intra_op_threads=1, # Threads within operations
#                         inter_op_threads=args.cpu_cores_per_node # Threads for parallelism between operations, usually its better to prefer intra-op parallelism for ML models
                         )

db.set_run_arg("export", "ALL")
if args.het_group is not None:
    db.set_run_arg("het-group", args.het_group)
    
#if not use_gpu:
#    db.set_cpus(args.cpu_cores_per_node)

exp.start(db, block=False, summary=True)

time.sleep(5)  # Wait a bit

address = db.get_address()
print(f"DB address: {address}")

if args.hostname_file is not None:
    with open(args.hostname_file, "w") as f:
        f.write(",".join(address))

print(f"Wrote database hostname to file: {args.hostname_file}", flush=True)


# Wait until there's a file indicating the solver is done, then stop the database and clean up the experiment.
done_file = "close_driver_" + env.get("SLURM_JOB_ID", "local") + ".txt"

print(f"Waiting for solver to finish (looking for file: {done_file})...", flush=True)
while not os.path.exists(done_file):
    time.sleep(1)
    print(f"Still waiting for solver to finish...", flush=True)
print("Solver finished, stopping database and cleaning up experiment...", flush=True)


exp.stop(db)

os.remove(done_file)