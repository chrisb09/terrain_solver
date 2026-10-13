#!/usr/bin/env python3
"""Pin one PhyDLL task within its inherited affinity, then exec its argv."""

import argparse
import os
from pathlib import Path
import socket
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--role", choices=("solver", "dl"), required=True)
    parser.add_argument("--rank-env", choices=("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    try:
        if args.budget not in (24, 96):
            raise ValueError("physical-core budget must be 24 or 96")
        if not command or args.count < 1 or args.offset < 0:
            raise ValueError("command, positive role count and nonnegative offset required")
        local_rank = int(os.environ[args.rank_env])
        index = local_rank - args.offset
        if not 0 <= index < args.count:
            raise ValueError(f"{args.rank_env}={local_rank} outside role range "
                             f"[{args.offset}, {args.offset + args.count})")
        if args.role == "solver" and args.count > args.budget:
            raise ValueError("solver count exceeds physical-core budget")
        allowed = os.sched_getaffinity(0)
        cores = {}
        for cpu in sorted(allowed):
            topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            key = (int((topology / "physical_package_id").read_text()),
                   int((topology / "core_id").read_text()))
            cores.setdefault(key, cpu)
        if len(cores) < args.budget:
            raise ValueError(f"need {args.budget} distinct allowed physical cores, "
                             f"found {len(cores)}; use launcher binding none and a sufficient job/step cpuset")
        # Keep one allowed logical CPU per physical core, in CPU-ID order.
        pool = list(cores.values())[:args.budget]
        chosen = {pool[index if args.role == "solver" else index % args.budget]}
        os.sched_setaffinity(0, chosen)
        effective = os.sched_getaffinity(0)
        if effective != chosen:
            raise ValueError(f"requested mask {chosen}, effective mask {effective}")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "TORCH_NUM_THREADS", "MLCOUPLING_INTRA_OP_THREADS",
                     "MLCOUPLING_INTER_OP_THREADS"):
            os.environ[name] = "1"
        if args.log or os.environ.get("PHYDLL_PIN_LOG_ENV", "0") == "1":
            print(f"PHYDLL_PIN role={args.role} host={socket.gethostname()} "
                  f"rank={os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('SLURM_PROCID', '?'))} "
                  f"local_rank={local_rank} index={index} budget={args.budget} "
                  f"mask={','.join(map(str, sorted(effective)))} intra=1 inter=1",
                  file=sys.stderr, flush=True)
        os.execvp(command[0], command)
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(2, f"PHYDLL_PIN error: role={args.role} host={socket.gethostname()}: {exc}\n")


if __name__ == "__main__":
    main()
