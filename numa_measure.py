#!/usr/bin/env python3
"""Measure the ingredients for the NUMA-level AIx workgroup assignment.

On one node with N ranks pinned one-per-core:
  * latency: tiny-message (8 B) round-trip time from every rank to every
    controller (median over repeats);
  * incast: for every (NUMA domain, controller), all ranks of that NUMA send
    the AIx payload to the controller simultaneously; the controller measures
    the time until the last sender completes.

Measurement hygiene (learned from the first run's artefact):
  * send and receive buffers are allocated once and reused (the first version
    allocated ~2.4 GB per repeat per rank, which perturbed the timings);
  * only the controller allocates receive buffers;
  * the (controller, NUMA) order is rotated/shuffled per repeat so the first
    combination is not always the same cold measurement;
  * median and minimum over repeats are reported.

Rank 0 writes a JSON with ``latency_rank_ctrl[rank][ctrl]``,
``incast_numa_ctrl[numa][ctrl]``, ``numa_of_rank`` and ``cores_per_numa``.

This is a standalone experiment; it does not touch AIx.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from pathlib import Path

from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.rank
SIZE = COMM.size


def median_or_zero(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def min_or_zero(values: list[float]) -> float:
    return min(values) if values else 0.0


def current_cpu() -> int:
    getter = getattr(os, "sched_getcpu", None)
    if getter is not None:
        return getter()
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("Cpus_allowed_list:"):
                first = line.split(":", 1)[1].strip().split(",")[0]
                return int(first.split("-")[0])
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", type=str, default="0,24,48,72")
    parser.add_argument("--payload", type=int, default=6_220_800)
    parser.add_argument("--small", type=int, default=8)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--numas", type=int, default=8)
    parser.add_argument("--order", choices=["fixed", "rotate", "shuffle"], default="rotate")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    controllers = [int(x) for x in args.controllers.split(",")]
    k = len(controllers)
    cpu = current_cpu()
    cores_per_numa = SIZE // args.numas
    my_numa = cpu // cores_per_numa
    numa_of_rank = COMM.allgather(my_numa)
    ranks_in_numa: dict[int, list[int]] = {}
    for r, nn in enumerate(numa_of_rank):
        ranks_in_numa.setdefault(nn, []).append(r)

    small_buf = bytearray(args.small)
    send_payload = bytearray(args.payload)
    my_combos = [(ci, nn) for ci in range(k) for nn in range(args.numas)]
    # controller reuses one receive buffer per possible sender (NUMA size)
    recv_bufs = [bytearray(args.payload) for _ in range(cores_per_numa)]

    # ---------------- tiny-message latency (rank -> controller) ------------
    lat_samples: list[list[float]] = [[] for _ in range(k)]
    for rep in range(args.warmup + args.reps):
        for ci, ctrl in enumerate(controllers):
            for src in range(SIZE):
                COMM.Barrier()
                dt = None
                if RANK == src and RANK != ctrl:
                    t0 = MPI.Wtime()
                    COMM.Send(small_buf, dest=ctrl, tag=1)
                    COMM.Recv(small_buf, source=ctrl, tag=2)
                    dt = (MPI.Wtime() - t0) / 2.0
                elif RANK == ctrl and src != ctrl:
                    COMM.Recv(small_buf, source=src, tag=1)
                    COMM.Send(small_buf, dest=src, tag=2)
                COMM.Barrier()
                if rep >= args.warmup and RANK == src and dt is not None:
                    lat_samples[ci].append(dt)

    # ---------------- payload incast (NUMA -> controller) ------------------
    incast_samples: list[list[list[float]]] = [[[] for _ in range(k)]
                                               for _ in range(args.numas)]
    for rep in range(args.warmup + args.reps):
        order = list(my_combos)
        if args.order == "rotate":
            shift = rep % len(order)
            order = order[shift:] + order[:shift]
        elif args.order == "shuffle":
            seed = COMM.bcast(random.randrange(1 << 30) if RANK == 0 else None, root=0)
            random.Random(seed).shuffle(order)
        for ci, nn in order:
            ctrl = controllers[ci]
            active = [r for r in ranks_in_numa[nn] if r != ctrl]
            COMM.Barrier()
            t0 = MPI.Wtime()
            dt = None
            if RANK == ctrl:
                reqs = [COMM.Irecv(recv_bufs[i], source=r, tag=10)
                        for i, r in enumerate(active)]
                for req in reqs:
                    req.Wait()
                dt = MPI.Wtime() - t0
            elif RANK in active:
                COMM.Send(send_payload, dest=ctrl, tag=10)
            COMM.Barrier()
            if rep >= args.warmup and RANK == ctrl:
                incast_samples[nn][ci].append(dt)

    # ---------------- collect and write ------------------------------------
    lat_row = [median_or_zero(lat_samples[ci]) for ci in range(k)]
    lat_rows = COMM.gather(lat_row, root=0)
    inc_local = [[median_or_zero(incast_samples[nn][ci]) for ci in range(k)]
                 for nn in range(args.numas)]
    inc_local_min = [[min_or_zero(incast_samples[nn][ci]) for ci in range(k)]
                     for nn in range(args.numas)]
    inc_rows = COMM.gather(
        ((RANK, inc_local, inc_local_min) if RANK in controllers else None), root=0)

    if RANK == 0:
        ctrl_index = {c: g for g, c in enumerate(controllers)}
        incast_final = [[0.0] * k for _ in range(args.numas)]
        incast_final_min = [[0.0] * k for _ in range(args.numas)]
        for item in inc_rows:
            if item is None:
                continue
            controller_rank, table, table_min = item
            g = ctrl_index[controller_rank]
            for nn in range(args.numas):
                incast_final[nn][g] = table[nn][g]
                incast_final_min[nn][g] = table_min[nn][g]
        latency_final = [list(row) for row in lat_rows]
        doc = {
            "controllers": controllers,
            "numa_of_rank": numa_of_rank,
            "cores_per_numa": cores_per_numa,
            "latency_rank_ctrl": latency_final,
            "incast_numa_ctrl": incast_final,
            "incast_numa_ctrl_min": incast_final_min,
            "payload_bytes": args.payload,
            "order": args.order,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc, indent=2))
        print(f"wrote {args.out}")
        print("incast median [us] (rows=NUMA, cols=controllers):")
        for nn in range(args.numas):
            print(f"  N{nn}: " + " ".join(f"{v*1e6:9.1f}" for v in incast_final[nn]))


if __name__ == "__main__":
    main()
