#!/usr/bin/env python3
"""Prototype: per-rank AIx workgroup assignment via min-cost flow.

Given a measured rank-to-rank transfer-cost matrix (from the affinity
``mpi_pair_latency`` benchmark), pick the optimal partition of ranks into ``k``
workgroups (one per controller) that minimises the total input-transfer cost,
subject to each rank going to exactly one group and the group sizes being as
even as possible (floor/ceil).

Outputs, for debugging:
  * assignment JSON (controllers + rank -> group + per-edge cost)
  * a synthetic log the layout visualiser can consume directly
    (``Rank r is bound to NUMA node n`` / ``Rank r/N will be in group g order o``)
  * a comparison of the optimal cost against the current greedy ``numa`` layout

This is a standalone experiment; it does not touch AIx yet.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

GIB = 1 << 30
LARGE_COST = 10**12
SLIT = [
    [10, 12, 12, 12, 21, 21, 21, 21],
    [12, 10, 12, 12, 21, 21, 21, 21],
    [12, 12, 10, 12, 21, 21, 21, 21],
    [12, 12, 12, 10, 21, 21, 21, 21],
    [21, 21, 21, 21, 10, 12, 12, 12],
    [21, 21, 21, 21, 12, 10, 12, 12],
    [21, 21, 21, 21, 12, 12, 10, 12],
    [21, 21, 21, 21, 12, 12, 12, 10],
]


# --------------------------------------------------------------------------- #
# Min-cost max-flow (successive shortest paths, SPFA; tiny graphs)            #
# --------------------------------------------------------------------------- #
class MinCostFlow:
    def __init__(self, n: int):
        self.n = n
        self.graph: list[list[list[int]]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int, cost: int) -> None:
        self.graph[u].append([v, cap, cost, len(self.graph[v])])
        self.graph[v].append([u, 0, -cost, len(self.graph[u]) - 1])

    def min_cost_flow(self, s: int, t: int, max_flow: int) -> tuple[int, int]:
        total_flow = 0
        total_cost = 0
        while total_flow < max_flow:
            dist = [LARGE_COST] * self.n
            in_queue = [False] * self.n
            prev_v = [-1] * self.n
            prev_e = [-1] * self.n
            dist[s] = 0
            from collections import deque

            queue = deque([s])
            in_queue[s] = True
            while queue:
                u = queue.popleft()
                in_queue[u] = False
                for ei, edge in enumerate(self.graph[u]):
                    v, cap, cost, _ = edge
                    if cap > 0 and dist[u] + cost < dist[v]:
                        dist[v] = dist[u] + cost
                        prev_v[v] = u
                        prev_e[v] = ei
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True
            if dist[t] >= LARGE_COST:
                break
            add = max_flow - total_flow
            v = t
            while v != s:
                add = min(add, self.graph[prev_v[v]][prev_e[v]][1])
                v = prev_v[v]
            v = t
            while v != s:
                edge = self.graph[prev_v[v]][prev_e[v]]
                edge[1] -= add
                self.graph[v][edge[3]][1] += add
                v = prev_v[v]
            total_flow += add
            total_cost += add * dist[t]
        return total_flow, total_cost


def solve_assignment(costs: list[list[int]], controllers: list[int], capacities: list[int]):
    """Return (group_of, total_cost) with controllers pinned to their group."""
    n = len(costs)
    k = len(controllers)
    controller_group = {c: g for g, c in enumerate(controllers)}

    # Pre-assign controllers (cost 0) and shrink their group's capacity.
    reduced = {c: g for c, g in controller_group.items()}
    cap = list(capacities)
    for c, g in controller_group.items():
        cap[g] -= 1

    source = 0
    rank_base = 1
    group_base = rank_base + n
    sink = group_base + k
    mcf = MinCostFlow(sink + 1)
    for r in range(n):
        if r in reduced:
            continue
        mcf.add_edge(source, rank_base + r, 1, 0)
        for g in range(k):
            if costs[r][g] >= LARGE_COST:
                continue
            mcf.add_edge(rank_base + r, group_base + g, 1, costs[r][g])
    for g in range(k):
        mcf.add_edge(group_base + g, sink, cap[g], 0)

    need = n - len(controllers)
    flow, cost = mcf.min_cost_flow(source, sink, need)
    if flow < need:
        raise RuntimeError(f"infeasible assignment: {flow}/{need}")

    # Recover the assignment from the residual graph.
    group_of = dict(reduced)
    for r in range(n):
        if r in reduced:
            continue
        node = rank_base + r
        for edge in mcf.graph[node]:
            v, cap_left, _, _ = edge
            if group_base <= v < group_base + k and cap_left == 0:
                group_of[r] = v - group_base
    return group_of, cost


def legacy_slit_greedy(controllers, capacities, numas):
    """Replicate the current C++ NUMA-local rule: nearest controller by SLIT
    distance with the upstream single-host tie-break (distance, group)."""
    n = len(numas)
    k = len(controllers)
    members = [1] * k
    group_of = {c: g for g, c in enumerate(controllers)}
    for r in range(n):
        if r in group_of:
            continue
        best_key = None
        best = -1
        for g, ctrl in enumerate(controllers):
            if members[g] >= capacities[g]:
                continue
            key = (SLIT[numas[r]][numas[ctrl]], g)
            if best_key is None or key < best_key:
                best_key = key
                best = g
        group_of[r] = best
        members[best] += 1
    return group_of


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype AIx per-rank assignment")
    parser.add_argument("--csv", type=Path, required=True,
                        help="mpi_pair_latency CSV with rank_i,rank_j,bandwidth_median_gbps")
    parser.add_argument("--controllers", type=str, default="0,2,4,6")
    parser.add_argument("--payload", type=int, default=6_220_800)
    parser.add_argument("--cost", choices=["latency", "bandwidth"], default="latency")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    ranks: set[int] = set()
    bw: dict[tuple[int, int], float] = {}
    lat: dict[tuple[int, int], float] = {}
    cpu: dict[int, int] = {}
    with args.csv.open() as fh:
        for row in csv.DictReader(fh):
            i, j = int(row["rank_i"]), int(row["rank_j"])
            ranks.update((i, j))
            cpu[i] = int(row["cpu_i"])
            bw[(i, j)] = float(row["bandwidth_median_gbps"])
            lat[(i, j)] = float(row["latency_median_us"])

    n = len(ranks)
    max_rank = max(ranks)
    if max_rank + 1 != n:
        raise SystemExit(f"expected contiguous ranks 0..{n-1}, got max {max_rank}")
    controllers = [int(x) for x in args.controllers.split(",")]
    k = len(controllers)

    def pair_bw(i: int, j: int) -> float:
        for key in ((i, j), (j, i)):
            value = bw.get(key)
            if value and value > 0:
                return value
        return 0.0

    def pair_lat(i: int, j: int) -> float:
        for key in ((i, j), (j, i)):
            value = lat.get(key)
            if value and value > 0:
                return value
        return 0.0

    # Integer costs (nanoseconds) either from the measured latency (topology
    # surrogate) or from payload / achieved bandwidth.
    costs_us = [[0] * k for _ in range(n)]
    for r in range(n):
        for g, c in enumerate(controllers):
            if r == c:
                costs_us[r][g] = 0
            elif args.cost == "latency":
                latency = pair_lat(r, c)
                costs_us[r][g] = int(round(latency * 1000)) if latency else LARGE_COST
            else:
                band = pair_bw(r, c)
                costs_us[r][g] = (int(round(args.payload / band / GIB * 1e6))
                                  if band else LARGE_COST)

    base = n // k
    remainder = n % k
    capacities = [base + (1 if g < remainder else 0) for g in range(k)]

    best = None
    for extra in itertools.combinations(range(k), remainder):
        cap = [base + (1 if g in extra else 0) for g in range(k)]
        try:
            group_of, cost = solve_assignment(costs_us, controllers, cap)
        except RuntimeError:
            continue
        if best is None or cost < best[0]:
            best = (cost, group_of, cap)
    if best is None:
        raise SystemExit("no feasible assignment found")
    opt_cost, opt_group, opt_cap = best

    numas = [cpu.get(r, 0) // 12 for r in range(n)]
    legacy_group = legacy_slit_greedy(controllers, capacities, numas)
    legacy_cost = sum(costs_us[r][legacy_group[r]] for r in range(n)
                      if r not in controllers)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    assignment = {
        "controllers": controllers,
        "group_sizes": {str(g): sum(1 for r in opt_group.values() if r == g)
                        for g in range(k)},
        "optimal_cost_us": opt_cost,
        "legacy_slit_cost_us": legacy_cost,
        "improvement_pct": round(100.0 * (legacy_cost - opt_cost) / legacy_cost, 2),
        "groups": {str(r): opt_group[r] for r in range(n)},
    }
    (args.out_dir / "numa_prototype_assignment.json").write_text(json.dumps(assignment, indent=2))

    # Synthetic log for plot_aix_rank_layout.py
    log_lines = []
    for r in range(n):
        log_lines.append(f"Rank {r} is bound to NUMA node {cpu.get(r, 0) // 12}")
    for r in range(n):
        order = 0 if r in controllers else r + k
        log_lines.append(f"Rank {r}/{n} will be in group {opt_group[r]} order {order}")
    (args.out_dir / "numa_prototype_groups.txt").write_text("\n".join(log_lines) + "\n")

    print(f"ranks={n} controllers={controllers} capacities={opt_cap}")
    print(f"optimal total cost = {opt_cost} us   "
          f"legacy SLIT greedy = {legacy_cost} us   "
          f"improvement = {100.0 * (legacy_cost - opt_cost) / legacy_cost:.1f}%")


if __name__ == "__main__":
    main()
