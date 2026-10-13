#!/usr/bin/env python3
"""NUMA-level AIx workgroup assignment from measured costs.

Consumes the JSON produced by ``numa_measure.py`` and solves an integer
transportation problem: each NUMA domain contributes its whole set of cores,
split across the workgroups (``x[numa][group]`` integer, summing to the number
of cores in the domain), such that every workgroup gets its balanced capacity.
The edge cost is the measured payload-incast time from the NUMA to that
controller, scaled per core.

Outputs the usual assignment JSON + a synthetic log for the layout visualiser,
and compares against the legacy SLIT-based rule.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numa_assign_prototype import SLIT, LARGE_COST, MinCostFlow  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    doc = json.loads(args.measure_json.read_text())
    controllers = doc["controllers"]
    numa_of_rank = doc["numa_of_rank"]
    cores_per_numa = doc["cores_per_numa"]
    incast = doc["incast_numa_ctrl"]
    latency = doc["latency_rank_ctrl"]

    n = len(numa_of_rank)
    k = len(controllers)
    numas = len(incast)
    ranks_in_numa: dict[int, list[int]] = defaultdict(list)
    for r, numa in enumerate(numa_of_rank):
        ranks_in_numa[numa].append(r)

    # Per-core cost from NUMA n to controller g = measured incast / cores.
    cost = [[int(round(incast[n][g] / cores_per_numa * 1e6)) for g in range(k)]
            for n in range(numas)]

    base = n // k
    remainder = n % k

    best = None
    for extra in itertools.combinations(range(k), remainder):
        cap_true = [base + (1 if g in extra else 0) for g in range(k)]
        # Pin each controller's NUMA to its own workgroup (whole domain when it
        # fits, otherwise as many cores as the group can take).
        supply = [cores_per_numa] * numas
        pinned = [[0] * k for _ in range(numas)]
        cap_flow = list(cap_true)
        for g, c in enumerate(controllers):
            nn = numa_of_rank[c]
            p = min(supply[nn], cap_flow[g])
            if p <= 0:
                continue
            pinned[nn][g] += p
            supply[nn] -= p
            cap_flow[g] -= p
        demand = sum(cap_flow)
        pinned_cost = sum(pinned[nn][g] * cost[nn][g]
                          for nn in range(numas) for g in range(k))

        source = 0
        numa_base = 1
        group_base = numa_base + numas
        sink = group_base + k
        mcf = MinCostFlow(sink + 1)
        for nn in range(numas):
            mcf.add_edge(source, numa_base + nn, supply[nn], 0)
            for g in range(k):
                mcf.add_edge(numa_base + nn, group_base + g, supply[nn], cost[nn][g])
        for g in range(k):
            mcf.add_edge(group_base + g, sink, cap_flow[g], 0)
        try:
            flow, flow_cost = mcf.min_cost_flow(source, sink, demand)
        except Exception:  # noqa: BLE001
            continue
        if flow < demand:
            continue
        total = pinned_cost + flow_cost
        if best is None or total < best[0]:
            x = [list(row) for row in pinned]
            for nn in range(numas):
                for edge in mcf.graph[numa_base + nn]:
                    v, cap_left, _, _ = edge
                    if group_base <= v < group_base + k:
                        x[nn][v - group_base] += supply[nn] - cap_left
            best = (total, x, cap_true)
    if best is None:
        raise SystemExit("infeasible NUMA-level assignment")
    opt_total, opt_x, opt_cap = best

    # Reconstruct rank -> group (controllers first, then the remaining cores).
    group_of: dict[int, int] = {}
    for nn in range(numas):
        ranks = sorted(ranks_in_numa[nn])
        assigned = [False] * len(ranks)
        counts = list(opt_x[nn])
        for c in [c for c in controllers if numa_of_rank[c] == nn]:
            g = controllers.index(c)
            group_of[c] = g
            assigned[ranks.index(c)] = True
            counts[g] -= 1
        idx = 0
        for g in range(k):
            for _ in range(max(0, counts[g])):
                while idx < len(ranks) and assigned[idx]:
                    idx += 1
                if idx >= len(ranks):
                    break
                group_of[ranks[idx]] = g
                assigned[idx] = True
                idx += 1

    # Legacy SLIT greedy at the NUMA level, evaluated on the same cost.
    def legacy_cost():
        members = [0] * k
        numa_to_group: dict[int, int] = {}
        total = 0
        for g, c in enumerate(controllers):
            nn = numa_of_rank[c]
            numa_to_group[nn] = g
            members[g] += cores_per_numa
            total += cores_per_numa * cost[nn][g]
        for nn in range(numas):
            if nn in numa_to_group:
                continue
            best_key = None
            best_g = -1
            for g, c in enumerate(controllers):
                if members[g] + cores_per_numa > opt_cap[g]:
                    continue
                key = (SLIT[nn][numa_of_rank[c]], g)
                if best_key is None or key < best_key:
                    best_key, best_g = key, g
            if best_g < 0:
                return LARGE_COST
            numa_to_group[nn] = best_g
            members[best_g] += cores_per_numa
            total += cores_per_numa * cost[nn][best_g]
        return total

    legacy = legacy_cost()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "numa_assignment.json").write_text(json.dumps({
        "controllers": controllers,
        "optimal_cost_us": opt_total,
        "legacy_slit_cost_us": legacy,
        "improvement_pct": round(100.0 * (legacy - opt_total) / legacy, 2),
        "x_numas_x_groups": opt_x,
        "groups": {str(r): group_of[r] for r in range(n)},
    }, indent=2))

    lines = [f"Rank {r} is bound to NUMA node {numa_of_rank[r]}" for r in range(n)]
    lines += [f"Rank {r}/{n} will be in group {group_of[r]} order "
              f"{0 if r in controllers else r + k}" for r in range(n)]
    (args.out_dir / "numa_assignment_groups.txt").write_text("\n".join(lines) + "\n")

    print(f"ranks={n} numas={numas} controllers={controllers} capacities={opt_cap}")
    print(f"optimal = {opt_total} us   legacy SLIT = {legacy} us   "
          f"improvement = {100.0 * (legacy - opt_total) / legacy:.1f}%")


if __name__ == "__main__":
    main()
