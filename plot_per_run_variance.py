#!/usr/bin/env python3
"""Per-run variance figure for the 5-repeat AIx layout campaign.

Reads the per-run medians CSV written by node_variance_report.py and draws one
panel per layout (default/affine/numa), with the 5 per-run medians per comm
mode as points (annotated with the node), plus the pooled 500-sample median as a
horizontal line. Makes node-level effects visible.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Integrate thesis_style
sys.path.insert(0, str(Path.home() / "Master-Thesis" / "gitlab" / "sourcecode"))
import thesis_style as ts
ts.apply()

import matplotlib.pyplot as plt
import numpy as np

LAYOUTS = ["default", "affine", "numa"]
COMMS = ["coll", "p2p1", "p2p23"]
COMM_LABELS = ["Collective", "P2P (credit=1)", "P2P (full credits)"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-run-csv", type=Path, required=True)
    ap.add_argument("--config-label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.per_run_csv.open()))
    fig, axes = plt.subplots(1, len(LAYOUTS), figsize=ts.figure_size(width=1.0, aspect=0.38), sharey=True)
    for ax, layout in zip(axes, LAYOUTS):
        for ci, comm in enumerate(COMMS):
            vals = sorted(
                ((int(r["repeat"]), float(r["median_ms"]), r.get("nodes", ""))
                 for r in rows
                 if r["layout"] == layout and r["comm"] == comm and r.get("median_ms", "").strip()),
                key=lambda t: t[0],
            )
            if not vals:
                continue
            ys = [v[1] for v in vals]
            xs = np.full(len(ys), ci, dtype=float) + np.linspace(-0.12, 0.12, len(ys))
            ax.scatter(xs, ys, color=COLORS[ci], edgecolor="black", linewidth=0.4,
                       s=28, zorder=4)
            pooled = float(np.median(ys))
            ax.hlines(pooled, ci - 0.28, ci + 0.28, color=COLORS[ci],
                      linewidth=1.8, zorder=3)
            for x, (_rep, y, node) in zip(xs, vals):
                ax.annotate(node or "?", (x, y), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=5.5,
                            rotation=90, color="#333333", zorder=5)
        ax.set_xticks(range(len(COMMS)))
        ax.set_xticklabels(COMM_LABELS, fontsize=ts.TICK)
        ax.set_title(f"layout '{layout}'", fontsize=ts.AXIS_TITLE)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Per-run median ML step time [ms]", fontsize=ts.AXIS_LABEL)
    fig.suptitle(f"{args.config_label}: per-run medians across repeats",
                 fontsize=ts.FIGURE_TITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    ts.save(fig, args.out)
    print(f"Wrote {args.out}.pgf, .pdf, and .png via thesis_style")


if __name__ == "__main__":
    main()
