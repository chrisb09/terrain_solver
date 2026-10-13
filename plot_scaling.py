#!/usr/bin/env python3
"""Combined strong-scaling figure for the AIx core/GPU campaign.

Reads one per-config summary CSV per (series, gpu-count) point (as written by
analyze_matrix_results.py) and draws, for each layout, two panels (single-node
and heterogeneous) with one line per communication mode: median ML step time
vs number of GPUs.

Points CSV columns: series,gpus,cfg,summary_csv
  series in {single, het}; summary_csv may be relative to CWD.
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

COMMS = ["coll", "p2p1", "p2p23"]
COMM_LABELS = {"coll": "Collective", "p2p1": "P2P (credit=1)", "p2p23": "P2P (full credits)"}
COLORS = {"coll": "#1f77b4", "p2p1": "#ff7f0e", "p2p23": "#2ca02c"}
LAYOUTS = ["default", "affine", "numa"]


def load_summary(path: Path) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                out[(row["layout"], row["comm_mode"])] = float(row["median_ms"])
            except (KeyError, ValueError):
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points-csv", type=Path, required=True)
    ap.add_argument("--config-label", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--single-label", default="single c23g node")
    ap.add_argument("--het-label", default="het c23mm + c23g")
    args = ap.parse_args()

    points = list(csv.DictReader(args.points_csv.open()))
    data: dict[str, dict[str, dict[int, dict[tuple[str, str], float]]]] = {
        "single": {}, "het": {}}
    for p in points:
        series = p["series"].strip()
        if series not in data:
            continue
        gpus = int(p["gpus"])
        data[series].setdefault(p["cfg"].strip(), {})[gpus] = load_summary(Path(p["summary_csv"].strip()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for layout in LAYOUTS:
        fig, axes = plt.subplots(1, 2, figsize=ts.figure_size(width=1.0, aspect=0.45), sharey=True)
        for ax, series in zip(axes, ["single", "het"]):
            for comm in COMMS:
                xs, ys = [], []
                for cfg, by_gpu in sorted(data[series].items()):
                    for gpus, summ in by_gpu.items():
                        if (layout, comm) in summ:
                            xs.append(gpus)
                            ys.append(summ[(layout, comm)])
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                xs = [xs[i] for i in order]
                ys = [ys[i] for i in order]
                if xs:
                    ax.plot(xs, ys, marker="o", color=COLORS[comm], label=COMM_LABELS[comm])
            ax.set_title(args.single_label if series == "single" else args.het_label, fontsize=ts.AXIS_TITLE)
            ax.set_xlabel("GPU Count", fontsize=ts.AXIS_LABEL)
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels([1, 2, 3, 4], fontsize=ts.TICK)
            ax.grid(True, linestyle="--", alpha=0.4)
        axes[0].set_ylabel("Median ML step time [ms]", fontsize=ts.AXIS_LABEL)
        axes[0].legend(loc="best", fontsize=ts.LEGEND)
        fig.suptitle(f"{args.config_label}: layout '{layout}'", fontsize=ts.FIGURE_TITLE)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = args.out_dir / f"aix_scaling_{layout}"
        ts.save(fig, out)
        print(f"Wrote {out}.pgf, .pdf, and .png via thesis_style")


if __name__ == "__main__":
    main()
