#!/usr/bin/env python3
"""Regenerate the AIx layout-matrix comparison figure from job logs.

Reads STEP_TIMING ML steps for each of the 3x3 cells -- either one job per cell
(``--job-map``) or several runs per cell (``--runs-csv``, 5-repeat campaign) --
draws the grouped bar chart (medians) plus the per-cell distribution boxplots,
and writes PNG + PDF.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from statistics import median

# Integrate thesis_style for publication-quality PGF/PDF export
sys.path.insert(0, str(Path.home() / "Master-Thesis" / "gitlab" / "sourcecode"))
import thesis_style as ts
ts.apply()

import matplotlib.pyplot as plt
import numpy as np

STEP_RE = re.compile(r"STEP_TIMING step=(\d+) solver=(\S+) step_ms=([0-9.]+)")
LAYOUTS = ["default", "affine", "numa"]
COMMS = ["coll", "p2p1", "p2p23"]
COMM_LABELS = ["Collective", "P2P (credit=1)", "P2P (full credits)"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def ml_steps(path: Path, skip_first: int = 1) -> list[float]:
    steps = []
    for line in path.read_text(errors="replace").splitlines():
        match = STEP_RE.search(line)
        if match and match.group(2) == "ML":
            steps.append(float(match.group(3)))
    return steps[skip_first:]


def parse_runs_csv(paths: list[Path], include_failed: bool = False) -> dict[tuple[str, str], list[str]]:
    cells: dict[tuple[str, str], list[str]] = {}
    for path in paths:
        if not path.exists():
            continue
        lines = path.read_text(errors="replace").splitlines()
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 6:
                continue
            run_id, _jobid, _repeat, layout, comm, rc = (p.strip() for p in parts[:6])
            if not include_failed and rc not in ("0", ""):
                continue
            if comm == "p2p24":  # super-high-res alias of full credits
                comm = "p2p23"
            cells.setdefault((layout, comm), []).append(run_id)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-label", required=True)
    parser.add_argument("--logs-dir", type=Path, default=Path("mini_app/logs"))
    parser.add_argument("--job-map", default=None,
                        help="comma-separated key=jobid, keys like default_coll")
    parser.add_argument("--runs-csv", type=Path, nargs="*", default=None,
                        help="campaign manifest CSVs (run_id_cell,jobid,repeat,layout,comm,rc)")
    parser.add_argument("--include-failed-runs", action="store_true")
    parser.add_argument("--skip-warmup", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data: dict[tuple[str, str], list[float]] = {}
    if args.runs_csv:
        runs_cells = parse_runs_csv(args.runs_csv, include_failed=args.include_failed_runs)
        for (layout, comm), run_ids in runs_cells.items():
            vals: list[float] = []
            for run_id in run_ids:
                vals.extend(ml_steps(args.logs_dir / f"mini_app_output_{run_id}.txt",
                                     skip_first=args.skip_warmup))
            data[(layout, comm)] = vals
    elif args.job_map:
        job_map = {}
        for item in args.job_map.split(","):
            key, value = item.strip().split("=")
            job_map[key] = int(value)
        for layout in LAYOUTS:
            for comm in COMMS:
                jid = job_map.get(f"{layout}_{comm}")
                if jid is not None:
                    data[(layout, comm)] = ml_steps(args.logs_dir / f"mini_app_output_{jid}.txt",
                                                    skip_first=args.skip_warmup)
    else:
        parser.error("provide --job-map or --runs-csv")

    active_comms = [c for c in COMMS if any(data.get((l, c)) for l in LAYOUTS)]
    if not active_comms:
        active_comms = COMMS
    n_comm = len(active_comms)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=ts.figure_size(width=1.0, aspect=0.45),
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(len(LAYOUTS))
    width = 0.75 / max(1, n_comm)
    for i, comm in enumerate(active_comms):
        comm_idx = COMMS.index(comm) if comm in COMMS else i
        meds = [median(data[(l, comm)]) if data.get((l, comm)) else float("nan")
                for l in LAYOUTS]
        offset = (i - (n_comm - 1) / 2.0) * width
        rects = ax1.bar(x + offset, meds, width, label=COMM_LABELS[comm_idx],
                        color=COLORS[comm_idx], edgecolor="black", linewidth=0.5)
        for rect, value in zip(rects, meds):
            if np.isfinite(value):
                ax1.annotate(f"{value:.1f}", (rect.get_x() + rect.get_width() / 2, value),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", va="bottom", fontsize=ts.ANNOT, weight="bold")
    n_runs = max((len(v) for v in data.values()), default=0)
    ax1.set_ylabel("Median ML step time [ms]", fontsize=ts.AXIS_LABEL)
    ax1.set_title(f"{args.config_label}: ML step time", fontsize=ts.AXIS_TITLE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(LAYOUTS, fontsize=ts.TICK)
    ax1.legend(loc="upper right", fontsize=ts.LEGEND)
    ax1.set_ylim(0, max((median(v) for v in data.values() if v), default=1) * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    positions, values, colors = [], [], []
    xtick_positions = []
    pos = 1.0
    for layout in LAYOUTS:
        layout_positions = []
        for comm in active_comms:
            comm_idx = COMMS.index(comm) if comm in COMMS else 0
            if data.get((layout, comm)):
                values.append(data[(layout, comm)])
                positions.append(pos)
                layout_positions.append(pos)
                colors.append(COLORS[comm_idx])
            pos += 1
        if layout_positions:
            xtick_positions.append(sum(layout_positions) / len(layout_positions))
        else:
            xtick_positions.append(pos - 1)
        pos += 0.8
    if values:
        bp = ax2.boxplot(values, positions=positions, widths=0.7, patch_artist=True,
                         showfliers=False, medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
    ax2.set_title(f"Distribution ({n_runs} samples)", fontsize=ts.AXIS_TITLE)
    ax2.set_xticks(xtick_positions)
    ax2.set_xticklabels(LAYOUTS, fontsize=ts.TICK)
    ax2.set_ylabel("ML step time [ms]", fontsize=ts.AXIS_LABEL)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    ts.save(fig, args.out)
    print(f"Wrote {args.out}.pgf, .pdf, and .png via thesis_style")


if __name__ == "__main__":
    main()
