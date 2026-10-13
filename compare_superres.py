#!/usr/bin/env python3
"""A/B/C/D comparison for the super-high-res P2P-vs-collective experiment.

Cells (all WaterCNN, `numa` layout):
  A 24c1g normal          3840x2160 / 6x4   24.88 MB/rank, 1 controller
  B 96c4g normal          3840x2160 / 12x8   6.22 MB/rank, 4 controllers
  C 96c4g super-high-res  7680x4320 / 12x8  24.88 MB/rank, 4 controllers
  D 24c1g super-high-res  7680x4320 / 6x4   99.5  MB/rank, 1 controller

Normal-res cells use `p2p23`, super-res use `p2p24` (equivalent: credits cap at
23 workers); both are mapped to a common `p2p` column.
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

CELLS = ["A", "B", "C", "D"]
CELL_LABELS = {
    "A": "A 24c1g normal\n(24.9MB/rank, 1 ctrl)",
    "B": "B 96c4g normal\n(6.2MB/rank, 4 ctrl)",
    "C": "C 96c4g super-hi\n(24.9MB/rank, 4 ctrl)",
    "D": "D 24c1g super-hi\n(99.5MB/rank, 1 ctrl)",
}
COMM_ALIASES = {"p2p23": "p2p", "p2p24": "p2p"}


def load_numa(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("layout") != "numa":
                continue
            comm = COMM_ALIASES.get(row["comm_mode"], row["comm_mode"])
            try:
                out[comm] = float(row["median_ms"])
            except (KeyError, ValueError):
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--c", type=Path, required=True)
    ap.add_argument("--d", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    paths = {"A": args.a, "B": args.b, "C": args.c, "D": args.d}
    data = {k: load_numa(v) for k, v in paths.items()}

    header = "cell,collective_ms,p2p1_ms,p2p_ms,delta_p2p1_ms,delta_p2p_ms"
    rows = [header]
    print(f"{'cell':4s} {'coll':>9s} {'p2p1':>9s} {'p2p':>9s} {'d(p2p1)':>9s} {'d(p2p)':>9s}")
    for cell in CELLS:
        d = data[cell]
        coll = d.get("coll", float("nan"))
        p1 = d.get("p2p1", float("nan"))
        p = d.get("p2p", float("nan"))
        print(f"{cell:4s} {coll:9.3f} {p1:9.3f} {p:9.3f} {p1-coll:9.3f} {p-coll:9.3f}")
        rows.append(f"{cell},{coll:.4f},{p1:.4f},{p:.4f},{p1-coll:.4f},{p-coll:.4f}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        args.out_csv.write_text("\n".join(rows) + "\n")
        print(f"Wrote {args.out_csv}")

    x = np.arange(len(CELLS))
    width = 0.35
    d1 = [data[c].get("p2p1", float("nan")) - data[c].get("coll", float("nan")) for c in CELLS]
    dp = [data[c].get("p2p", float("nan")) - data[c].get("coll", float("nan")) for c in CELLS]
    fig, ax = plt.subplots(figsize=ts.figure_size(width=1.0, aspect=0.55))
    ax.bar(x - width / 2, d1, width, label="P2P credit=1 - collective", color="#ff7f0e",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, dp, width, label="P2P full credits - collective", color="#2ca02c",
           edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in CELLS], fontsize=ts.TICK)
    ax.set_ylabel("ML step time delta vs. collective [ms]\n(negative = P2P faster)", fontsize=ts.AXIS_LABEL)
    ax.set_title("Super-high-res: P2P delta vs. collective", fontsize=ts.AXIS_TITLE)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=ts.LEGEND)
    fig.tight_layout()
    ts.save(fig, args.out)
    print(f"Wrote {args.out}.pgf, .pdf, and .png via thesis_style")


if __name__ == "__main__":
    main()
