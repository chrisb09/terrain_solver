#!/usr/bin/env python3
"""Local boxplots for the CPU terrain-solver campaign (warm ML-step times).

Reads the STEP_TIMING lines from the campaign logs (10 steady ML steps x 5 reps
= 50 samples per config) and produces one boxplot per model.

Outputs (local, not committed):
  campaign_figures/boxplots_cpu_campaign.png / .pdf
"""
import glob
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MINI_APP = "/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
LOGS = os.path.join(MINI_APP, "logs")
OUT = os.path.join(MINI_APP, "campaign_figures")
os.makedirs(OUT, exist_ok=True)

ORDER = ["AIX", "PHYDLL_CPP", "SS_DEFAULT",
         "SS_TPQ1_I96_B96", "SS_TPQ12_I8_B96", "SS_TPQ96_I1_B96"]

LABELS = {
    "AIX": "AIx",
    "PHYDLL_CPP": "PhyDLL\n(C++ DL)",
    "SS_DEFAULT": "SmartSim\ndefault",
    "SS_TPQ1_I96_B96": "SmartSim\nTPQ=1\nINTRA=96",
    "SS_TPQ12_I8_B96": "SmartSim\nTPQ=12\nINTRA=8",
    "SS_TPQ96_I1_B96": "SmartSim\nTPQ=96\nINTRA=1",
}

# colour per config: AIX blue, PhyDLL purple, tuned SmartSim green, traps/red-ish
COLORS = {
    "AIX": "#0072B2",
    "PHYDLL_CPP": "#CC79A7",
    "SS_DEFAULT": "#D55E00",
    "SS_TPQ1_I1_B1": "#E69F00",
    "SS_TPQ1_I96_B96": "#009E73",
    "SS_TPQ12_I8_B96": "#56B4E9",
    "SS_TPQ96_I1_B96": "#117733",
}

STEP_RE = re.compile(r"STEP_TIMING step=(\d+) solver=ML step_ms=([0-9.]+)")


def collect(model):
    data = defaultdict(list)
    for cfg in ORDER:
        for path in glob.glob(os.path.join(LOGS, f"campaign_{model}_r*_{cfg}_*.log")):
            with open(path) as fh:
                for line in fh:
                    m = STEP_RE.search(line)
                    if not m:
                        continue
                    step = int(m.group(1))
                    if step == 2:          # warm-up step
                        continue
                    data[cfg].append(float(m.group(2)) / 1000.0)
    return data


def boxplot(ax, data, title):
    series = [data[c] for c in ORDER]
    bp = ax.boxplot(series, patch_artist=True, widths=0.6, showfliers=True,
                    medianprops=dict(color="black", linewidth=1.4),
                    flierprops=dict(marker="d", markersize=3, alpha=0.5))
    for patch, cfg in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COLORS[cfg])
        patch.set_alpha(0.65)
    for i, cfg in enumerate(ORDER):
        vals = data[cfg]
        if vals:
            ax.annotate(f"{st.median(vals):.2f}", (i + 1, max(vals)),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(ORDER) + 1))
    ax.set_xticklabels([LABELS[c] for c in ORDER], fontsize=8)
    ax.set_ylabel("Warm ML-step time (s, log scale)")
    ax.set_title(title)
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.35)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, (model, title) in zip(axes, [("watercnn", "WaterCNN (4K, 96 ranks)"),
                                         ("giant", "Giant MLP (240x192, 96 ranks)")]):
        data = collect(model)
        boxplot(ax, data, title)
        print(f"--- {model} ---")
        for c in ORDER:
            v = data[c]
            print(f"  {c:<18} n={len(v):>2}  median={st.median(v):8.3f}  "
                  f"min={min(v):8.3f}  max={max(v):8.3f}")
    fig.suptitle("CPU terrain-solver campaign: warm ML-step time (50 samples/config)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"boxplots_cpu_campaign.{ext}"), dpi=200)
    print(f"\nWrote {OUT}/boxplots_cpu_campaign.png|.pdf")


if __name__ == "__main__":
    main()
