#!/usr/bin/env python3
"""HH3 scaling plots: step time and inference-device time vs CPU:GPU ratio (s8-s11).

Reads the same artifacts as build_hh3_overview.py and renders one panel per
AIx layout (default/affine/numa) with one line per library configuration.
Writes scaling_step_time.png and scaling_inference_device.png into
results/hethires3/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hh3_overview import (ALL_SCENS, AIX_LIBS, LAYOUTS, SS_LIBS,  # noqa: E402
                               metric_for, SHORT)
from hh3_render import save_figure, close_figure, PGF  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if not PGF:
    matplotlib.use("Agg")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgf", action="store_true",
                    help="Write .pgf/.pdf via the thesis style instead of .png.")
    ap.parse_args()
    out_dir = Path("results") / "hethires3"

SCENS = [8, 9, 10, 11]
MM_RANKS = {8: 192, 9: 384, 10: 768, 11: 1536}
RATIO = {s: MM_RANKS[s] // 96 for s in SCENS}

AIX_COLORS = {"aix_coll": "tab:blue", "aix_p2p": "tab:orange",
              "aix_p2p_fullcredits": "tab:green"}
AIX_MARKERS = {"aix_coll": "o", "aix_p2p": "s", "aix_p2p_fullcredits": "^"}
SS_COLORS = {"smartsim_c0": "tab:red", "smartsim_c3": "tab:purple",
             "smartsim_per_gpu_db": "tab:brown"}
SS_MARKERS = {"smartsim_c0": "v", "smartsim_c3": "D", "smartsim_per_gpu_db": "P"}


def render(metric, title, ylabel, out_path):
    fig, axes = plt.subplots(1, len(LAYOUTS), figsize=(15, 4.4), sharey=True)
    for ax, layout in zip(axes, LAYOUTS):
        for lib in AIX_LIBS:
            xs, ys = [], []
            for scn in SCENS:
                v = metric_for(scn, lib, layout, metric)
                if v is not None and not isinstance(v, str):
                    xs.append(RATIO[scn])
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, marker=AIX_MARKERS[lib], color=AIX_COLORS[lib],
                        label=SHORT[lib], linewidth=1.6)
        for lib in SS_LIBS:
            xs, ys = [], []
            for scn in SCENS:
                v = metric_for(scn, lib, "default", metric)
                if v is not None and not isinstance(v, str):
                    xs.append(RATIO[scn])
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, marker=SS_MARKERS[lib], color=SS_COLORS[lib],
                        linestyle="--", linewidth=1.4, label=SHORT[lib])
        ax.set_xscale("log", base=2)
        ax.set_xticks([RATIO[s] for s in SCENS])
        ax.set_xticklabels([str(RATIO[s]) for s in SCENS])
        ax.set_xlabel("CPU:GPU rank ratio (mm ranks / 96)")
        ax.set_title(f"layout = {layout}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    if PGF:
        save_figure(fig, Path(out_path))
    else:
        fig.savefig(out_path, dpi=150)
    close_figure(fig)
    print(f"wrote {out_path}")


def main():
    out_dir = Path("results") / "hethires3"
    render("t1", "HH3 scaling: avg warm ML-step time vs CPU:GPU ratio (s8-s11)",
           "avg warm ML-step time (ms)", out_dir / "scaling_step_time.png")
    render("t2", "HH3 scaling: inference device time vs CPU:GPU ratio (s8-s11)",
           "inference device time per step (ms)", out_dir / "scaling_inference_device.png")


if __name__ == "__main__":
    main()
