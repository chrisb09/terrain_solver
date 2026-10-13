#!/usr/bin/env python3
"""
plot_5node_traffic.py
=====================
Generates thesis-style figures for the 5-node distributed SmartSim traffic
suite (1 solver node + 4 dedicated DB nodes on c23ms, account rwth2150).

Compares Clustered (single shared Redis keyspace, 4 shards) against Standalone
(4 independent per-ML-node databases) for the 4K WaterCNN workload, 102 steps
(51 ML steps), 96 solver ranks, 5 replicates.

Figures written (PNG for inspection; .pgf/.pdf also emitted by thesis_style):
  traffic_decomposition   per-ML-step byte budget, measured stack + total
  traffic_direction       rx/tx per step, measured vs predicted
  step_time_series        steady ML-step latency across the run (mean + range)
  step_distribution       steady ML-step latency distribution
  scorep_regions          Score-P regional breakdown per step
  prediction_vs_measured  headline traffic-ratio validation

Usage:
  python3 plot_5node_traffic.py --manifest 5node_traffic_manifest.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --- thesis style -----------------------------------------------------------
_thesis_src = Path.home() / "Master-Thesis" / "gitlab" / "sourcecode"
if _thesis_src.exists():
    sys.path.insert(0, str(_thesis_src))
import thesis_style as ts  # noqa: E402
ts.apply()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reuse the analytical model + parsers already implemented.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_traffic_comparison import (  # noqa: E402
    compute_analytical_predictions,
    parse_scorep_cubex,
)

# --- configuration ----------------------------------------------------------
RANKS = 96
SHARDS = 4
WIDTH, HEIGHT = 3840, 2160
STEPS = 102
ML_STEPS = STEPS // 2  # 51

C_STANDALONE = "#1f77b4"  # tab:blue
C_CLUSTERED = "#d62728"   # tab:red
C_BASE = "#bdbdbd"
C_REDUNDANT = "#d62728"   # redundant input-0 reflection
C_IN1 = "#ff7f0e"         # input-1 cross-shard copy
C_OUT = "#ffbb78"         # output copy

MINI_APP = HERE


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_STEP_RE = re.compile(
    r"STEP_TIMING\s+step=(\d+)\s+solver=ML\s+step_ms=([0-9.]+)"
    r"(?:\s+local_moved=\S+\s+rank_min_ms=([0-9.]+)\s+rank_max_ms=([0-9.]+))?"
)
_NET_RE = re.compile(r"NET_USAGE\s+if=(\w+)\s+rx_mb=([0-9.]+)\s+tx_mb=([0-9.]+)")


def parse_step_timings(log_path: Path):
    """Return {step: (step_ms, rank_min_ms, rank_max_ms)} for ML steps."""
    out = {}
    if not log_path.exists():
        return out
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for m in _STEP_RE.finditer(text):
        step = int(m.group(1))
        ms = float(m.group(2))
        rmin = float(m.group(3)) if m.group(3) else np.nan
        rmax = float(m.group(4)) if m.group(4) else np.nan
        out[step] = (ms, rmin, rmax)
    return out


def parse_net_usage(log_path: Path, iface: str = "ib0"):
    """Return {'rx_mb':..,'tx_mb':..} from the cumulative NET_USAGE line."""
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    best = None
    for m in _NET_RE.finditer(text):
        if m.group(1) == iface:
            best = {"rx_mb": float(m.group(2)), "tx_mb": float(m.group(3))}
    return best


def parse_manifest(path: Path):
    entries = []
    if not path.exists():
        return entries
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            entries.append({
                "rep": int(parts[0]),
                "job_id": parts[1],
                "order": parts[2] if len(parts) > 2 else "unknown",
            })
    return entries


def load_condition(job_id: str, cond: str):
    log = MINI_APP / f"logs/solver_{cond}_{job_id}.log"
    root = MINI_APP / f"results/5node_traffic_{job_id}" / cond
    return {
        "step": parse_step_timings(log),
        "net": parse_net_usage(log, "ib0"),
        "cubex": parse_scorep_cubex(root / "scorep" / "profile.cubex"),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def gather(manifest_path: Path):
    pred = compute_analytical_predictions(RANKS, SHARDS, WIDTH, HEIGHT, STEPS)
    reps = []
    for e in parse_manifest(manifest_path):
        cl = load_condition(e["job_id"], "clustered")
        st = load_condition(e["job_id"], "standalone")
        if cl["net"] and st["net"]:
            reps.append({"rep": e["rep"], "job_id": e["job_id"], "order": e["order"],
                         "clustered": cl, "standalone": st})
    return pred, reps


def predicted_directions(pred):
    """Predicted tx/rx per ML step (MiB) for both conditions."""
    in0, in1, out0 = pred["in0_mb_step"], pred["in1_mb_step"], pred["out0_mb_step"]
    return {
        "standalone": {"tx": in0 + in1, "rx": out0},
        "clustered": {"tx": 2 * (in0 + in1) + out0, "rx": (in0 + in1) + 2 * out0},
    }


def steady_samples(cond_data):
    """Pooled steady (step>2) step_ms across all reps."""
    vals = []
    for step, (ms, _rmin, _rmax) in cond_data["step"].items():
        if step > 2:
            vals.append(ms)
    return np.asarray(vals)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_traffic_decomposition(pred, reps, outdir):
    dirs = predicted_directions(pred)
    in0, in1, out0 = pred["in0_mb_step"], pred["in1_mb_step"], pred["out0_mb_step"]

    # Measured total per step (mean over reps)
    meas = {
        "standalone": np.mean([r["standalone"]["net"]["rx_mb"] + r["standalone"]["net"]["tx_mb"]
                               for r in reps]) / ML_STEPS,
        "clustered": np.mean([r["clustered"]["net"]["rx_mb"] + r["clustered"]["net"]["tx_mb"]
                              for r in reps]) / ML_STEPS,
    }

    fig, ax = ts.figure(0.62, aspect=0.95, constrained_layout=True)
    x = [0.0, 1.0]
    labels = ["Standalone\n(4 independent DBs)", "Clustered\n(shared keyspace)"]

    st_segments = [in0, in1, out0]
    cl_segments = [in0, in1, out0, 2 * in0, 2 * in1, 2 * out0]
    st_colors = [C_BASE, C_BASE, C_BASE]
    cl_colors = [C_BASE, C_BASE, C_BASE, C_REDUNDANT, C_IN1, C_OUT]

    def stack(xpos, segs, colors):
        bottom = 0.0
        for v, c in zip(segs, colors):
            ax.bar(xpos, v, bottom=bottom, width=0.5, color=c, edgecolor="black", linewidth=0.5)
            bottom += v

    stack(0.0, st_segments, st_colors)
    stack(1.0, cl_segments, cl_colors)

    ax.scatter([0.0], [meas["standalone"]], marker="D", s=22, color="black", zorder=6,
               label="Measured total")
    ax.scatter([1.0], [meas["clustered"]], marker="D", s=22, color="black", zorder=6)

    ax.text(0.0, max(sum(st_segments), meas["standalone"]) + 60,
            f"pred {sum(st_segments):.0f}\nmeas {meas['standalone']:.0f}",
            ha="center", va="bottom", fontsize=8.0)
    ax.text(1.0, max(sum(cl_segments), meas["clustered"]) + 60,
            f"pred {sum(cl_segments):.0f}\nmeas {meas['clustered']:.0f}",
            ha="center", va="bottom", fontsize=8.0)

    # Legend proxies
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_BASE, edgecolor="black", label="Base payload (water+terrain+output)"),
        Patch(facecolor=C_REDUNDANT, edgecolor="black", label=f"Redundant input-0 reflection (2x{in0:.0f})"),
        Patch(facecolor=C_IN1, edgecolor="black", label=f"Input-1 cross-shard copy (2x{in1:.0f})"),
        Patch(facecolor=C_OUT, edgecolor="black", label=f"Output copy (2x{out0:.1f})"),
        plt.Line2D([], [], marker="D", color="black", linestyle="None", label="Measured total"),
    ]
    ax.legend(handles=handles, fontsize=6.6, loc="upper left", ncol=1, framealpha=0.9,
              borderpad=0.3, handlelength=1.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Network traffic per ML step (MiB)")
    ax.set_ylim(0, 2450)
    ax.set_title("run_model traffic budget per ML step")
    ts.save(fig, outdir / "traffic_decomposition")
    print(f"[fig] traffic_decomposition: pred {sum(st_segments):.0f} / {sum(cl_segments):.0f}"
          f" MiB; measured {meas['standalone']:.0f} / {meas['clustered']:.0f} MiB per step")


def fig_traffic_direction(pred, reps, outdir):
    pd = predicted_directions(pred)
    cats = ["Standalone", "Clustered"]
    measured = {"rx": [], "tx": []}
    for cond in ("standalone", "clustered"):
        rx = np.mean([r[cond]["net"]["rx_mb"] for r in reps]) / ML_STEPS
        tx = np.mean([r[cond]["net"]["tx_mb"] for r in reps]) / ML_STEPS
        rx_s = np.std([r[cond]["net"]["rx_mb"] / ML_STEPS for r in reps])
        tx_s = np.std([r[cond]["net"]["tx_mb"] / ML_STEPS for r in reps])
        measured["rx"].append((rx, rx_s))
        measured["tx"].append((tx, tx_s))

    fig, ax = ts.figure(0.62, aspect=0.8, constrained_layout=True)
    x = np.arange(len(cats))
    w = 0.34
    offs = {"rx": -w * 0.55, "tx": w * 0.55}

    from matplotlib.patches import Patch
    for direction, color in (("rx", "#2ca02c"), ("tx", "#9467bd")):
        vals = [measured[direction][i][0] for i in range(2)]
        errs = [measured[direction][i][1] for i in range(2)]
        pvals = [pd[c]["rx" if direction == "rx" else "tx"] for c in ("standalone", "clustered")]
        ax.bar(x + offs[direction], vals, width=w, yerr=errs, capsize=3,
               color=color, edgecolor="black", linewidth=0.5,
               error_kw={"elinewidth": 0.9, "ecolor": "black"})
        # predicted: hollow black outline on top of the measured bar
        ax.bar(x + offs[direction], pvals, width=w, color="none", edgecolor="black",
               linewidth=1.0, linestyle="--", zorder=5)
        for xi, v in zip(x + offs[direction], vals):
            ax.text(xi, v + 18, f"{v:.0f}", ha="center", va="bottom", fontsize=7.0)

    handles = [
        Patch(facecolor="#2ca02c", edgecolor="black", label="Measured rx"),
        Patch(facecolor="#9467bd", edgecolor="black", label="Measured tx"),
        Patch(facecolor="none", edgecolor="black", linestyle="--", linewidth=1.0, label="Predicted (outline)"),
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("InfiniBand ib0 traffic per ML step (MiB)")
    ax.set_ylim(0, 1500)
    ax.set_title("Direction-resolved traffic (solver NIC)")
    ax.legend(handles=handles, fontsize=7.2, loc="upper left", framealpha=0.9)
    ts.save(fig, outdir / "traffic_direction")
    print(f"[fig] traffic_direction: standalone rx/tx={measured['rx'][0][0]:.0f}/{measured['tx'][0][0]:.0f},"
          f" clustered rx/tx={measured['rx'][1][0]:.0f}/{measured['tx'][1][0]:.0f} MiB/step")


def fig_step_time_series(pred, reps, outdir):
    fig, ax = ts.figure(0.72, aspect=0.6, constrained_layout=True)
    for cond, color, label in (("standalone", C_STANDALONE, "Standalone"),
                               ("clustered", C_CLUSTERED, "Clustered")):
        steps = sorted({s for r in reps for s in r[cond]["step"] if s > 2})
        mean = []
        lo = []
        hi = []
        for s in steps:
            vals = [r[cond]["step"][s][0] for r in reps if s in r[cond]["step"]]
            mean.append(np.mean(vals))
            lo.append(np.min(vals))
            hi.append(np.max(vals))
        ax.plot(steps, mean, color=color, label=label, linewidth=1.6)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.set_xlim(2, 102)
    ax.set_xlabel("Simulation step (ML steps only)")
    ax.set_ylabel("ML step wall time (ms)")
    ax.set_title("Steady ML-step latency (mean $\\pm$ range, 5 reps)")
    ax.legend(fontsize=9)
    ts.save(fig, outdir / "step_time_series")
    print("[fig] step_time_series")


def fig_step_distribution(pred, reps, outdir):
    st = np.concatenate([steady_samples(r["standalone"]) for r in reps])
    cl = np.concatenate([steady_samples(r["clustered"]) for r in reps])
    fig, ax = ts.figure(0.5, aspect=0.85, constrained_layout=True)
    bp = ax.boxplot([st, cl], widths=0.45, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.0})
    for patch, color in zip(bp["boxes"], (C_STANDALONE, C_CLUSTERED)):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
    # per-rep medians
    for i, cond in enumerate(("standalone", "clustered"), start=1):
        meds = []
        for r in reps:
            v = steady_samples(r[cond])
            if v.size:
                meds.append(np.median(v))
        ax.scatter(np.full(len(meds), i), meds, color="black", s=14, zorder=5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Standalone", "Clustered"], fontsize=9)
    ax.set_ylabel("Steady ML step wall time (ms)")
    ax.set_title("Steady-step latency distribution")
    ts.save(fig, outdir / "step_distribution")
    print(f"[fig] step_distribution: standalone median={np.median(st):.1f} ms,"
          f" clustered median={np.median(cl):.1f} ms")


def fig_scorep_regions(pred, reps, outdir):
    regions = ["smartsim_run_model", "smartsim_put_tensor", "smartsim_unpack_tensor"]
    labels = ["run_model", "put_tensor", "unpack_tensor"]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]

    data = {}
    for cond in ("standalone", "clustered"):
        per = {r: [] for r in regions}
        step_ms = []
        for rep in reps:
            cb = rep[cond]["cubex"]
            if not cb or "error" in cb:
                continue
            for r in regions:
                if r in cb:
                    per[r].append(cb[r]["mean"] / ML_STEPS * 1000.0)
            v = steady_samples(rep[cond])
            if v.size:
                step_ms.append(np.mean(v))
        means = {r: np.mean(per[r]) if per[r] else 0.0 for r in regions}
        total_step = np.mean(step_ms) if step_ms else 0.0
        residual = max(0.0, total_step - sum(means.values()))
        data[cond] = (means, residual, total_step)

    fig, ax = ts.figure(0.6, aspect=0.85, constrained_layout=True)
    x = np.arange(2)
    bottoms = np.zeros(2)
    for r, lab, c in zip(regions, labels, colors):
        vals = np.array([data["standalone"][0][r], data["clustered"][0][r]])
        ax.bar(x, vals, bottom=bottoms, width=0.5, color=c, edgecolor="black",
               linewidth=0.5, label=lab)
        bottoms += vals
    res = np.array([data["standalone"][1], data["clustered"][1]])
    ax.bar(x, res, bottom=bottoms, width=0.5, color="#cccccc", edgecolor="black",
           linewidth=0.5, label="other / overhead")
    for i, cond in enumerate(("standalone", "clustered")):
        ax.text(i, data[cond][2] + 6, f"{data[cond][2]:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Standalone", "Clustered"], fontsize=9)
    ax.set_ylabel("Time per ML step (ms)")
    ax.set_ylim(0, 490)
    ax.set_title("Score-P regional breakdown")
    ax.legend(fontsize=6.9, ncol=2, loc="upper center", framealpha=0.9, columnspacing=1.0)
    ts.save(fig, outdir / "scorep_regions")
    print(f"[fig] scorep_regions: run_model {data['standalone'][0]['smartsim_run_model']:.1f} ->"
          f" {data['clustered'][0]['smartsim_run_model']:.1f} ms")


def fig_prediction_vs_measured(pred, reps, outdir):
    pd = predicted_directions(pred)
    in0, in1, out0 = pred["in0_mb_step"], pred["in1_mb_step"], pred["out0_mb_step"]
    pred_ratio = pred["clustered_step_full_mb"] / pred["base_step_full_mb"]

    ratios = []
    for r in reps:
        cl = (r["clustered"]["net"]["rx_mb"] + r["clustered"]["net"]["tx_mb"])
        st = (r["standalone"]["net"]["rx_mb"] + r["standalone"]["net"]["tx_mb"])
        ratios.append(cl / st)
    ratios = np.array(ratios)

    fig, axes = plt.subplots(1, 2, figsize=ts.figure_size(1.0, aspect=0.42), constrained_layout=True)

    # (a) ratio
    ax = axes[0]
    ax.axhline(pred_ratio, color="black", linestyle="--", linewidth=1.0, label=f"Predicted {pred_ratio:.2f}x")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.9)
    ax.scatter(np.ones(len(ratios)) * 0.0, ratios, color=C_CLUSTERED, s=22, zorder=5, label="Measured reps")
    ax.scatter([0.0], [np.mean(ratios)], marker="D", color="black", s=30, zorder=6, label="Measured mean")
    ax.annotate(f"{np.mean(ratios):.3f}$\\times$ ($\\sigma$={np.std(ratios):.3f})",
                (0.0, np.mean(ratios)), xytext=(10, 6), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.0)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.9, 3.3)
    ax.set_xticks([0.0])
    ax.set_xticklabels(["Clustered / Standalone"], fontsize=9)
    ax.set_ylabel("Traffic ratio per ML step")
    ax.set_title("Traffic ratio")
    ax.legend(fontsize=8, loc="lower right")

    # (b) predicted vs measured per direction
    ax2 = axes[1]
    cats = ["Std rx", "Std tx", "Clu rx", "Clu tx"]
    meas = [
        np.mean([r["standalone"]["net"]["rx_mb"] for r in reps]) / ML_STEPS,
        np.mean([r["standalone"]["net"]["tx_mb"] for r in reps]) / ML_STEPS,
        np.mean([r["clustered"]["net"]["rx_mb"] for r in reps]) / ML_STEPS,
        np.mean([r["clustered"]["net"]["tx_mb"] for r in reps]) / ML_STEPS,
    ]
    predv = [pd["standalone"]["rx"], pd["standalone"]["tx"],
             pd["clustered"]["rx"], pd["clustered"]["tx"]]
    x = np.arange(len(cats))
    ax2.bar(x - 0.19, predv, width=0.36, color="#cccccc", edgecolor="black",
            linewidth=0.5, label="Predicted")
    ax2.bar(x + 0.19, meas, width=0.36, color=[C_STANDALONE, C_STANDALONE, C_CLUSTERED, C_CLUSTERED],
            edgecolor="black", linewidth=0.5, label="Measured")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, fontsize=8.5)
    ax2.set_ylabel("MiB per ML step")
    ax2.set_title("Per-step bytes")
    ax2.legend(fontsize=8)

    ts.save(fig, outdir / "prediction_vs_measured")
    print(f"[fig] prediction_vs_measured: ratio pred {pred_ratio:.3f}, meas {np.mean(ratios):.3f}"
          f" (sigma {np.std(ratios):.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=MINI_APP / "5node_traffic_manifest.txt")
    ap.add_argument("--outdir", type=Path,
                    default=MINI_APP / "campaign_figures" / "5node_traffic_suite")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    pred, reps = gather(args.manifest)
    if not reps:
        print("ERROR: no complete replicates found.")
        sys.exit(1)
    print(f"Loaded {len(reps)} replicates from {args.manifest} -> {args.outdir}")

    fig_traffic_decomposition(pred, reps, args.outdir)
    fig_traffic_direction(pred, reps, args.outdir)
    fig_step_time_series(pred, reps, args.outdir)
    fig_step_distribution(pred, reps, args.outdir)
    fig_scorep_regions(pred, reps, args.outdir)
    fig_prediction_vs_measured(pred, reps, args.outdir)


if __name__ == "__main__":
    main()
