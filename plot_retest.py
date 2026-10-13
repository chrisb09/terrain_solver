#!/usr/bin/env python3
"""Thesis-styled figures for the synchronized-timing retest (local staging).

Outputs .pgf/.pdf/.png into results/retest_plots/ (NOT the thesis repo).

Layout-matrix figures (per model, split by config group):
    <model>_layout_single : rows 24c1g, 96c4g
    <model>_layout_het    : rows 24p24c1g, 96p96c4g
  each 2x2: left column = median bar chart (y from 0, small value labels,
  headroom so labels stay inside the axes); right column = box plot of the
  per-step distribution (y auto-scaled, no labels).

Coupling-library figures (per model):
    <model>_coupling_libraries : 1x2, left median bar chart, right box plot, comparing
  AIx default/coll vs SmartSim shared vs SmartSim per-GPU DB.
"""
from __future__ import annotations

import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "Master-Thesis" / "gitlab" / "sourcecode"))
import thesis_style as ts  # noqa: E402

ts.apply()
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("results/retest_plots")
STEP_RE = re.compile(r"STEP_TIMING step=(\d+) solver=ML step_ms=([0-9.]+)")

LAYOUTS = ["default", "affine", "numa"]
COMMS = ["coll", "p2p1", "p2p24"]
COMM_LABEL = {"coll": "collective", "p2p1": "p2p (credit=1)", "p2p24": "p2p (full credits)"}
COLORS = {"coll": "#1f77b4", "p2p1": "#ff7f0e", "p2p24": "#2ca02c"}
PROV = [("aix", "AIx default (collective)", "#1f77b4"),
        ("shared", "SmartSim default", "#ff7f0e"),
        ("pergpu", "SmartSim per-GPU", "#2ca02c")]

CFG_LABEL = {"24c1g": "24c / 1 GPU", "24p24c1g": "24+24c / 1 GPU",
             "96c4g": "96c / 4 GPUs", "96p96c4g": "96+96c / 4 GPUs"}
MODEL_LABEL = {"watercnn8k": "[TS] WaterCNN, 7680 x 4320", "giantmlp288": "[TS] Giant MLP, 480 x 288"}

AIX_RID = re.compile(
    r"retest_(?P<cfg>\S+?)_(?P<model>watercnn8k|giantmlp288)_(?P<layout>\w+?)_(?P<comm>\w+?)_r(?P<rep>\d+)$")
SS_RID = re.compile(
    r"retest_smartsim_(?P<cfg>\S+?)_(?P<model>watercnn8k|giantmlp288)_(?P<db>shared|pergpu)_c0_r(?P<rep>\d+)$")


def _jobid(path: str) -> int:
    m = re.search(r"_(\d+)\.txt$", path)
    return int(m.group(1)) if m else -1


def _collect(marker: str) -> dict[str, list[float]]:
    """Return run_id -> ML step samples, taking the highest-jobid log per run_id."""
    best: dict[str, tuple[int, list[float]]] = {}
    for log in glob.glob("logs/retest_aix_*.txt"):
        jid = _jobid(log)
        text = Path(log).read_text(errors="replace")
        parts = re.split(rf"\[.*?\] {marker} run_id=(\S+)", text)
        for i in range(1, len(parts), 2):
            rid, body = parts[i], parts[i + 1]
            vals = [float(m.group(2)) for m in STEP_RE.finditer(body) if int(m.group(1)) > 2]
            if not vals:
                continue
            if rid not in best or jid > best[rid][0]:
                best[rid] = (jid, vals)
    return {rid: v[1] for rid, v in best.items()}


def load_aix() -> dict[tuple[str, str, str, str], list[float]]:
    out: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for rid, vals in _collect("cell").items():
        m = AIX_RID.match(rid)
        if m:
            g = m.groupdict()
            out[(g["model"], g["cfg"], g["layout"], g["comm"])].extend(vals)
    return out


def load_smartsim() -> dict[tuple[str, str, str], list[float]]:
    out: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for rid, vals in _collect("case").items():
        m = SS_RID.match(rid)
        if m:
            g = m.groupdict()
            out[(g["model"], g["cfg"], g["db"])].extend(vals)
    return out


def _annotate(ax, rects, vals, fs: float, above=False, rotation=None) -> None:
    for rect, v in zip(rects, vals):
        if np.isfinite(v):
            angle = rotation if rotation is not None else (0 if v < 500 else 90)
            outside = above
            if above is None:
                # Compare physical bar height with the rotated label plus padding.
                height_pt = (ax.get_position().height * ax.figure.get_figheight() * 72
                             * v / ax.get_ylim()[1])
                label_height_pt = fs * (0.65 * len(f"{v:.0f}") if angle == 90 else 1.0)
                outside = height_pt < label_height_pt * 1.3 + 6
            ax.annotate(f"{v:.0f}", (rect.get_x() + rect.get_width() / 2, v),
                        xytext=(0, 2 if outside else -3), textcoords="offset points",
                        ha="center", va="bottom" if outside else "top", fontsize=fs,
                        rotation=angle)


def _bar_axis(ax, data, xlabels, series, labels, colors, xtick_fs, annot_fs,
              rotation=None) -> None:
    x = np.arange(len(xlabels))
    w = 0.8 / len(series)
    for i, key in enumerate(series):
        vals = [data.get((lab, key), np.nan) for lab in xlabels]
        rects = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w,
                       label=labels[key], color=colors[key],
                       edgecolor="black", linewidth=0.4)
        _annotate(ax, rects, vals, annot_fs, rotation=rotation)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=xtick_fs, va="baseline")
    ax.tick_params(axis="x", pad=12)
    finite = [v for k in series for v in (data.get((lab, k), np.nan) for lab in xlabels) if np.isfinite(v)]
    if finite:
        ax.set_ylim(0, max(finite) * 1.16)
    ax.set_ylabel("ML step time [ms]", fontsize=9)
    ax.grid(axis="y")


def _box_axis(ax, samples, xlabels, series, colors, xtick_fs) -> None:
    x = np.arange(len(xlabels))
    w = 0.8 / len(series)
    for i, key in enumerate(series):
        for j, lab in enumerate(xlabels):
            vals = samples.get((lab, key), [])
            if not vals:
                continue
            pos = x[j] + (i - (len(series) - 1) / 2) * w
            bp = ax.boxplot([vals], positions=[pos], widths=w * 0.85, patch_artist=True,
                            showfliers=False,
                            medianprops=dict(color="black", linewidth=1.0),
                            whiskerprops=dict(color="black", linewidth=0.7),
                            capprops=dict(color="black", linewidth=0.7),
                            boxprops=dict(edgecolor="black", linewidth=0.5))
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[key])
                patch.set_alpha(0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=xtick_fs, va="baseline")
    ax.tick_params(axis="x", pad=12)
    ax.set_ylabel("ML step time [ms]", fontsize=9)
    ax.grid(axis="y")


def layout_figure(model: str, group: str, cfgs: list[str]) -> None:
    aix = load_aix()
    # Preserve room for text while reducing each panel's physical height by 30%.
    panel_height = ts.TEXTWIDTH_IN * 0.66 * 0.72 / 2.52 * 0.70
    height = 0.44 + 0.65 + 0.50 + 2 * panel_height
    fig, axes = plt.subplots(2, 2, figsize=(ts.TEXTWIDTH_IN, height))
    for row, cfg in enumerate(cfgs):
        bar_data = {(layout, comm): np.median(aix.get((model, cfg, layout, comm), [np.nan]))
                    for layout in LAYOUTS for comm in COMMS}
        bar_data = {k: v for k, v in bar_data.items() if np.isfinite(v)}
        _bar_axis(axes[row, 0], bar_data, LAYOUTS, COMMS, COMM_LABEL, COLORS,
                  ts.TICK, 6.0, rotation=0 if model == "watercnn8k" else None)
        _box_axis(axes[row, 1], {(layout, comm): aix.get((model, cfg, layout, comm), [])
                                 for layout in LAYOUTS for comm in COMMS},
                  LAYOUTS, COMMS, COLORS, ts.TICK)
        axes[row, 0].set_title(CFG_LABEL[cfg], fontsize=12)
        axes[row, 1].set_title(CFG_LABEL[cfg], fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, 0.015), fontsize=11)
    fig.suptitle(f"{MODEL_LABEL[model]}: layout matrix ({'single node' if group=='single' else 'heterogeneous'})",
                 y=0.985, fontsize=12)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.65 / height,
                        top=1 - 0.44 / height, wspace=0.35,
                        hspace=0.50 / panel_height)
    ts.save(fig, OUT / f"{model}_layout_{group}")
    print(f"wrote {OUT}/{model}_layout_{group}.{{pgf,pdf,png}}")


def coupling_library_figure(model: str) -> None:
    aix = load_aix()
    ss = load_smartsim()
    cfgs = ["24c1g", "24p24c1g", "96c4g", "96p96c4g"]
    # Halve panel height, not the numeric y-range or the font sizes.
    panel_height = ts.TEXTWIDTH_IN * 0.50 * 0.59 * 0.50
    height = 0.44 + 0.72 + panel_height
    fig, axes = plt.subplots(1, 2, figsize=(ts.TEXTWIDTH_IN, height))
    x = np.arange(len(cfgs))
    w = 0.26
    annotations = []
    for i, (key, label, color) in enumerate(PROV):
        med, boxes = [], []
        for cfg in cfgs:
            if key == "aix":
                v = aix.get((model, cfg, "default", "coll"), [])
            else:
                v = ss.get((model, cfg, key), [])
            med.append(np.median(v) if v else np.nan)
            boxes.append(v)
        rects = axes[0].bar(x + (i - 1) * w, med, w, label=label, color=color,
                            edgecolor="black", linewidth=0.4)
        annotations.append((rects, med))
        for j, v in enumerate(boxes):
            if not v:
                continue
            bp = axes[1].boxplot([v], positions=[x[j] + (i - 1) * w], widths=w * 0.85,
                                 patch_artist=True, showfliers=False,
                                 medianprops=dict(color="black", linewidth=1.0),
                                 whiskerprops=dict(color="black", linewidth=0.7),
                                 capprops=dict(color="black", linewidth=0.7),
                                 boxprops=dict(edgecolor="black", linewidth=0.5))
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
    finite = []
    for cfg in cfgs:
        for key in ("aix", "shared", "pergpu"):
            v = aix.get((model, cfg, "default", "coll"), []) if key == "aix" else ss.get((model, cfg, key), [])
            if v:
                finite.append(np.median(v))
    axes[0].set_ylim(0, max(finite) * 1.22)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([CFG_LABEL[c].replace(" / ", "\n") for c in cfgs],
                           fontsize=8, rotation=0, ha="center")
        ax.set_ylabel("ML step time [ms]", fontsize=9)
        ax.grid(axis="y")
    axes[0].set_title("median", fontsize=ts.AXIS_TITLE)
    axes[1].set_title("distribution", fontsize=ts.AXIS_TITLE)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                bbox_to_anchor=(0.5, 0.015), fontsize=8)
    fig.suptitle(f"{MODEL_LABEL[model]}: coupling library comparison", y=0.99,
                 fontsize=ts.FIGURE_TITLE)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.72 / height,
                        top=1 - 0.44 / height, wspace=0.35)
    for rects, med in annotations:
        _annotate(axes[0], rects, med, 4.3, above=None, rotation=90)
    ts.save(fig, OUT / f"{model}_coupling_libraries")
    print(f"wrote {OUT}/{model}_coupling_libraries.{{pgf,pdf,png}}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model in ("watercnn8k", "giantmlp288"):
        layout_figure(model, "single", ["24c1g", "96c4g"])
        layout_figure(model, "het", ["24p24c1g", "96p96c4g"])
        coupling_library_figure(model)


if __name__ == "__main__":
    main()
