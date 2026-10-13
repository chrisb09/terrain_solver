#!/usr/bin/env python3
"""Regenerate all AIx layout matrix and variance figures using the thesis_style pipeline
(Latin Modern fonts, Type 1 embedding, PGF + PDF + PNG).
"""

from __future__ import annotations

import glob
import subprocess
import sys
from pathlib import Path

ROOT = Path("/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app")
MATRIX_DIR = ROOT / "results" / "aix_layout_matrix"


def run_cmd(cmd: list[str]):
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] {' '.join(cmd)}\n{res.stderr}")
    else:
        print(f"[OK] {cmd[1]} -> {Path(cmd[-1]).name}")


def main():
    cfg_dirs = sorted([d for d in MATRIX_DIR.iterdir() if d.is_dir() and not d.name.startswith("numa_") and d.name != "scaling_superres"])

    print(f"=== Found {len(cfg_dirs)} configuration directories to regenerate ===")
    for d in cfg_dirs:
        cfg = d.name
        manifests = sorted(glob.glob(str(d / "runs_*.csv")))
        if not manifests:
            continue

        # 1. Comparison figure
        comp_out = d / f"aix_layout_matrix_{cfg}_comparison.png"
        cmd_comp = [
            sys.executable, "plot_layout_matrix.py",
            "--config-label", cfg,
            "--logs-dir", "logs",
            "--runs-csv", *manifests,
            "--out", str(comp_out),
        ]
        run_cmd(cmd_comp)

        # 2. Variance figure
        var_csv = d / "aix_5r_per_run_medians.csv"
        if var_csv.exists():
            var_out = d / "aix_5r_per_run_variance.png"
            cmd_var = [
                sys.executable, "plot_per_run_variance.py",
                "--config-label", cfg,
                "--per-run-csv", str(var_csv),
                "--out", str(var_out),
            ]
            run_cmd(cmd_var)

    # 3. Scaling figures
    points_csv = MATRIX_DIR / "superres_points.csv"
    if points_csv.exists():
        cmd_scale = [
            sys.executable, "plot_scaling.py",
            "--points-csv", str(points_csv),
            "--config-label", "super-high-res (7680x4320)",
            "--single-label", "single c23g node",
            "--het-label", "het surrogate (c23g + c23g, CPU GPUs hidden)",
            "--out-dir", str(MATRIX_DIR / "scaling_superres"),
        ]
        run_cmd(cmd_scale)

    # 4. A/B/C/D Delta figure
    a = MATRIX_DIR / "24c1g/aix_layout_matrix_24c1g_summary.csv"
    b = MATRIX_DIR / "96c4g/aix_layout_matrix_96c4g_c23g_summary.csv"
    c = MATRIX_DIR / "96c4g_superres/aix_layout_matrix_96c4g_superres_summary.csv"
    d = MATRIX_DIR / "24c1g_superres/aix_layout_matrix_24c1g_superres_summary.csv"
    if all(p.exists() for p in [a, b, c, d]):
        cmd_abcd = [
            sys.executable, "compare_superres.py",
            "--a", str(a), "--b", str(b), "--c", str(c), "--d", str(d),
            "--out", str(MATRIX_DIR / "aix_superres_abcd_delta.png"),
            "--out-csv", str(MATRIX_DIR / "aix_superres_abcd_comparison.csv"),
        ]
        run_cmd(cmd_abcd)

    print("\n=== Regeneration complete ===")


if __name__ == "__main__":
    main()
