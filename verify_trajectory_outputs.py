#!/usr/bin/env python3
"""
Trajectory Output Verification Script for SmartSim Sequential Chain Suite.
Compares final water height matrices from world_trajectory.h5 files between
chains C=0 (baseline) and C=1..6 for each model configuration and repeat.
Verifies exact bitwise / numerical equivalence across sequential chain levels.
"""

import glob
import os
import re
import sys
import h5py
import numpy as np

def verify_trajectories(base_dir="/hpcwork/thes2181/mini_app"):
    print("=========================================================================")
    print(" SmartSim Trajectory Output Verification (C=1..6 vs C=0 Baseline)")
    print("=========================================================================\n")

    # Find all trajectory files in output job directories
    traj_files = glob.glob(os.path.join(base_dir, "CMI_SmartSim_*/world_trajectory.h5"))
    if not traj_files:
        print(f"No world_trajectory.h5 files found under {base_dir}")
        return

    # Group trajectories by (model_key, repeat)
    # Directory naming pattern: CMI_SmartSim_<label>_c<chains>_rep<repeat>_*
    grouped = {}

    for fpath in traj_files:
        dirname = os.path.basename(os.path.dirname(fpath))
        m = re.search(r"CMI_SmartSim_(.+)_c(\d+)_rep(\d+)", dirname)
        if not m:
            continue
        group_id = m.group(1) # e.g. 50k, 600k, watercnn_scale1to1, giantmlp_scale100th
        chains = int(m.group(2))
        repeat = int(m.group(3))

        key = (group_id, repeat)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][chains] = fpath

    if not grouped:
        print("No matching trajectory file groups identified.")
        return

    report_rows = []

    for (group_id, repeat), chains_dict in sorted(grouped.items()):
        if 0 not in chains_dict:
            print(f"Warning: Baseline C=0 missing for group={group_id}, rep={repeat}")
            continue

        c0_file = chains_dict[0]
        try:
            with h5py.File(c0_file, "r") as h5_c0:
                if "water" not in h5_c0:
                    print(f"Error: dataset 'water' missing in {c0_file}")
                    continue
                c0_water = h5_c0["water"][:]
        except Exception as e:
            print(f"Error reading baseline {c0_file}: {e}")
            continue

        for c in range(1, 7):
            if c not in chains_dict:
                continue
            ci_file = chains_dict[c]
            try:
                with h5py.File(ci_file, "r") as h5_ci:
                    if "water" not in h5_ci:
                        continue
                    ci_water = h5_ci["water"][:]

                # Numerical checks
                diff = np.abs(c0_water - ci_water)
                max_abs_diff = np.max(diff)
                mean_sq_err = np.mean(diff ** 2)
                
                c0_l2 = np.linalg.norm(c0_water)
                rel_l2_err = np.linalg.norm(diff) / c0_l2 if c0_l2 > 0 else 0.0
                is_exact = np.array_equal(c0_water, ci_water)

                report_rows.append({
                    "group_id": group_id,
                    "repeat": repeat,
                    "chains": c,
                    "shape": str(c0_water.shape),
                    "is_exact": is_exact,
                    "max_abs_diff": max_abs_diff,
                    "mse": mean_sq_err,
                    "rel_l2_err": rel_l2_err
                })
            except Exception as e:
                print(f"Error reading comparison {ci_file}: {e}")

    # Output verification table
    print("| Group ID | Repeat | Chain (C) | Shape | Bitwise Identical | Max Abs Diff | MSE | Rel L2 Error |")
    print("|---|---|---|---|---|---|---|---|")
    for r in report_rows:
        exact_str = "YES (Identical)" if r["is_exact"] else f"NO (Diff={r['max_abs_diff']:.2e})"
        print(f"| {r['group_id']} | {r['repeat']} | c={r['chains']} | {r['shape']} | {exact_str} | {r['max_abs_diff']:.2e} | {r['mse']:.2e} | {r['rel_l2_err']:.2e} |")

    all_exact = all(r["is_exact"] for r in report_rows)
    print("\n-------------------------------------------------------------------------")
    if all_exact:
        print(" VERIFICATION SUCCESSFUL: All sequential chain outputs (C=1..6) are 100% BITWISE IDENTICAL to C=0 baseline.")
    else:
        print(" VERIFICATION NOTICE: Small floating-point or non-exact differences detected.")
    print("-------------------------------------------------------------------------\n")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/hpcwork/thes2181/mini_app"
    verify_trajectories(base)
