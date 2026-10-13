#!/usr/bin/env python3
"""
Ingest PhyDLL 4K step timings into results/hethires3/ml_step_makespan.csv
and regenerate the scaling plot.
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np

MINI_APP_DIR = Path("/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app")
LOGS_DIR = MINI_APP_DIR / "logs"
CSV_PATH = MINI_APP_DIR / "results/hethires3/ml_step_makespan.csv"

def parse_solver_log(log_file: Path) -> dict:
    if not log_file.is_file():
        return {}
    txt = log_file.read_text(errors="ignore")
    pattern = re.compile(r"^Step (\d+), ML, local moved: [^,]+, time: ([0-9.eE+-]+) ms", re.MULTILINE)
    steps = {}
    for m in pattern.finditer(txt):
        steps[int(m.group(1))] = float(m.group(2))
    return steps

def ingest_all():
    if not CSV_PATH.is_file():
        print(f"Error: {CSV_PATH} not found!")
        return

    df = pd.read_csv(CSV_PATH)
    scenarios = [7, 8, 9, 10, 11]
    variants = ["default", "readiness"]

    updated_cols = []
    for sid in scenarios:
        for var in variants:
            col_name = f"s{sid}_phydll_{var}"
            target_tag = f"hethires3_s{sid}_phydll_{var}"
            
            candidates = [
                LOGS_DIR / f"mini_app_output_{target_tag}.txt",
            ]
            for p in LOGS_DIR.glob(f"*{target_tag}*.txt"):
                if p not in candidates:
                    candidates.append(p)

            steps = {}
            found_log = None
            for p in candidates:
                st = parse_solver_log(p)
                if st:
                    steps = st
                    found_log = p
                    break

            if steps:
                print(f"[INGEST] Found {len(steps)} steps for {col_name} in {found_log.name}")
                # Map into df['solver_step']
                if col_name not in df.columns:
                    df[col_name] = np.nan
                for step_num, val in steps.items():
                    df.loc[df["solver_step"] == step_num, col_name] = val
                updated_cols.append(col_name)
            else:
                print(f"[WAITING] No log found yet for {col_name}")

    if updated_cols:
        df.to_csv(CSV_PATH, index=False)
        print(f"\nSuccessfully wrote updated columns to {CSV_PATH}:")
        for c in updated_cols:
            series = df[df["solver_step"] > 2][c].dropna()
            med = float(np.median(series)) if len(series) else None
            print(f"  {c:25s} -> {len(series)} steady samples, median = {med:.2f} ms" if med else f"  {c:25s} -> no steady samples")

if __name__ == "__main__":
    ingest_all()
