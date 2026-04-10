#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import socket
import subprocess
import time
from pathlib import Path

PING_TIME_RE = re.compile(r"time=([0-9]+(?:\.[0-9]+)?)\s*ms")


def parse_args():
    parser = argparse.ArgumentParser(description="Per-node ping mesh sampler and aggregator")
    sub = parser.add_subparsers(dest="mode", required=True)

    worker = sub.add_parser("worker", help="Run on one node and collect per-second RTTs")
    worker.add_argument("--targets", required=True, help="Comma-separated host list")
    worker.add_argument("--samples", type=int, required=True, help="Number of 1-second samples")
    worker.add_argument("--start-epoch", type=float, required=True, help="Unix epoch for synchronized start")
    worker.add_argument("--timeout-sec", type=int, default=1, help="Ping timeout in seconds")
    worker.add_argument("--out-dir", required=True, help="Directory for per-node raw JSON")

    agg = sub.add_parser("aggregate", help="Aggregate per-node raw JSON to one CSV")
    agg.add_argument("--targets", required=True, help="Comma-separated host list (desired ordering)")
    agg.add_argument("--samples", type=int, required=True, help="Number of expected samples")
    agg.add_argument("--out-dir", required=True, help="Directory containing per-node raw JSON")
    agg.add_argument("--csv", required=True, help="Output CSV path")

    return parser.parse_args()


def ping_once(dst: str, timeout_sec: int):
    cmd = ["ping", "-n", "-c", "1", "-W", str(timeout_sec), dst]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    output = f"{result.stdout}\n{result.stderr}"
    m = PING_TIME_RE.search(output)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def run_worker(args):
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = socket.gethostname().split(".")[0]
    out_file = out_dir / f"raw_{src}.json"

    rows = []
    for sample in range(args.samples):
        target_epoch = args.start_epoch + sample
        while True:
            now = time.time()
            if now >= target_epoch:
                break
            time.sleep(min(0.05, target_epoch - now))

        row_time = time.strftime("%H:%M:%S", time.localtime(target_epoch))
        rtt_map = {}
        for dst in targets:
            if dst == src:
                rtt_map[dst] = 0.0
            else:
                rtt_map[dst] = ping_once(dst, args.timeout_sec)

        rows.append({
            "sample": sample,
            "time": row_time,
            "src": src,
            "rtt_ms": rtt_map,
        })

    payload = {
        "src": src,
        "targets": targets,
        "samples": args.samples,
        "rows": rows,
    }
    out_file.write_text(json.dumps(payload), encoding="utf-8")


def run_aggregate(args):
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    out_dir = Path(args.out_dir)
    samples = args.samples

    data_by_src = {}
    for src in targets:
        path = out_dir / f"raw_{src}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("rows", [])
        sample_map = {int(r.get("sample", -1)): r for r in rows if isinstance(r.get("sample"), int)}
        data_by_src[src] = sample_map

    header = ["time"]
    for src in targets:
        for dst in targets:
            header.append(f"{src}->{dst}")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for sample in range(samples):
            row_time = ""
            # Prefer time from the first available source for this sample.
            for src in targets:
                src_rows = data_by_src.get(src, {})
                if sample in src_rows:
                    row_time = src_rows[sample].get("time", "")
                    if row_time:
                        break

            row = [row_time]
            for src in targets:
                src_rows = data_by_src.get(src, {})
                item = src_rows.get(sample)
                rtts = item.get("rtt_ms", {}) if item else {}
                for dst in targets:
                    value = rtts.get(dst)
                    if value is None or (isinstance(value, float) and math.isnan(value)):
                        row.append("")
                    else:
                        row.append(f"{float(value):.3f}")
            writer.writerow(row)


def main():
    args = parse_args()
    if args.mode == "worker":
        run_worker(args)
    else:
        run_aggregate(args)


if __name__ == "__main__":
    main()
