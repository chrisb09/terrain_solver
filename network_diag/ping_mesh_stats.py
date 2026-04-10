#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-link ping mesh statistics")
    parser.add_argument("--csv", required=True, help="Input mesh CSV from ping_mesh.py aggregate")
    parser.add_argument("--stats-csv", required=True, help="Output per-link stats CSV")
    parser.add_argument("--p95-matrix-csv", required=True, help="Output NxN p95 matrix CSV")
    parser.add_argument("--loss-matrix-csv", required=True, help="Output NxN loss matrix CSV")
    return parser.parse_args()


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def parse_link(link):
    if "->" not in link:
        return link, ""
    src, dst = link.split("->", 1)
    return src, dst


def main():
    args = parse_args()
    in_csv = Path(args.csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise RuntimeError("Input CSV has no link columns")
        links = header[1:]

        values = {link: [] for link in links}
        samples = 0
        for row in reader:
            if not row:
                continue
            samples += 1
            for idx, link in enumerate(links, start=1):
                if idx >= len(row):
                    continue
                cell = row[idx].strip()
                if not cell:
                    continue
                try:
                    values[link].append(float(cell))
                except ValueError:
                    continue

    stats_rows = []
    sources = []
    destinations = []
    for link in links:
        src, dst = parse_link(link)
        if src not in sources:
            sources.append(src)
        if dst and dst not in destinations:
            destinations.append(dst)

        vals = sorted(values.get(link, []))
        received = len(vals)
        loss_pct = 100.0 * (samples - received) / samples if samples > 0 else 0.0
        min_v = vals[0] if vals else None
        max_v = vals[-1] if vals else None
        avg_v = (sum(vals) / received) if vals else None
        p50_v = percentile(vals, 0.50)
        p95_v = percentile(vals, 0.95)

        stats_rows.append({
            "link": link,
            "src": src,
            "dst": dst,
            "samples": samples,
            "received": received,
            "loss_pct": loss_pct,
            "min_ms": min_v,
            "avg_ms": avg_v,
            "p50_ms": p50_v,
            "p95_ms": p95_v,
            "max_ms": max_v,
        })

    stats_path = Path(args.stats_csv)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["link", "src", "dst", "samples", "received", "loss_pct", "min_ms", "avg_ms", "p50_ms", "p95_ms", "max_ms"])
        for r in stats_rows:
            writer.writerow([
                r["link"],
                r["src"],
                r["dst"],
                r["samples"],
                r["received"],
                f"{r['loss_pct']:.3f}",
                "" if r["min_ms"] is None else f"{r['min_ms']:.3f}",
                "" if r["avg_ms"] is None else f"{r['avg_ms']:.3f}",
                "" if r["p50_ms"] is None else f"{r['p50_ms']:.3f}",
                "" if r["p95_ms"] is None else f"{r['p95_ms']:.3f}",
                "" if r["max_ms"] is None else f"{r['max_ms']:.3f}",
            ])

    p95_by_link = {r["link"]: r["p95_ms"] for r in stats_rows}
    loss_by_link = {r["link"]: r["loss_pct"] for r in stats_rows}

    p95_matrix_path = Path(args.p95_matrix_csv)
    p95_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with p95_matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src\\dst", *destinations])
        for src in sources:
            row = [src]
            for dst in destinations:
                key = f"{src}->{dst}"
                v = p95_by_link.get(key)
                row.append("" if v is None else f"{v:.3f}")
            writer.writerow(row)

    loss_matrix_path = Path(args.loss_matrix_csv)
    loss_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with loss_matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src\\dst", *destinations])
        for src in sources:
            row = [src]
            for dst in destinations:
                key = f"{src}->{dst}"
                v = loss_by_link.get(key)
                row.append("" if v is None else f"{v:.3f}")
            writer.writerow(row)


if __name__ == "__main__":
    main()
