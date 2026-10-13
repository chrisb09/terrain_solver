#!/usr/bin/env python3
"""Build the HH3 overview tables (Table 1/2/3) across scenarios and layouts.

Sources:
  - results/hethires3/<s<N>_<lib>_<layout>/cmi_phase_summary.csv   (HH3 AIx + SmartSim scale)
  - results/hethires2/<s<N>_<lib>/cmi_phase_summary.csv           (HH2 SmartSim base + per_gpu_db)
  - results/<suite>/<tag>/p2p_inference_metrics.csv               (pure GPU forward, p2p)
  - logs/aix_p2p_timeline_<suite>_<tag>/                          (device-time union, p2p)

Columns (single table per metric):
  aix_coll (default/affine/numa) | aix_p2p (d/a/n) | aix_p2p_fullcredits (d/a/n)
  | smartsim_c0 | smartsim_c3 | smartsim_per_gpu_db | phydll (not run)

Writes overview_t1/t2/t3.{md,csv} into results/hethires3/.
"""

import argparse
import csv
import glob
import json
import statistics as st
from pathlib import Path

SUITE = "hethires3"   # overridden by --suite
CACHE = Path("results/hethires3/overview_metric_cache.json")

AIX_LIBS = ["aix_coll", "aix_p2p", "aix_p2p_fullcredits"]
LAYOUTS = ["default", "affine", "numa"]
SS_LIBS = ["smartsim_c0", "smartsim_c3", "smartsim_per_gpu_db"]

SCEN_NAMES = {
    1: "s1 0+4c, 1g", 2: "s2 4+0c, 1g", 3: "s3 0+24c, 1g", 4: "s4 24+0c, 1g",
    5: "s5 24+24c, 1g", 6: "s6 0+96c, 4g", 7: "s7 96+96c, 4g",
    8: "s8 192+96c, 4g", 9: "s9 384+96c, 4g", 10: "s10 768+96c, 4g",
    11: "s11 1536+96c, 4g",
}
ALL_SCENS = list(range(1, 12))


def read_phase_rows(path):
    if not path.exists():
        return []
    with open(path) as stream:
        return list(csv.DictReader(stream))


def pick(rows, region):
    for row in rows:
        if row.get("Region") != region:
            continue
        cp = row.get("Callpath", "")
        if "solver_step_ml_steady" not in cp or "warmup" in cp:
            continue
        return row
    return None


def phase_value(path, region, col):
    row = pick(read_phase_rows(path), region)
    if row is None:
        return None
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None


def device_union_ms(timeline_dir):
    """Mean over steady steps of the per-workgroup H2D..D2H interval-union (ms).

    Two-pass over the timeline dir: a cheap one-line peek per rank file to
    find the controller files (scale runs have up to 1632 rank files), then a
    full parse of only those.
    """
    controller_files = []
    for f in glob.glob(str(timeline_dir / "aix_p2p_timeline_rank_*.csv")):
        try:
            with open(f) as stream:
                if stream.readline().startswith("step,"):
                    if stream.readline().rstrip("\r\n").split(",")[4] == "1":
                        controller_files.append(f)
        except (OSError, IndexError):
            continue
    step_dev = {}
    for f in controller_files:
        with open(f) as stream:
            rows = list(csv.DictReader(stream))
        if not rows or rows[0]["is_controller"] != "1":
            continue
        wg = int(rows[0]["workgroup_rank"])
        for r in rows:
            st_num = int(r["step"])
            e = r["event"]
            if e == "torch_h2d_start":
                step_dev.setdefault(st_num, {}).setdefault(wg, {}).setdefault("h2", []).append(float(r["time_s"]))
            elif e == "torch_d2h_end":
                step_dev.setdefault(st_num, {}).setdefault(wg, {}).setdefault("d2", []).append(float(r["time_s"]))
    means = []
    for st_num, wgs in sorted(step_dev.items()):
        if st_num < 2:
            continue
        wg_unions = []
        for wg, d in wgs.items():
            if "h2" in d and "d2" in d:
                iv = sorted(zip(sorted(d["h2"]), sorted(d["d2"])))
                tot = 0.0
                cs = ce = None
                for a, b in iv:
                    if cs is None:
                        cs, ce = a, b
                    elif a <= ce:
                        ce = max(ce, b)
                    else:
                        tot += ce - cs
                        cs, ce = a, b
                if cs is not None:
                    tot += ce - cs
                wg_unions.append(tot * 1e3)
        if wg_unions:
            means.append(st.mean(wg_unions))
    return st.mean(means) if means else None


def p2p_forward_ms(path):
    per = {}
    with open(path) as stream:
        for r in csv.DictReader(stream):
            if int(r["step"]) >= 2:
                per.setdefault(int(r["step"]), []).append(float(r["torch_forward_sum_ms"]))
    if not per:
        return None
    return st.mean(st.mean(v) for v in per.values())


def results_dir_for(suite, tag):
    return Path("results") / suite / tag


def locate(scn, lib, layout=None):
    """-> (suite, tag, results_dir) or (None, None, None)."""
    if lib in AIX_LIBS:
        tag = f"s{scn}_{lib}_{layout}"
        d = results_dir_for(SUITE, tag)
        if (d / "cmi_phase_summary.csv").exists():
            return (SUITE, tag, d)
        return (None, None, None)
    # non-AIX: layout fixed to default
    if lib == "phydll_cpp":
        d = results_dir_for(SUITE, f"s{scn}_{lib}_default")
        if (d / "cmi_phase_summary.csv").exists():
            return (SUITE, f"s{scn}_{lib}_default", d)
        return (None, None, None)
    # SmartSim
    if SUITE == "hethires3" and scn <= 7:
        tag = f"s{scn}_{lib}"
        d = results_dir_for("hethires2", tag)
        if (d / "cmi_phase_summary.csv").exists():
            return ("hethires2", tag, d)
    tag = f"s{scn}_{lib}_default"
    d = results_dir_for(SUITE, tag)
    if (d / "cmi_phase_summary.csv").exists():
        return (SUITE, tag, d)
    return (None, None, None)


def metric_for(scn, lib, layout, metric):
    suite, tag, d = locate(scn, lib, layout)
    if d is None:
        return None
    key = f"{metric}|{suite}|{tag}"
    cached = _cache_get(key, d)
    if cached is not None:
        return None if cached == "" else cached
    csv_path = d / "cmi_phase_summary.csv"
    if metric == "t1":
        val = phase_value(csv_path, "solver_step_ml_steady", "Rank_Mean (ms)")
    elif metric == "t2":
        if lib in AIX_LIBS:
            if lib == "aix_coll":
                val = phase_value(csv_path, "inferenceDevice", "Controller_Rank0 (ms)")
            else:
                td = Path("logs") / f"aix_p2p_timeline_{suite}_{tag}"
                val = device_union_ms(td) if td.is_dir() else None
        elif lib == "phydll_cpp":
            val = phase_value(csv_path, "app_provider_inference", "Controller_Rank0 (ms)")
        else:
            val = phase_value(csv_path, "smartsim_run_model", "Rank_Mean (ms)")
    elif metric == "t3":
        if lib == "aix_coll":
            val = phase_value(csv_path, "torchInference::forward", "Controller_Rank0 (ms)")
        elif lib in ("aix_p2p", "aix_p2p_fullcredits"):
            m = d / "p2p_inference_metrics.csv"
            val = p2p_forward_ms(m) if m.exists() else None
        else:
            val = "n/a"
    else:
        val = None
    _cache_put(key, d, val)
    return val


def _cache_get(key, d):
    cache_path = CACHE
    if not cache_path.exists():
        return None
    try:
        cache = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    entry = cache.get(key)
    if entry is None:
        return None
    stamp = _source_stamp(d)
    if stamp is None or entry.get("stamp") != stamp:
        return None
    return entry["value"]


def _cache_put(key, d, value):
    cache = {}
    if CACHE.exists():
        try:
            cache = json.loads(CACHE.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}
    stamp = _source_stamp(d)
    cache[key] = {"value": "" if value is None else value, "stamp": stamp}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))


def set_suite(name):
    global SUITE, CACHE
    SUITE = name
    CACHE = Path("results") / name / "overview_metric_cache.json"


def _source_stamp(d):
    """Cheap invalidation: newest mtime among the metric source files."""
    stamps = []
    for p in (d / "cmi_phase_summary.csv", d / "p2p_inference_metrics.csv"):
        try:
            stamps.append(p.stat().st_mtime)
        except OSError:
            pass
    return max(stamps) if stamps else None


def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, str):
        return v
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


SHORT = {"aix_coll": "coll", "aix_p2p": "p2p", "aix_p2p_fullcredits": "p2pfc",
         "smartsim_c0": "ss c0", "smartsim_c3": "ss c3", "smartsim_per_gpu_db": "ss pgdb",
         "phydll_cpp": "phydll"}
COLUMNS = [(lib, layout, f"{SHORT[lib]}·{layout[:3]}")
           for lib in AIX_LIBS for layout in LAYOUTS]
COLUMNS += [(lib, "default", SHORT[lib]) for lib in SS_LIBS]
COLUMNS.append(("phydll_cpp", "default", SHORT["phydll_cpp"]))


def build_table(metric):
    md = ["| scenario | " + " | ".join(c[2] for c in COLUMNS) + " |",
          "|" + "---|" * (len(COLUMNS) + 1)]
    csv_rows = []
    for scn in ALL_SCENS:
        vals = []
        row_csv = {"scenario": SCEN_NAMES[scn]}
        for lib, layout, label in COLUMNS:
            v = metric_for(scn, lib, layout, metric)
            vals.append(fmt(v))
            row_csv[label] = "" if v is None else (v if isinstance(v, str) else f"{v:.3f}")
        md.append(f"| {SCEN_NAMES[scn]} | " + " | ".join(vals) + " |")
        csv_rows.append(row_csv)
    return md, csv_rows


def main():
    global CACHE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="hethires3",
                        help="Results suite under results/ (default: hethires3).")
    parser.add_argument("--pgf", action="store_true",
                        help="(ignored here; tables only)")
    args = parser.parse_args()
    set_suite(args.suite)
    out_dir = Path("results") / SUITE
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "t1": "Table 1 — avg warm ML-step time (Rank_Mean, ms)",
        "t2": "Table 2 — inference device time per step (H2D+forward+D2H union, ms)",
        "t3": "Table 3 — pure GPU forward per step (per-controller avg, ms)",
    }
    for key, h in headers.items():
        md, rows = build_table(key)
        (out_dir / f"overview_{key}.md").write_text("# " + h + "\n\n" + "\n".join(md) + "\n")
        with open(out_dir / f"overview_{key}.csv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_dir}/overview_{key}.md and .csv")


if __name__ == "__main__":
    main()
