#!/usr/bin/env python3
"""Read-only native terrain consolidation and physically authored thesis figures.

Only the two source files and the explicitly owned output folders are written.
No profile observations, summary means, job submission, or fastest-run selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/thesis_gpu_figures"
TABLES = OUT / "tables/native"
MODELS = ("watercnn8k", "giantmlp288")
HARDWARE = ("24c1g", "24p24c1g", "96c4g", "96p96c4g")
HW_LABELS = ("24c\n1GPU", "24+24c\n1GPU", "96c\n4GPUs", "96+96c\n4GPUs")
LAYOUTS = ("default", "affine", "numa")
COMMS = ("coll", "p2p1", "p2p24")
T95 = 4.302652729911275
STEP = re.compile(r"STEP_TIMING step=(\d+) solver=ML step_ms=([0-9.eE+-]+)")
ORIGINAL = "gpu_refresh_20261012_optimized"
RETRY = "gpu_refresh_20261012_optimized_timing_retry1"
GAPS = "gpu_refresh_20261014_thesis_native_gaps"
# Original 24+24-rank allocations failed (rc=143/90); these six complete
# replacement logs match the overwritten success manifests, not file age/speed.
AIX_REPLACEMENTS = {
    "4518442": "4578479", "4518444": "4578481", "4518446": "4578483",
    "4518460": "4578485", "4518462": "4578487", "4518464": "4578489",
}


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def expected_matrix():
    variants = [("aix", f"{layout}_{comm}") for layout in LAYOUTS for comm in COMMS]
    variants += [("phydll", v) for v in ("default", "readiness")]
    variants += [("smartsim", v) for v in ("shared_c0", "shared_c1", "shared_c3", "pergpu_c0")]
    return set(itertools.product(MODELS, HARDWARE, variants, (1, 2, 3)))


def mean_ci(values):
    if len(values) != 3 or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Exactly three finite nonnegative repetition means required")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = T95 * sd / math.sqrt(3)
    return dict(mean_s=mean, sd_s=sd, ci_half_s=half, ci_low_s=mean-half,
                ci_high_s=mean+half, n_reps=3, df=2, t95=T95,
                **{f"rep{i}_s": v for i, v in enumerate(values, 1)})


def parse_calls(text, model):
    calls = [(int(m[1]), float(m[2])) for m in STEP.finditer(text)]
    n = 100 if model == "watercnn8k" else 5
    if [step for step, _ in calls] != list(range(2, 2*(n+1)+1, 2)):
        raise ValueError(f"Expected one warmup and {n} steady calls, got {len(calls)}")
    if any(not math.isfinite(v) or v < 0 for _, v in calls):
        raise ValueError("Invalid timing")
    return calls


def validate_workload(text, model):
    width, height, chunk, cap = ((7680, 4320, 60, 20000000) if model == MODELS[0]
                                else (480, 288, 16, 500000))
    for key, value in (("TARGET_WIDTH", width), ("TARGET_HEIGHT", height), ("CHUNK_SIZE", chunk),
                       ("USE_SCOREP", 0)):
        if f"Using {key} from environment variable: {value}\n" not in text:
            raise ValueError(f"Missing native/workload receipt {key}={value}")
    grid = re.search(r"Loaded grid metadata: ([^\n]+)", text)
    if not grid or any(f"{key}={value}" not in grid[1] for key, value in
                       (("nx", width), ("nz", height), ("chunk_size", chunk), ("ml_batch_size", cap))):
        raise ValueError("Missing matching loaded grid/batch receipt")
    model_path = re.findall(r"Resolved MODEL_PATH=([^\n]+)", text)
    if len(model_path) != 1:
        raise ValueError("Ambiguous model artifact")
    return dict(width=width, height=height, chunk=chunk, batch_cap=cap,
                model_path=model_path[0], grid_metadata=grid[1])


def consolidate():
    sources, errors, ledger, runs, inferences = {}, [], [], [], []

    def receipt(path):
        path = Path(path).resolve()
        data = path.read_bytes()
        sources[str(path)] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        return data.decode(errors="replace")

    def rows(path):
        return list(csv.DictReader(receipt(path).splitlines()))

    def accept(model, hw, provider, variant, rep, job, rid, text, path, status_path, reason,
               nominal="", config="", environment=""):
        identity = (model, hw, (provider, variant), int(rep))
        entry = dict(model=model, hardware=hw, provider=provider, variant=variant, rep=int(rep),
                     job_id=str(job), run_id=rid, source_path=str(Path(path).resolve()),
                     status_path=str(Path(status_path).resolve()), decision="selected", reason=reason,
                     nominal_solver_log=nominal, error="")
        ledger.append(entry)
        try:
            if identity not in expected_matrix():
                raise ValueError("Unexpected native identity")
            calls = parse_calls(text, model)
            workload = validate_workload(text, model)
            rank = re.search(r"world_size=(\d+)", workload["grid_metadata"])
            expected_ranks = int(re.match(r"\d+", hw)[0]) * (2 if "p" in hw else 1)
            if not rank or int(rank[1]) != expected_ranks:
                raise ValueError("Solver rank count mismatch")
            row = {k: entry[k] for k in ("model", "hardware", "provider", "variant", "rep", "job_id",
                                         "run_id", "source_path", "status_path")}
            row.update(workload, solver_ranks=expected_ranks, gpu_count=4 if "4g" in hw else 1,
                       n_warmup=1, n_measured=len(calls)-1, mean_s=statistics.mean(v for _, v in calls[1:])/1000,
                       exit_status=0, config_path=config, environment_path=environment,
                       tpq=1 if provider == "smartsim" else "", intra_op=8 if provider == "smartsim" else "",
                       inter_op=1 if provider == "smartsim" else "")
            runs.append(row)
            for i, (step, value) in enumerate(calls):
                inferences.append(dict(run_id=rid, job_id=str(job), call=i+1, step=step,
                                       phase="warmup" if i == 0 else "measured", latency_ms=value,
                                       source_path=entry["source_path"]))
        except ValueError as exc:
            entry.update(decision="error", error=str(exc))
            errors.append(f"{rid}: {exc}")

    # These job receipts, not the historical analyzer's pooled summaries, select AIx.
    aix_jobs = ROOT / "results/retest_aix/jobids.txt"
    for model, hw, rep, job in csv.reader(receipt(aix_jobs).splitlines()):
        reason = "Explicit retained AIx jobids.txt and matching CELL_RESULT/manifest"
        if job in AIX_REPLACEMENTS:
            original = ROOT / f"logs/retest_aix_{hw}_{job}.txt"
            old_text = receipt(original)
            if re.search(r"CELL_RESULT run_id=\S+ rc=0(?:\s|$)", old_text):
                errors.append(f"AIx replacement predecessor contains success: {job}")
                continue
            ledger.append(dict(model=model, hardware=hw, provider="aix", variant="all nine",
                               rep=int(rep), job_id=job, run_id="", source_path=str(original),
                               status_path=str(original), decision="superseded",
                               reason=f"Failed original; explicit replacement {AIX_REPLACEMENTS[job]}",
                               nominal_solver_log="", error=""))
            reason = f"Explicit AIx replacement {job}->{AIX_REPLACEMENTS[job]}; failed predecessor; matching success manifest"
            job = AIX_REPLACEMENTS[job]
        path = ROOT / f"logs/retest_aix_{hw}_{job}.txt"
        text = receipt(path)
        parts = re.split(r"\[[^\n]*\] cell run_id=(\S+)", text)
        bodies = defaultdict(list)
        for i in range(1, len(parts), 2):
            bodies[parts[i]].append(parts[i+1])
        status_path = ROOT / f"results/retest_aix/{model}/{hw}/runs_{hw}_{model}_r{rep}.csv"
        statuses = rows(status_path)
        for layout, comm in itertools.product(LAYOUTS, COMMS):
            rid = f"retest_{hw}_{model}_{layout}_{comm}_r{rep}"
            match = [r for r in statuses if r["run_id_cell"] == rid and r["rc"] == "0"]
            if len(match) != 1 or len(bodies[rid]) != 1 or f"CELL_RESULT run_id={rid} rc=0" not in bodies[rid][0]:
                errors.append(f"{rid}: ambiguous/missing successful status or log segment")
                continue
            accept(model, hw, "aix", f"{layout}_{comm}", rep, job, rid, bodies[rid][0], path,
                   status_path, reason)

    # SmartSim job list records the TPQ1 campaign and appended two-node correction.
    ss_jobs = ROOT / "results/retest_smartsim/jobids_tpq1_batch.txt"
    jobs = list(csv.reader(receipt(ss_jobs).splitlines()))
    grouped = defaultdict(list)
    for model, hw, rep, job in jobs:
        grouped[(model, hw, rep)].append(job)
    for (model, hw, rep), candidates in grouped.items():
        # The later appended jobs only target 24p24c1g, now supplied by native gaps.
        if hw == "24p24c1g":
            for job in candidates:
                ledger.append(dict(model=model, hardware=hw, provider="smartsim", variant="shared_c0",
                                   rep=int(rep), job_id=job, run_id="", source_path=str(ss_jobs),
                                   status_path=str(ss_jobs), decision="excluded", reason="24p24c1g corrected native-gap campaign is authoritative",
                                   nominal_solver_log="", error=""))
            continue
        if len(candidates) != 1:
            errors.append(f"Ambiguous TPQ1 jobs: {model}/{hw}/{rep}: {candidates}")
            continue
        job = candidates[0]
        path = ROOT / f"logs/retest_aix_{hw}_{job}.txt"
        text = receipt(path)
        parts = re.split(r"\[[^\n]*\] case run_id=(\S+)[^\n]*", text)
        for variant in (("shared_c0", "pergpu_c0") if "4g" in hw else ("shared_c0",)):
            rid = f"retest_smartsim_{hw}_{model}_{variant}_r{rep}"
            bodies = [parts[i+1] for i in range(1, len(parts), 2) if parts[i] == rid]
            if len(bodies) != 1 or f"CASE_RESULT run_id={rid} rc=0" not in bodies[0]:
                errors.append(f"{rid}: ambiguous/missing successful CASE_RESULT")
                continue
            # The controller log is append-only across old campaigns. Select the
            # unique startup section containing this Slurm allocation identity.
            controller = ROOT / f"logs/mini_app_driver_{rid}.txt"
            ctrl = receipt(controller)
            sections = re.split(r"(?=Experiment configuration:)", ctrl)
            matching = [s for s in sections if re.search(rf"SLURM_JOB_ID={job}(?:\s|$)", s)]
            if len(matching) != 1 or "intra_op_threads=8, inter_op_threads=1, threads_per_queue=1" not in matching[0]:
                errors.append(f"{rid}: ambiguous/missing job-matched TPQ1 controller receipt")
                continue
            topology = "shared" if variant == "shared_c0" else "per-ml-node"
            if f"Database layout: {topology}\n" not in matching[0]:
                errors.append(f"{rid}: DB topology receipt mismatch")
                continue
            accept(model, hw, "smartsim", variant, rep, job, rid, bodies[0], path, path,
                   "Explicit TPQ1/batch job ledger; CASE_RESULT=0; job-matched controller section")

    refresh = ROOT / "results/gpu_refresh"
    original_rows = {}
    for campaign in (ORIGINAL, RETRY, GAPS):
        base = refresh / campaign
        for group in rows(base / "groups.csv"):
            if group["mode"] != "timing":
                continue
            manifest = base / f"{group['group']}_{group['job_id']}" / "manifest.csv"
            for row in rows(manifest):
                key = (row["model"], row["hardware"], row["variant"], row["repeat"])
                if campaign == ORIGINAL:
                    if key in original_rows:
                        errors.append(f"Duplicate original timing identity: {key}")
                    original_rows[key] = (row, manifest)
                    # Retry matrix names the other three placements explicitly.
                    if row["hardware"] != "96p96c4g":
                        ledger.append(dict(model=row["model"], hardware=row["hardware"], provider="phydll",
                                           variant=row["variant"], rep=int(row["repeat"]), job_id=row["job_id"],
                                           run_id=row["run_id"], source_path=str(manifest), status_path=str(manifest),
                                           decision="superseded", reason=f"Explicit timing_retry1 replacement; original rc={row['rc']}, observed_ml_calls={row['observed_ml_calls']}",
                                           nominal_solver_log=row["solver_log"], error=""))
                        continue
                if campaign == RETRY:
                    old = original_rows.get(key)
                    if not old or old[0]["rc"] == "0":
                        errors.append(f"Retry lacks failed predecessor: {key}")
                        continue
                if row["rc"] != "0" or row["validation_error"] or row["cleanup_safe"] != "True":
                    errors.append(f"{row['run_id']}: unsuccessful selected manifest")
                    continue
                directory = manifest.parent / row["run_id"]
                env_path = directory / "runtime.env.json"
                env = json.loads(receipt(env_path))
                if (env.get("USE_SCOREP_ENV") != "0" or
                        any(env.get(k, "false") != "false" for k in
                            ("SCOREP_ENABLE_PROFILING_ENV", "SCOREP_ENABLE_TRACING_ENV"))):
                    errors.append(f"{row['run_id']}: instrumented environment")
                    continue
                provider = row["provider"].lower()
                variant = row["variant"]
                if provider == "smartsim":
                    variant = {"shared": "shared_c0", "pergpu": "pergpu_c0"}.get(variant, variant)
                    concurrency = variant[-1] if variant.startswith("shared_c") else "0"
                    required = {"SMARTSIM_THREADS_PER_QUEUE_ENV": "1", "SMARTSIM_INTRA_OP_THREADS": "8",
                                "SMARTSIM_INTER_OP_THREADS": "1", "SMARTSIM_MPI_SEQUENTIAL_PUT": concurrency}
                    if any(env.get(k) != v for k, v in required.items()):
                        errors.append(f"{row['run_id']}: SmartSim configuration mismatch")
                        continue
                config = directory / "config.toml"
                config_text = receipt(config)
                if provider == "phydll":
                    ready = "true" if variant == "readiness" else "false"
                    if (f"solver_readiness_wait = {ready}\n" not in config_text or
                            'transport_layout = "packed"' not in config_text):
                        errors.append(f"{row['run_id']}: readiness/transport mismatch")
                        continue
                receipt(directory / "driver.sha256")
                nominal = Path(row["solver_log"]).resolve()
                path = nominal if nominal.exists() else directory / "driver.log"
                text = receipt(path)
                # Exact same run directory receipt, never recursive fallback.
                if row["run_id"] not in text:
                    errors.append(f"{row['run_id']}: missing driver identity")
                    continue
                n = 100 if row["model"] == MODELS[0] else 5
                if (row["warm"], row["steady"], row["observed_ml_calls"]) != ("1", str(n), str(n+1)):
                    errors.append(f"{row['run_id']}: manifest schedule mismatch")
                    continue
                accept(row["model"], row["hardware"], provider, variant, row["repeat"], row["job_id"],
                       row["run_id"], text, path, manifest,
                       ("Explicit failed-predecessor timing retry" if campaign == RETRY else "Explicit native campaign")
                       + ("; missing nominal solver_log resolved to exact run-directory driver.log" if path != nominal else ""),
                       str(nominal), str(config), str(env_path))

    actual = defaultdict(list)
    for r in runs:
        actual[(r["model"], r["hardware"], (r["provider"], r["variant"]), r["rep"])].append(r)
    for key in sorted(expected_matrix()):
        if len(actual[key]) != 1:
            errors.append(f"Expected exactly one successful repetition: {key}, got {len(actual[key])}")
    values = []
    for model, hw, (provider, variant), _ in sorted(k for k in expected_matrix() if k[-1] == 1):
        group = [actual[(model, hw, (provider, variant), rep)] for rep in (1, 2, 3)]
        if any(len(g) != 1 for g in group):
            continue
        selected = [g[0] for g in group]
        values.append(dict(model=model, hardware=hw, provider=provider, variant=variant,
                           **mean_ci([r["mean_s"] for r in selected]), calls_per_rep=selected[0]["n_measured"],
                           run_ids=";".join(r["run_id"] for r in selected),
                           job_ids=";".join(r["job_id"] for r in selected)))
    TABLES.mkdir(parents=True, exist_ok=True)
    write_csv(TABLES / "selection_ledger.csv", ledger)
    write_csv(TABLES / "runs.csv", runs)
    write_csv(TABLES / "calls.csv", inferences)
    if values:
        write_csv(TABLES / "values.csv", values)
    write_csv(TABLES / "selection_errors.csv", [dict(error=e) for e in errors], ["error"])
    coverage = [dict(model=m, hardware=h, provider=p, variant=v,
                     selected_repetitions=sum(len(actual[(m, h, (p, v), rep)]) == 1 for rep in (1, 2, 3)),
                     expected_repetitions=3) for m, h, (p, v), rep in sorted(expected_matrix()) if rep == 1]
    write_csv(TABLES / "coverage.csv", coverage)
    receipt(Path(__file__))
    for name in ("analyze_retest_aix.py", "analyze_retest_smartsim.py", "retest_aix_cells.zsh", "retest_smartsim_cells.zsh"):
        receipt(ROOT / name)
    for path in (Path.home() / "Master-Thesis/gitlab/sourcecode/thesis_style.py",
                 Path.home() / "Master-Thesis/notes/thesis-figures-guide.md"):
        receipt(path)
    report = dict(status="error" if errors else "ok", errors=errors, expected_runs=360,
                  selected_runs=len(runs), expected_cells=120, complete_cells=len(values),
                  capacity="unavailable: telemetry not verified; no capacity bars or inferred peaks",
                  provenance=sources)
    (TABLES / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2))
    return values, runs, report


def figure(values, variants, labels, colors, models=MODELS, suppress_one_gpu=False,
           axis_height_mm=32):
    """Native full ML-step bars only; no profile-derived pipeline predictions."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    if any("SS" in label or "SmartSim" in label and label not in
           ("SmartSim", "SmartSim shared", "SmartSim per-GPU") for label in labels):
        raise ValueError("Use full, unambiguous SmartSim labels")
    width = 150 if len(models) == 2 else 73.5
    height = axis_height_mm + (22 if len(models) == 2 else 20)
    fig = plt.figure(figsize=(width/25.4, height/25.4))
    for index, model in enumerate(models):
        start = index * 76.5
        ax = fig.add_axes([(start+11)/width, 12/height, 61/width, axis_height_mm/height])
        subset = []
        for j, ((provider, variant), color) in enumerate(zip(variants, colors)):
            for i, hw in enumerate(HARDWARE):
                if suppress_one_gpu and variant == "pergpu_c0" and "1g" in hw:
                    continue
                rows = [r for r in values if (r["model"], r["hardware"], r["provider"], r["variant"])
                        == (model, hw, provider, variant)]
                x = i + (j-(len(variants)-1)/2)*(.8/len(variants))
                if not rows:
                    ax.text(x, .03, "N/A", transform=ax.get_xaxis_transform(), rotation=90,
                            fontsize=5.5, ha="center", va="bottom")
                    continue
                if len(rows) != 1:
                    raise ValueError("Duplicate plotted cell")
                row = rows[0]
                subset.append(row)
                ax.bar(x, row["mean_s"], width=.72/len(variants), color=color,
                       edgecolor=".25", linewidth=.35, yerr=row["ci_half_s"], capsize=1.2,
                       error_kw=dict(linewidth=.6, capthick=.6))
        low = min([0]+[r["ci_low_s"] for r in subset])
        high = max([1e-6]+[r["ci_high_s"] for r in subset])
        ax.set_ylim(low*1.08, high*1.08)
        ax.set_xlim(-.55, 3.55)
        ax.set_xticks(range(4), HW_LABELS, fontsize=6.5)
        ax.tick_params(length=2, pad=2)
        ax.set_ylabel("ML-step time (s)", fontsize=9, labelpad=3)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_axisbelow(True)
        ax.grid(False, axis="x")
        ax.grid(True, axis="y", linewidth=.4, color=".85", linestyle=":")
        if len(models) == 2:
            ax.set_title({MODELS[0]: "WaterCNN (8K)", MODELS[1]: "GiantMLP (480 x 288)"}[model],
                         loc="left", fontsize=9, pad=4)
    fig.legend(handles=[Patch(facecolor=c, edgecolor=".25", linewidth=.35, label=l)
                        for l, c in zip(labels, colors)], loc="upper center",
               bbox_to_anchor=(.5, .985), ncol=len(labels) if len(models) == 2 else 2,
               fontsize=6, handlelength=1, handletextpad=.35, columnspacing=.8,
               borderaxespad=0, labelspacing=.35)
    return fig


NOTES = """Native uninstrumented terrain full ML-step timings, not CUDA-forward or pipeline-only spans.
One warmup is excluded in every repetition. WaterCNN: 7680 x 4320, chunk 60,
cap 20000000, 100 measured calls; GiantMLP: 480 x 288, chunk 16, cap 500000,
5 measured calls. Each bar is the arithmetic mean of three independent
repetition means, with a 95% Student-t interval (sample SD, df=2,
t=4.302652729911275). Calls/ranks/GPUs are never independent repetitions.
Signed confidence bounds are preserved in tables and axes; no clamping.
Placements denote 24, 48, 96, 192 total solver ranks, respectively. These
are not asserted to be fixed-work strong scaling. Loaded grid metadata,
actual model artifact paths, jobs and sources are in tables/native/runs.csv.

SmartSim uses TPQ1, intra-op8, inter-op1, matching model-specific batch caps.
SmartSim upload scheduling variants are Default, c=1 and c=3, not AIx credit limits.
Raw shared_c0/shared_c1/shared_c3 keys and existing output paths are retained.
DB-per-GPU
is a baseline deployment, colored #B89B47, distinct from shared #8064A2.
Separately measured one-GPU per-DB runs are retained in the audit but omitted
from baseline bars: both topologies coincide on one GPU. No observation is
duplicated or counted as an independent deployment/repetition.

Selection is explicit, never fastest or based on modification times. AIx uses
the retained job ledger with six named failed-original 24p24c1g replacements.
PhyDLL uses the optimized campaign's successful 96p96c4g allocations and the
explicit timing_retry1 matrix for the three other placements, each checked
against a failed predecessor. SmartSim uses the TPQ1 job ledger except
24p24c1g Default, which is supplied by the native-gap campaign; the same campaign
supplies c=1/c=3 and separately measured one-GPU per-DB. Old TPQ3 and earlier
bugged per-GPU campaigns are not candidates. Append-only old controller logs
are segmented and matched to the selected allocation, not read as one run.
Missing nominal solver paths are resolved only to the exact manifest run
directory's driver.log, with identity, schedule and workload validation.
Every selected/excluded receipt is documented in the selection ledger.

Capacity telemetry has not been verified: RAM/VRAM are unavailable, not zero.
No capacity figure, guessed peak, phase fraction, or native-derived pipeline
prediction is produced. Profile phases, measured pipeline spans and timelines
are separately owned and must not be conflated with these native ML-step bars.
All three AIx layouts have their own measured native collective/P2P comparison.
ch7_layout_comparison provides the inverse: Default/Affine/NUMA side by side
for each hardware placement, with one communication mode per figure:
Collective (coll), P2P Credit=1 (p2p1), P2P Fullcredit (p2p24).
Fullcredit is a human label, not a rewritten credit count: this selected native
campaign explicitly requested 24 credits; other campaigns/MMCP may use 23.
Those 23/24 settings are not interchangeable. Actual p2p24 variant identities,
run IDs and hashed runner receipts are retained without normalization.
Layout colors are stable: Default #577787, Affine #9784B4, NUMA #568775.
Default uses the standard rank grouping. Affine uses GPU-affine rank grouping
with hardware locality support (build with WITH_HWLOC=ON). NUMA uses NUMA-local
rank groups, with WITH_HWLOC=ON and WITH_NUMA_LOCAL_GROUPS=ON.
The within-layout communication figures
hold the layout fixed; the cross-layout figures compare Default/Affine/NUMA.

Figures are authored at 150 mm, independent panels at 73.5 mm, with a 3 mm
pair gap and no resizing. Baseline/upload scheduling axes are 32 mm high, readiness
axes 40 mm, layout-comparison axes 41.6052 mm, matching the corresponding
MMCP authored data-area heights (upload scheduling retains the baseline height).
Canvases add 22 mm for paired previews or 20 mm for independent panels,
using thesis_style.apply/save, PGF/pdflatex Latin Modern, self-contained PDFs
and PDF-derived 300 dpi PNGs. No baked-in captions or methodological footers.
Independent panels omit model titles so LaTeX owns identification.
"""


def render(values, runs):
    sys.path.insert(0, str(Path.home() / "Master-Thesis/gitlab/sourcecode"))
    import thesis_style as ts
    ts.apply()
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .5, "savefig.bbox": None})
    programmes = [
        ("ch6_baselines", "latency_baselines", [("aix", "default_coll"), ("phydll", "default"),
          ("smartsim", "shared_c0"), ("smartsim", "pergpu_c0")],
         ["AIx", "PhyDLL", "SmartSim shared", "SmartSim per-GPU"],
         ["#577787", "#C97AA7", "#8064A2", "#B89B47"], True),
        ("ch6_baselines", "latency_baselines_without_pergpu",
         [("aix", "default_coll"), ("phydll", "default"), ("smartsim", "shared_c0")],
         ["AIx", "PhyDLL", "SmartSim"], ["#577787", "#C97AA7", "#8064A2"], False),
        ("ch7_readiness", "readiness_latency", [("aix", "default_coll"), ("phydll", "default"),
          ("phydll", "readiness")], ["AIx", "PhyDLL blocking", "PhyDLL readiness"],
         ["#577787", "#C97AA7", "#009E73"], False),
        ("ch7_smartsim_chaining", "chaining_latency", [("smartsim", f"shared_c{c}") for c in (0, 1, 3)],
         ["Default", "c=1", "c=3"], ["#8064A2", "#C99449", "#568775"], False),
    ]
    programmes += [("ch7_layout_pipeline_native", f"{layout}_native_latency",
                    [("aix", f"{layout}_{comm}") for comm in COMMS],
                    ["Collective", "P2P Credit=1", "P2P Fullcredit"],
                     ["#577787", "#9784B4", "#568775"], False) for layout in LAYOUTS]
    programmes += [("ch7_layout_comparison", f"{comm}_layout_latency",
                    [("aix", f"{layout}_{comm}") for layout in LAYOUTS],
                    ["Default", "Affine", "NUMA"],
                    ["#577787", "#9784B4", "#568775"], False) for comm in COMMS]
    galleries, geometry = defaultdict(list), []
    # Hash existing data products before touching layouts; new subsets are excluded.
    protected = [*TABLES.glob("*.csv")]
    protected = [p for p in protected if p.name != "figure_geometry.csv"]
    for folder in {p[0] for p in programmes}:
        protected.extend((OUT / folder).glob("*_values.csv"))
        protected.extend((OUT / folder).glob("*_runs.csv"))
    hashes = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in protected}
    for folder, name, variants, labels, colors, suppress in programmes:
        directory = OUT / folder
        directory.mkdir(parents=True, exist_ok=True)
        selected = [r for r in values if (r["provider"], r["variant"]) in variants and not
                    (suppress and r["variant"] == "pergpu_c0" and "1g" in r["hardware"])]
        write_csv(directory / f"{name}_values.csv", selected)
        write_csv(directory / f"{name}_runs.csv", [r for r in runs if (r["provider"], r["variant"]) in variants])
        for models, suffix in ((MODELS, "1x2"), ((MODELS[0],), MODELS[0]), ((MODELS[1],), MODELS[1])):
            stem = f"{name}_{suffix}"
            axis_height = (40 if folder == "ch7_readiness" else
                           41.6052 if folder in ("ch7_layout_pipeline_native", "ch7_layout_comparison") else 32)
            fig = figure(selected, variants, labels, colors, models, suppress, axis_height)
            # PGF's renderer gives actual TeX text metrics, not an Agg estimate.
            fig.draw_without_rendering()
            renderer = fig.canvas.get_renderer()
            bounds = fig.bbox
            artists = [*fig.legends]
            for ax in fig.axes:
                # Locators may propose ticks outside ylim; Matplotlib does not draw them.
                low, high = ax.get_ylim()
                artists += [ax.yaxis.label, *ax.get_xticklabels(),
                            *(tick.label1 for tick in ax.yaxis.get_major_ticks()
                              if low <= tick.get_loc() <= high)]
                if len(models) == 2:
                    artists.append(ax._left_title)
            for artist in artists:
                box = artist.get_window_extent(renderer)
                if not (box.x0 >= bounds.x0-.5 and box.y0 >= bounds.y0-.5 and
                        box.x1 <= bounds.x1+.5 and box.y1 <= bounds.y1+.5):
                    raise ValueError(f"Text outside authored canvas: {stem}: {artist}")
            legend_box = fig.legends[0].get_window_extent(renderer)
            header_top = max((ax._left_title.get_window_extent(renderer).y1 if len(models) == 2
                              else ax.get_window_extent(renderer).y1) for ax in fig.axes)
            gap_mm = (legend_box.y0-header_top) / fig.dpi * 25.4
            if not 0 < gap_mm <= 3.1:
                raise ValueError(f"Legend/header gap outside compact contract: {stem}: {gap_mm:.3f} mm")
            geometry.append(dict(figure=f"{folder}/{stem}", width_mm=fig.get_figwidth()*25.4,
                                 height_mm=fig.get_figheight()*25.4, text_bounds="inside canvas",
                                 axes_width_mm=61, axes_height_mm=axis_height,
                                 pair_gap_mm=3,
                                 legend_header_gap_mm=gap_mm))
            ts.save(fig, directory / stem)
            for ext in ("pgf", "pdf", "png"):
                if not (directory / f"{stem}.{ext}").is_file():
                    raise RuntimeError(f"Missing real render: {stem}.{ext}")
            galleries[folder].append(stem)
        (directory / f"{name}_example.tex").write_text(
            "\\documentclass[12pt]{article}\n"
            "\\usepackage[textwidth=150mm,marginparwidth=0mm]{geometry}\n"
            "\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}\n\\usepackage{pgf}\n"
            "\\begin{document}\n\\noindent\n"
            "\\begin{minipage}[t]{73.5mm}\n"
            f"\\input{{{name}_{MODELS[0]}.pgf}}\n"
            "\\end{minipage}\\hspace{3mm}%\n"
            "\\begin{minipage}[t]{73.5mm}\n"
            f"\\input{{{name}_{MODELS[1]}.pgf}}\n"
            "\\end{minipage}\n\\end{document}\n")
    titles = {
        "latency_baselines": "Provider And Deployment Baselines",
        "latency_baselines_without_pergpu": "Provider Baselines Without DB-per-GPU",
        "readiness_latency": "PhyDLL Readiness",
        "chaining_latency": "SmartSim Upload Scheduling: Default, c=1, c=3",
        **{f"{layout}_native_latency": f"AIx {label}: Communication Comparison"
           for layout, label in zip(LAYOUTS, ("Default", "Affine", "NUMA"))},
        **{f"{comm}_layout_latency": f"AIx Affinity And NUMA: {label}"
           for comm, label in zip(COMMS, ("Collective", "P2P Credit=1", "P2P Fullcredit"))},
    }
    for folder, stems in galleries.items():
        gallery = "\n\n".join(
            f"### {titles[s.rsplit('_', 1)[0]]} ({s.rsplit('_', 1)[1]})\n\n"
            f"![{titles[s.rsplit('_', 1)[0]]}]({s}.png)\n\n[PDF]({s}.pdf) | [PGF]({s}.pgf)"
            for s in stems)
        heading = ("AIx Affinity And NUMA" if folder.startswith("ch7_layout") else
                   "SmartSim Upload Scheduling" if folder == "ch7_smartsim_chaining" else
                   "Terrain Native GPU Figures")
        (OUT / folder / "README.md").write_text(f"# {heading}\n\n{gallery}\n\n## Methods And Scope\n\n{NOTES}\n\n"
            "Exact values and run inputs are beside the figures. Canonical call-level data,\n"
            "selection ledger, errors/completeness report and SHA256 source paths are in\n"
            "`../tables/native/`. Regenerate from the workspace root with:\n\n```sh\n"
            "PATH=/home/ro092286/opt/texlive/2026/bin/x86_64-linux:$PATH PYTHONDONTWRITEBYTECODE=1 "
            "/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 "
            "mini_app/render_terrain_gpu_native.py\n```\n")
    write_csv(TABLES / "figure_geometry.csv", geometry)
    for relative, before in hashes.items():
        if hashlib.sha256((OUT / relative).read_bytes()).hexdigest() != before:
            raise ValueError(f"Layout-only render changed data: {relative}")
    (TABLES / "layout_data_hashes.json").write_text(json.dumps(hashes, indent=2)+"\n")
    (TABLES / "README.md").write_text("# Native Selection And Statistics\n\n"+NOTES+
        "\n`report.json` lists exact source paths and SHA256 hashes. `selection_ledger.csv`\n"
        "records provenance decisions. `calls.csv` includes warmups explicitly; `runs.csv`\n"
        "contains measured-call means; `values.csv` contains repetition-level CIs.\n"
        f"There are 360 successful native repetitions, 120 cells, {len(geometry)} native figure stems.\n"
        "`layout_data_hashes.json` verifies unchanged existing data products.\n"
        "Each figure family has an `_example.tex` using independent PGF inputs in\n"
        "73.5 mm minipages separated by 3 mm, without resizebox or scaling.\n")
    # Index existing exports without rendering or changing other owners' figures.
    inventory = ['<h2 id="all-figures">All 60 Figure Triplets</h2>', '<div class="inventory">']
    for path in sorted(OUT.glob("*/*.pgf")):
        relative = path.relative_to(OUT).with_suffix("").as_posix()
        title = titles.get(path.stem.rsplit("_", 1)[0], path.stem.replace("_", " "))
        inventory.append(
            f'<figure><a href="{relative}.pdf"><img loading="lazy" src="{relative}.png" '
            f'alt="{title}"></a><figcaption>{title} '
            f'<a href="{relative}.png">PNG</a> | <a href="{relative}.pdf">PDF</a> | '
            f'<a href="{relative}.pgf">PGF</a></figcaption></figure>')
    inventory.append('</div>')
    index = OUT / "index.html"
    html = index.read_text()
    html = re.sub(r'<!-- figure inventory -->.*?<!-- /figure inventory -->', '', html, flags=re.S)
    html = html.replace('</html>', '<!-- figure inventory -->\n'+"\n".join(inventory)+
                        '\n<!-- /figure inventory -->\n</html>')
    index.write_text(html)
    print(f"Rendered {len(geometry)} figures, each PGF/PDF/PNG")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidate-only", action="store_true")
    args = parser.parse_args(argv)
    values, runs, report = consolidate()
    if report["errors"]:
        return 1
    if not args.consolidate_only:
        render(values, runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
