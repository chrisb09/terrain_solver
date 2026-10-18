#!/usr/bin/env python3
"""Authorized grouped GPU refresh campaign. Never submits by default."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

MINI = Path(__file__).resolve().parent
HARDWARE = ("24c1g", "96c4g", "24p24c1g", "96p96c4g")
MODELS = ("watercnn8k", "giantmlp288")
AIX_LAYOUTS = ("default", "affine", "numa")
AIX_VARIANT_SHA = "86968b64380307a5113db8077ffe4b9494309c71"
REUSE_ORIGINAL_SOLVER_ENV = "GPU_REFRESH_REUSE_ORIGINAL_SOLVER_ENV"
ORIGINAL_MODEL_PATH_CHECK = b'''    if (!cfg.print_build_timestamp) {
        require(!cfg.model_path.empty(), "--model-path is required.");
'''
APPROVED_MODEL_PATH_CHECK = b'''    if (!cfg.print_build_timestamp) {
        if (cfg.ml_interface != "none") {
            require(!cfg.model_path.empty(), "--model-path is required when an ML interface is used.");
        }
'''


def case_fields(case):
    # Earlier shipped campaigns persist three-element case tuples.
    if isinstance(case, dict):
        return case["model"], case["provider"], case["variant"], case["layout"]
    return (*case, "default")


def missing65_profile_groups():
    """Only the remaining current-workload profiles, not native timing repeats."""
    # Rounded-up maxima of mean_step_ms over each hardware/model's nine cells:
    # results/retest_aix/summary_{watercnn8k,giantmlp288}.csv. Using the maximum
    # covers every selected layout/communication without depending on live logs.
    mean_bounds = {"24c1g": (735, 2468), "96c4g": (231, 780),
                   "24p24c1g": (566, 2561), "96p96c4g": (201, 776)}
    result = []
    for hw in HARDWARE:
        for model_index, model in enumerate(MODELS):
            cases = [dict(model=model, provider="AIX", variant=comm, layout=layout)
                     for layout in AIX_LAYOUTS for comm in ("default", "p2p1", "p2p24")
                     if not (layout == "default" and
                             (comm == "default" or (model_index == 0 and
                              hw in ("24c1g", "24p24c1g") and comm == "p2p1")))]
            if (hw == "24c1g" and model_index == 1) or hw == "24p24c1g":
                cases.append(dict(model=model, provider="SMARTSIM", variant="pergpu", layout="default"))
            # Trace pilot jobs 5006151/5006153 AIx driver totals minus solver
            # runtime: 24-4=20s and 17-5=12s. Reserve 30s startup per AIx
            # case (45s at 96 ranks), plus 30s cleanup/profile flush per case.
            # Fresh native jobs 5006190/5006218 per-GPU giant means ~2469/2712ms,
            # startup 62-14=48s / 70-16=54s. Water 24p uses a conservative
            # 4000ms bound (native job 5006209 steps 4/6/12: 2068/1929/1817ms);
            # trace SmartSim job 5006151 startup was 169-37=132s.
            aix_count = sum(c["provider"] == "AIX" for c in cases)
            ss_count = len(cases) - aix_count
            aix_startup = 45 if hw.startswith("96") else 30
            ss_mean = 4000 if model_index == 0 else (2712 if "p" in hw else 2469)
            inference = 6 * (aix_count * mean_bounds[hw][model_index] + ss_count * ss_mean) / 1000
            startup = aix_count * aix_startup + ss_count * (150 if model_index == 0 else 60)
            seconds = 60 + 1.5 * (inference + startup + 30 * len(cases))
            budget = dict(expected_calls_per_case=6, aix_mean_bound_ms=mean_bounds[hw][model_index],
                          smartsim_mean_bound_ms=ss_mean if ss_count else 0,
                          inference_seconds=inference, startup_seconds=startup,
                          cleanup_seconds=30 * len(cases), headroom_factor=1.5,
                          allocation_overhead_seconds=60, estimated_seconds=seconds)
            result.append(dict(id=f"missing65_profile_{hw}_{model}_r1", hw=hw, mode="profile",
                               repeat=1, cases=cases, minutes=max(5, math.ceil(seconds / 60)), budget=budget))
    assert len(result) == 8 and sum(len(g["cases"]) for g in result) == 65
    return result


def groups():
    result = []
    for hw in HARDWARE:
        for model in MODELS:
            for repeat in (1, 2, 3):
                order = ("default", "readiness") if repeat % 2 else ("readiness", "default")
                result.append(dict(id=f"timing_{hw}_{model}_r{repeat}", hw=hw,
                                   mode="timing", repeat=repeat,
                                   cases=[(model, "PHYDLL", v) for v in order]))
        cases = [(m, "PHYDLL", v) for m in MODELS for v in ("default", "readiness")]
        if hw in ("96c4g", "24p24c1g"):
            cases.append(("watercnn8k", "AIX", "default"))
        cases.append(("giantmlp288", "AIX", "default"))
        if hw in ("24c1g", "96c4g"):
            cases.append(("giantmlp288", "SMARTSIM", "shared"))
        # Add only the missing current-workload shared-DB upload-chain profiles.
        # Historical c3 profiles used TPQ3 / 1080p variants and are not equivalent.
        cases.extend((model, "SMARTSIM", f"shared_c{chain}")
                     for model in MODELS for chain in (1, 3))
        result.append(dict(id=f"profile_{hw}_r1", hw=hw, mode="profile", repeat=1, cases=cases))
    assert len(result) == 28 and sum(len(g["cases"]) for g in result) == 88
    return result


def profile_retry_groups():
    result = []
    limits = dict(zip(HARDWARE, (35, 45, 55, 25)))
    for group in groups():
        if group["mode"] != "profile":
            continue
        cases = group["cases"]
        # Reuse the successful 24c SmartSim smoke and 96+96c PHYDLL/AIx profiles.
        if group["hw"] == "24c1g":
            cases = [c for c in cases if c[2] not in ("shared_c1", "shared_c3")]
        elif group["hw"] == "96p96c4g":
            cases = [c for c in cases if c[1] == "SMARTSIM"]
        result.append(dict(group, id=f"profile_retry1_{group['hw']}_r1", cases=cases,
                           minutes=limits[group["hw"]]))
    assert len(result) == 4 and sum(len(g["cases"]) for g in result) == 31
    return result


def profile_recovery_group():
    return dict(id="profile_retry2_24p24c1g_r1", hw="24p24c1g", mode="profile",
                repeat=1, minutes=20,
                cases=[("watercnn8k", "SMARTSIM", "shared_c1")])


def smartsim_gap_cases(hw):
    """Native SmartSim cells missing from the earlier retest campaigns."""
    cases = [(model, "SMARTSIM", f"shared_c{chain}") for model in MODELS for chain in (1, 3)]
    if hw in ("24c1g", "24p24c1g"):
        cases.extend((model, "SMARTSIM", "pergpu") for model in MODELS)
    if hw == "24p24c1g":
        cases.extend((model, "SMARTSIM", "shared") for model in MODELS)
    assert len(cases) == {"24c1g": 6, "96c4g": 4, "24p24c1g": 8, "96p96c4g": 4}[hw]
    return cases


def thesis_native_groups():
    """Fill the 22 native SmartSim gap cells with the matrix's three repeats."""
    limits = {("24c1g", "watercnn8k"): 75, ("24c1g", "giantmlp288"): 20,
              ("96c4g", "watercnn8k"): 40, ("96c4g", "giantmlp288"): 20,
              ("24p24c1g", "watercnn8k"): 90, ("24p24c1g", "giantmlp288"): 25,
              ("96p96c4g", "watercnn8k"): 45, ("96p96c4g", "giantmlp288"): 20}
    result = []
    for hw in HARDWARE:
        gap = smartsim_gap_cases(hw)
        for model in MODELS:
            for repeat in (1, 2, 3):
                cases = [c for c in gap if c[0] == model]
                result.append(dict(id=f"thesis_native_{hw}_{model}_r{repeat}", hw=hw, mode="timing",
                                   repeat=repeat, minutes=limits[(hw, model)], cases=cases))
    assert len(result) == 24 and sum(len(g["cases"]) for g in result) == 66
    return result


def targeted_retest_groups():
    """Diagnostic repeats of three completed suspect cells, not missing profiles."""
    cells = [
        ("96p96c4g", "watercnn8k", "PHYDLL", "default", 25,
         ["4938455", "4938457", "4938459"], [8.487, 2.908, 4.789], 8.5),
        ("96c4g", "giantmlp288", "SMARTSIM", "shared", 8,
         ["4735499", "4735500", "4735501"], [1.874, .862, .775], 2.0),
        ("96c4g", "watercnn8k", "SMARTSIM", "shared_c1", 25,
         ["5006197", "5006199", "5006201"], [6.719, 4.629, 4.542], 10.7),
    ]
    result = []
    for hw, model, provider, variant, limit, jobs, means, bound in cells:
        calls = 101 if model == "watercnn8k" else 6
        budget = dict(purpose="diagnostic-retest", original_job_ids=jobs,
                      original_mean_seconds=means, native_call_bound_seconds=bound,
                      expected_calls_per_case=calls, inference_seconds=calls * bound,
                      startup_cleanup_margin_seconds=limit * 60 - calls * bound,
                      telemetry="not-integrated")
        for repeat in (1, 2, 3):
            result.append(dict(id=f"targeted_retest_{hw}_{model}_{provider.lower()}_{variant}_r{repeat}",
                               hw=hw, mode="timing", repeat=repeat, minutes=limit,
                               cases=[(model, provider, variant)], budget=dict(budget)))
    return result


def thesis_trace_groups():
    """One short five-call trace capture per library before the full expansion."""
    return [
        dict(id="thesis_trace_24c1g_r1", hw="24c1g", mode="trace", repeat=1, minutes=30,
             cases=[("watercnn8k", "PHYDLL", "default"), ("watercnn8k", "AIX", "p2p1"),
                    ("watercnn8k", "SMARTSIM", "shared"), ("watercnn8k", "SMARTSIM", "pergpu")]),
        dict(id="thesis_trace_24p24c1g_r1", hw="24p24c1g", mode="trace", repeat=1, minutes=35,
             cases=[("watercnn8k", "PHYDLL", "default"), ("watercnn8k", "AIX", "p2p1")]),
    ]


def thesis_aix_trace_groups():
    """Six default-layout water traces using the isolated Score-P AIx build."""
    # Latest missing65_20261016_memory_retry1 driver.log total-time maxima
    # (AIx only), not solver inference times or the older fast trace pilots.
    # sacct start/end elapsed includes allocation setup and between-case work:
    # 5050769 12:07:54-12:13:15; 5050771 13:49:38-14:03:19;
    # 5050773 23:13:52-23:24:27; 5050777 18:46:54-19:02:13.
    measured = {"24c1g": ("5050769", 321, 7, 67),
                "96c4g": ("5050771", 821, 8, 103),
                "24p24c1g": ("5050773", 635, 8, 83),
                "96p96c4g": ("5050777", 919, 8, 140)}
    result = []
    for hw in HARDWARE:
        cases = [dict(model="watercnn8k", provider="AIX", variant=variant, layout="default")
                 for variant in (("default", "p2p1") if hw.startswith("96") else ("default",))]
        job, elapsed, count, driver = measured[hw]
        # Group 5050773 also contains SmartSim; use only AIx driver totals.
        # Reserve 30s/case for cleanup/trace flush, 50% trace headroom, and
        # 60s allocation overhead. Six minutes is the single-case floor.
        seconds = 60 + 1.5 * len(cases) * (driver + 30)
        budget = dict(measured_campaign="missing65_20261016_memory_retry1", measured_job_id=job,
                      measured_group_seconds=elapsed, measured_group_cases=count,
                      measured_aix_driver_max_seconds=driver, expected_calls_per_case=6,
                      driver_seconds=len(cases) * driver, cleanup_seconds=30 * len(cases),
                      headroom_factor=1.5, allocation_overhead_seconds=60, estimated_seconds=seconds)
        result.append(dict(id=f"thesis_aix_trace_{hw}_r1", hw=hw, mode="trace", repeat=1,
                           cases=cases, minutes=max(6, math.ceil(seconds / 60)), budget=budget))
    return result


def minutes(group):
    if "minutes" in group:
        return group["minutes"]
    if group["mode"] == "profile":
        return 65 if "p" in group["hw"] else 55
    if group["cases"][0][0] == "giantmlp288":
        return 20
    return 75 if "p" in group["hw"] else (60 if group["hw"] == "24c1g" else 40)


def case_env(group, case):
    model, provider, variant, layout = case_fields(case)
    ranks = 96 if group["hw"].startswith("96") else 24
    gpus = ranks // 24
    water = model == "watercnn8k"
    profile = group["mode"] == "profile"
    traced = group["mode"] == "trace"
    instrumented = profile or traced
    steps = 12 if instrumented or not water else 202
    cap = 20000000 if water else 500000
    build = "scorep" if instrumented else "plain"
    chain_count = {"shared_c1": "1", "shared_c3": "3"}.get(variant, "0")
    pergpu = provider == "SMARTSIM" and variant == "pergpu"
    pipelined = provider == "AIX" and variant in ("p2p1", "p2p24")
    cmi = MINI.parent / "CPP-ML-Interface"
    env = dict(USE_SCOREP_ENV=str(int(instrumented)), USE_SCOREP=str(int(instrumented)),
               SCOREP_MPP_ENV="mpi", SKIP_COMPILE_ENV="1",
               COMPILE_OUTPUT_PATH_ENV=f"build_gpu_refresh_{build}",
               SOLVER_BIN=str(MINI / "solver_cpp" / f"build_gpu_refresh_{build}" / "terrain_solver"),
               PHYDLL_DL_BUILD_DIR=str(cmi / "dl_clients" / f"build-gpu-refresh-{build}"),
               PHYDLL_DL_CLIENT=str(cmi / "dl_clients" / f"build-gpu-refresh-{build}" / "phydll_dl_client"),
               AIX_INSTALL_PREFIX_ENV=str(cmi / "extern/AIxeleratorService" / ("INSTALL-SCOREP" if instrumented else "INSTALL")),
               USE_SMARTSIM="0", USE_CPP_ML_INTERFACE="1", USE_DIRECT_AIX="0", ML_INTERFACE_ENV="cpp",
               CPP_ML_INTERFACE_PROVIDER_ENV=provider,
               MODEL_NAME_ENV="watercnn" if water else "benchmark_giant_mlp",
               TARGET_WIDTH_ENV="7680" if water else "480", TARGET_HEIGHT_ENV="4320" if water else "288",
               CHUNK_SIZE_ENV="60" if water else "16", ML_BATCH_SIZE_ENV=str(cap),
               TOTAL_STEPS_ENV=str(steps), SAVE_EVERY_ENV=str(steps), MPI_RANKS_ENV=str(ranks),
               GPUS_PER_NODE_ENV=str(gpus), HET_G_SIDE_TASKS_ENV=str(ranks), ML_INFERENCE_CPU_CORES_ENV="24",
               RANK_GRID_X_ENV="" if "p" in group["hw"] else ("12" if ranks == 96 else "6"),
               RANK_GRID_Z_ENV="" if "p" in group["hw"] else ("8" if ranks == 96 else "4"),
               FORCE_FRESH_RUN_ENV="1", OVERWRITE_OUTPUT_ENV="0", SKIP_RENDERING_ENV="1",
                USE_LOCAL_RUNTIME_STAGE_ENV="0", SMARTSIM_CREATE_RUNTIME_TARS="0",
                USE_LOCAL_MODEL_CACHE_ENV="1", SOLVER_CPU_BIND_ENV="cores",
               SMARTSIM_RUNTIME_ROOT_ENV=os.environ.get("GPU_REFRESH_RUNTIME_ROOT_ENV", str(MINI.parent / "python")),
               NIC_SAMPLING_ENV="0", GPU_MEM_SAMPLING_ENV="0",
               AIX_P2P_TIMELINE_ENV="1" if pipelined else "0",
               AIX_COMMUNICATION_MODE_ENV="pipelined" if pipelined else "collective",
               DB_NODES_ENV=str(gpus) if pergpu else "1",
               DB_LAYOUT_ENV="per-ml-node" if pergpu else "shared",
               SMARTSIM_THREADS_PER_QUEUE_ENV="1", SMARTSIM_THREADS_PER_QUEUE="1",
                SMARTSIM_INTRA_OP_THREADS="8", SMARTSIM_INTER_OP_THREADS="1", SMARTSIM_MPI_SEQUENTIAL_PUT=chain_count,
               ML_DB_BATCH_SIZE_ENV=str(cap), PHYDLL_PIN_ENV="1", PHYDLL_PIN_LOG_ENV="1",
               PHYDLL_PIN_CPU_BUDGET_ENV=str(ranks), PHYDLL_DL_NODES_ENV="1",
               PHYDLL_NP_DL_ENV=str(gpus), PHYDLL_NP_DL=str(gpus), PHYDLL_DL_COUNT="1", PHYDLL_DL_FIELD_COUNT="1",
               PHYDLL_REBUILD_DL_CLIENT_ENV="0", PHYDLL_REBUILD_ENV="0", USE_PYTHON_DL_CLIENT="0",
               MLCOUPLING_INTRA_OP_THREADS="1", MLCOUPLING_INTER_OP_THREADS="1", OMP_NUM_THREADS="1",
               MLCOUPLING_MEM_LOG_VERBOSE="0", MLCOUPLING_MEM_LOG_COLLECTIVE="0", MLCOUPLING_STEP_TIMING_SYNC="1",
               SCOREP_ENABLE_TRACING_ENV=str(traced).lower(), SCOREP_ENABLE_TRACING=str(traced).lower(),
               SCOREP_ENABLE_PROFILING_ENV=str(instrumented).lower(), SCOREP_ENABLE_PROFILING=str(instrumented).lower(),
               SCOREP_SHARED_PROFILE_ENV="1", SKIP_PAPI_METRICS="1", SCOREP_METRIC_PAPI="",
               SCOREP_DIR_TAG_ENV="gpu_refresh", OVERWRITE_JOB_NAME_ENV="1", APPLY_CUDA_STUBS_ENV="0")
    env["CPP_ML_CONFIG_ENV"] = (f"config_phydll_gpu_refresh_{variant}.toml" if provider == "PHYDLL"
                                 else ("config_aix_pipelined.toml" if pipelined else "config_aix.toml")
                                 if provider == "AIX" else "config.toml")
    if pipelined:
        env["AIX_P2P_INITIAL_CREDITS_ENV"] = {"p2p1": "1", "p2p24": "24"}[variant]
        # The solver calls MPI_Init, so the async controller must stay explicit.
        env["AIX_P2P_ASYNC_CONTROLLER"] = "0"
    if provider == "AIX" and isinstance(case, dict):
        prefix = MINI / "solver_cpp/build_thesis_aix_variants" / f"{layout}_{build}" / "aix-install"
        env["AIX_INSTALL_PREFIX_ENV"] = str(prefix)
        env["GPU_REFRESH_AIX_LIBRARY_DIR_ENV"] = str(prefix / "lib")
    return env


def aix_ready(layout, build):
    if layout not in AIX_LAYOUTS or build not in ("plain", "scorep"):
        raise ValueError("Invalid AIx layout/build")
    variant = MINI / "solver_cpp/build_thesis_aix_variants" / f"{layout}_{build}"
    text = (variant / "READY").read_text()
    entries = dict(line.split("=", 1) for line in text.splitlines())
    prefix = variant / "aix-install"
    library = prefix / "lib/libAIxeleratorService.so"
    checksums = variant / "checksums.sha256"
    expected = dict(format="THESIS_AIX_READY_V1", layout=layout, mode=build,
                    gitsha=AIX_VARIANT_SHA, gpu_refresh_job=os.environ["GPU_REFRESH_BUILD_JOB_ID_ENV"],
                    prefix=str(prefix), library=str(library), cache=str(variant / "aix-build/CMakeCache.txt"),
                    solver=str(MINI / "solver_cpp" / f"build_gpu_refresh_{build}" / "terrain_solver"),
                    dl_client=str(MINI.parent / "CPP-ML-Interface/dl_clients" / f"build-gpu-refresh-{build}" / "phydll_dl_client"),
                    checksums=str(checksums), checksums_sha256=hashlib.sha256(checksums.read_bytes()).hexdigest(),
                    validation="cpu-build-only", runtime_validation="pending")
    if any(entries.get(k) != v for k, v in expected.items()) or not re.fullmatch(r"[0-9]+", entries.get("job", "")):
        raise RuntimeError(f"{variant}: READY contract mismatch")
    if entries.get("library_size") != str(library.stat().st_size) or entries.get("library_mtime") != str(int(library.stat().st_mtime)):
        raise RuntimeError(f"{variant}: library metadata mismatch")
    # Require coverage of the selected DSO, sources, and reused executables.
    covered = {line.split(maxsplit=1)[1].lstrip(" *") for line in checksums.read_text().splitlines()}
    if not {str(library), expected["solver"], expected["dl_client"], str(variant / "source.sha256")} <= covered:
        raise RuntimeError(f"{variant}: incomplete checksums")
    subprocess.run(["sha256sum", "--check", str(checksums)], check=True, stdout=subprocess.DEVNULL)
    linkage = (variant / "library.ldd.txt").read_text()
    if re.search(r"not found|/stubs|libscorep_mpp_mockup|libscorep_adapter_(compiler|cuda)", linkage):
        raise RuntimeError(f"{variant}: invalid library linkage")
    if build == "scorep" and not all(s in linkage for s in ("libscorep_mpp_mpi", "libscorep_adapter_user")):
        raise RuntimeError(f"{variant}: missing Score-P MPI/user linkage")
    return text


def check_sources(provenance):
    manifest = provenance / "source.sha256"
    if os.environ.get(REUSE_ORIGINAL_SOLVER_ENV) != "1":
        subprocess.run(["sha256sum", "--check", str(manifest)], check=True, stdout=subprocess.DEVNULL)
        return None, None
    source = MINI / "solver_cpp/terrain_solver.cpp"
    lines = manifest.read_text().splitlines(keepends=True)
    solver_lines = [line for line in lines if line[66:].rstrip("\n") == str(source)]
    if len(solver_lines) != 1:
        raise RuntimeError("Original solver review requires exactly one solver source record")
    current = source.read_bytes()
    if current.count(APPROVED_MODEL_PATH_CHECK) != 1:
        raise RuntimeError("Original solver review requires exactly one approved patch")
    original = current.replace(APPROVED_MODEL_PATH_CHECK, ORIGINAL_MODEL_PATH_CHECK, 1)
    digest = hashlib.sha256(original).hexdigest()
    if digest != solver_lines[0][:64]:
        raise RuntimeError("Original solver reverse patch does not match recorded source SHA-256")
    subprocess.run(["sha256sum", "--check"], input="".join(line for line in lines if line != solver_lines[0]),
                   text=True, check=True, stdout=subprocess.DEVNULL)
    review = dict(approval="KeepOriginalBuild", build_identity=provenance.name.removeprefix("provenance-"),
                  original_binary_profile="plain", provenance=str(provenance), source=str(source),
                  current_sha256=hashlib.sha256(current).hexdigest(), original_sha256=digest,
                  source_original_snapshot="terrain_solver.original.cpp",
                  approved_patch=dict(original=ORIGINAL_MODEL_PATH_CHECK.decode(), current=APPROVED_MODEL_PATH_CHECK.decode(),
                                      replacements=1, scope="Model-path requirement only for ml_interface != none; ML paths unchanged"),
                  binaries_manifest_sha256=hashlib.sha256((provenance / "binaries.sha256").read_bytes()).hexdigest())
    return review, original


def check_original_solver_groups(selected_groups):
    if os.environ.get(REUSE_ORIGINAL_SOLVER_ENV) == "1":
        authorized = {g["id"] for g in targeted_retest_groups()}
        if any(g["id"] not in authorized or g["mode"] != "timing" for g in selected_groups):
            raise RuntimeError("Original solver reuse is restricted to targeted_retest_groups plain timing")


def archive_original_solver(root):
    if os.environ.get(REUSE_ORIGINAL_SOLVER_ENV) == "1":
        provenance = MINI / "solver_cpp/build_gpu_refresh_plain" / f"provenance-{os.environ['GPU_REFRESH_BUILD_JOB_ID_ENV']}"
        review, original = check_sources(provenance)
        (root / review["source_original_snapshot"]).write_bytes(original)
        (root / "original-solver-review.json").write_text(json.dumps(review, indent=2) + "\n")


def ready(build, aix_layouts=()):
    if os.environ.get(REUSE_ORIGINAL_SOLVER_ENV) == "1" and build != "plain":
        raise RuntimeError("Original solver reuse requires plain timing binaries")
    solver = MINI / "solver_cpp" / f"build_gpu_refresh_{build}"
    dl = MINI.parent / "CPP-ML-Interface/dl_clients" / f"build-gpu-refresh-{build}"
    identity = os.environ.get("GPU_REFRESH_BUILD_JOB_ID_ENV", "")
    if not re.fullmatch(r"[0-9]+", identity):
        raise RuntimeError("GPU_REFRESH_BUILD_JOB_ID_ENV must identify the completed fresh build job")
    marker = (solver / "BUILD_READY").read_text()
    if marker != (dl / "BUILD_READY").read_text() or not re.match(rf"BUILD_READY job={identity} mode={build} completed=\S+ validation=cpu-build-only runtime_validation=pending\n", marker):
        raise RuntimeError(f"{build}: nonmatching BUILD_READY markers")
    # Read build.env in an isolated shell; never let it override the case environment.
    reply = subprocess.check_output(["bash", "-c", 'set -eu; source "$1"; printf "%s\\n" "$SOLVER_BIN" "$PHYDLL_DL_CLIENT" "$USE_SCOREP" "$SCOREP_MPP" "$PROVENANCE" "$PHYDLL_BUILD_DIR" "$GPU_REFRESH_LIBRARY_PATH"',
                                     "guard", str(solver / "build.env")], text=True).splitlines()
    expected = [str(solver / "terrain_solver"), str(dl / "phydll_dl_client"),
                "1" if build == "scorep" else "0", "mpi", str(solver / f"provenance-{identity}"),
                str(MINI.parent / "CPP-ML-Interface/extern/phydll/build-gpu-refresh")]
    if len(reply) != 7 or reply[:6] != expected or expected[5] + "/lib" not in reply[6].split(":"):
        raise RuntimeError(f"{build}: unexpected build.env: {reply}")
    for binary in (solver / "terrain_solver", dl / "phydll_dl_client"):
        if not os.access(binary, os.X_OK):
            raise RuntimeError(f"Not executable: {binary}")
    for directory in (solver, dl):
        cache = (directory / "CMakeCache.txt").read_text()
        for flag, value in (("WITH_SCOREP", "ON" if build == "scorep" else "OFF"), ("PHYDLL_DL_WITH_TORCH", "ON")):
            if f"{flag}:BOOL={value}" not in cache.splitlines():
                raise RuntimeError(f"{directory}: missing {flag}={value}")
        if "CMAKE_BUILD_TYPE:STRING=Release" not in cache.splitlines():
            raise RuntimeError(f"{directory}: nonproduction build type")
        if build == "scorep" and not re.search(r"^CPPML_SCOREP_MPP:.*=mpi$", cache, re.M):
            raise RuntimeError("Missing MPI instrumentation")
    cache = (solver / "CMakeCache.txt").read_text().splitlines()
    for flag in ("USE_CPP_ML_INTERFACE", "WITH_PHYDLL", "WITH_SMARTSIM", "WITH_AIX", "WITH_TORCH"):
        if f"{flag}:BOOL=ON" not in cache:
            raise RuntimeError(f"Missing {flag}; driver could force a rebuild")
    provenance = solver / f"provenance-{identity}"
    check_sources(provenance)
    subprocess.run(["sha256sum", "--check", str(provenance / "binaries.sha256")], check=True, stdout=subprocess.DEVNULL)
    for name in ("terrain_solver", "phydll_dl_client"):
        linkage = (provenance / f"{name}.ldd.txt").read_text()
        if ("libscorep_mpp_mpi" not in linkage if build == "scorep" else "libscorep" in linkage):
            raise RuntimeError(f"{build}: invalid Score-P linkage for {name}")
    if "solver_readiness_wait" not in (solver / "generated_registry.hpp").read_text():
        raise RuntimeError("Readiness registry missing")
    if (solver / "generated_registry.hpp").read_bytes() != (solver / "cpp-ml-interface-build/generated_registry.hpp").read_bytes():
        raise RuntimeError("Generated registries differ")
    # Audit the only conditional paths that can silently reset SKIP_COMPILE.
    for driver in ("5r_driver_single.zsh", "5r_driver_hetsurrogate.zsh"):
        text = (MINI / driver).read_text()
        if text.count("SKIP_COMPILE=0") != 3 or "WITH_PHYDLL:BOOL=ON" not in text:
            raise RuntimeError(f"{driver}: compile source guard changed; review before launch")
    subprocess.run(["sha256sum", "--check"], input="\n".join(marker.splitlines()[1:]) + "\n", text=True,
                    check=True, stdout=subprocess.DEVNULL)
    for layout in aix_layouts:
        aix_ready(layout, build)
    return marker


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def submit(mode, campaign, selected_groups=None, notifier="swatch", smoke_first=False):
    """Optionally gate later groups on the first; dry runs use afterok:DRY_RUN."""
    selected_groups = groups() if selected_groups is None else selected_groups
    if mode not in ("--dry-run", "--submit") or not selected_groups:
        raise ValueError("Expected dry-run/submit and at least one group")
    check_original_solver_groups(selected_groups)
    markers = {}
    selected_layouts = {}
    for build in sorted({"scorep" if g["mode"] in ("profile", "trace") else "plain" for g in selected_groups}):
        layouts = sorted({case_fields(c)[3] for g in selected_groups for c in g["cases"]
                          if ("scorep" if g["mode"] in ("profile", "trace") else "plain") == build
                          and isinstance(c, dict) and c["provider"] == "AIX"})
        selected_layouts[build] = layouts
        markers[build] = ready(build, layouts) if layouts else ready(build)
    root = MINI / "results/gpu_refresh" / campaign
    if root.exists():
        raise RuntimeError("Campaign already exists; use a fresh token")
    if notifier not in ("swatch", "snotify"):
        raise ValueError(f"Unknown notifier: {notifier}")
    watcher = os.environ.get(f"GPU_REFRESH_{notifier.upper()}", str(Path.home() / "scripts" / notifier))
    if mode == "--submit" and not os.access(watcher, os.X_OK):
        raise RuntimeError("Mandatory swatch executable missing")
    root.mkdir(parents=True)
    archive_original_solver(root)
    (root / "ready.json").write_text(json.dumps(markers, indent=2) + "\n")
    (root / "budget.json").write_text(json.dumps({g["id"]: g["budget"] for g in selected_groups if "budget" in g}, indent=2) + "\n")
    (root / "aix-ready.json").write_text(json.dumps({f"{layout}_{build}": aix_ready(layout, build)
        for build in markers for layout in selected_layouts[build]}, indent=2) + "\n")
    group_rows, case_rows, ids = [], [], []
    cleanenv = {key: value for key, value in os.environ.items()
                if not key.startswith(("SLURM_", "SBATCH_", "SRUN_"))}
    for group in selected_groups:
        cmd = ["sbatch", "--parsable", f"--time=00:{minutes(group):02d}:00" if minutes(group) < 60 else f"--time=01:{minutes(group)-60:02d}:00",
               f"--chdir={MINI}", f"--output={root}/{group['id']}_%j.out", f"--error={root}/{group['id']}_%j.err",
               f"--export=ALL,GPU_REFRESH_MINI_ENV={MINI},GPU_REFRESH_CAMPAIGN_ENV={campaign},GPU_REFRESH_BUILD_JOB_ID_ENV={os.environ['GPU_REFRESH_BUILD_JOB_ID_ENV']}",
               str(MINI / f"gpu_refresh_{group['hw']}.sbatch"), group["id"]]
        if "p" in group["hw"]:
            cmd[-2:-2] = [":", cmd[2]]
        if smoke_first and group_rows:
            # Dependencies are global Slurm options, including for het jobs.
            cmd.insert(2, f"--dependency=afterok:{group_rows[0]['job_id']}")
            cmd.insert(3, "--kill-on-invalid-dep=yes")
        print(shlex.join(cmd), flush=True)
        jid = "DRY_RUN"
        if mode == "--submit":
            reply = subprocess.check_output(cmd, text=True, env=cleanenv).strip()
            if not re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", reply):
                raise RuntimeError(f"Unexpected sbatch reply: {reply}")
            jid = reply.split(";")[0]
            ids.append(jid)
        group_rows.append(dict(group=group["id"], mode=group["mode"], hardware=group["hw"], repeat=group["repeat"], minutes=minutes(group), job_id=jid))
        for index, case in enumerate(group["cases"], 1):
            env = case_env(group, case)
            model, provider, variant, layout = case_fields(case)
            case_rows.append(dict(group=group["id"], order=index, model=model, provider=provider, variant=variant, layout=layout,
                                  aix_library_dir=env.get("GPU_REFRESH_AIX_LIBRARY_DIR_ENV", ""), job_id=jid,
                                  steps=env["TOTAL_STEPS_ENV"], cap=env["ML_BATCH_SIZE_ENV"], warm=1,
                                  steady=int(env["TOTAL_STEPS_ENV"]) // 2 - 1))
        write_csv(root / "groups.csv", group_rows, list(group_rows[0]))
        write_csv(root / "cases.csv", case_rows, list(case_rows[0]))
    if ids:
        notify_args = (["--jobid"] if notifier == "swatch" else
                       ["-N", "GPU Refresh Profiles", "-m", f"{campaign}: {len(selected_groups)} allocations"])
        subprocess.run([watcher, *notify_args, *ids], check=True,
                       env={**os.environ, "SBATCH_ACCOUNT": "thes2181"})
    print(f"{len(selected_groups)} allocations; {sum(len(g['cases']) for g in selected_groups)} cases; manifests: {root}")


def clean_steps(job):
    """Cancel only non-batch steps in this exclusive allocation, never other jobs."""
    reply = subprocess.check_output(["squeue", "--steps", "--noheader", f"--jobs={job}", "--format=%i"], text=True)
    steps = [s.strip() for s in reply.splitlines() if re.fullmatch(rf"{job}(?:\+\d+)?\.\d+", s.strip())]
    for step in steps:
        subprocess.run(["scancel", step], check=True)
    for _ in range(30):
        reply = subprocess.check_output(["squeue", "--steps", "--noheader", f"--jobs={job}", "--format=%i"], text=True)
        if not any(s.strip() in steps for s in reply.splitlines()):
            return
        time.sleep(1)
    raise RuntimeError("Allocation steps did not drain; refusing next case")


def run(group, campaign):
    check_original_solver_groups([group])
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job):
        raise RuntimeError("Slurm allocation required")
    ranks = 96 if group["hw"].startswith("96") else 24
    if "p" in group["hw"]:
        if os.environ.get("SLURM_HET_SIZE") != "2" or any(os.environ.get(f"SLURM_NTASKS_HET_GROUP_{i}") != str(ranks) or os.environ.get(f"SLURM_JOB_NUM_NODES_HET_GROUP_{i}") != "1" for i in (0, 1)):
            raise RuntimeError("Het allocation mismatch")
    elif os.environ.get("SLURM_NTASKS") != str(ranks) or os.environ.get("SLURM_JOB_NUM_NODES") != "1":
        raise RuntimeError("Single allocation mismatch")
    build = "scorep" if group["mode"] in ("profile", "trace") else "plain"
    layouts = sorted({case_fields(c)[3] for c in group["cases"] if isinstance(c, dict) and c["provider"] == "AIX"})
    if layouts:
        ready(build, layouts)
    else:
        ready(build)
    root = MINI / "results/gpu_refresh" / campaign / f"{group['id']}_{job}"
    root.mkdir(parents=True, exist_ok=False)
    archive_original_solver(root)
    rows = []
    failed = False
    for index, case in enumerate(group["cases"], 1):
        model, provider, variant, layout = case_fields(case)
        run_id = f"gpu_refresh_{campaign}_{group['id']}_{job}_{index}_{model}_{provider.lower()}_{variant}"
        if isinstance(case, dict):
            run_id += f"_{layout}"
        out = root / run_id
        out.mkdir()
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(("AIX_P2P_", "SMARTSIM_DB_", "SCOREP_")) or key in ("SSDB", "GPU_REFRESH_AIX_LIBRARY_DIR_ENV", "SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU"):
                del env[key]
        # Exclusive GPU allocations request --mem=0, including older exports.
        env["SLURM_MEM_PER_NODE"] = "0"
        env.update(case_env(group, case))
        env.update(RUN_ID_ENV=run_id, JOB_NAME_ENV=run_id, SMARTSIM_PORT_OFFSET_ENV=str(index * 31))
        env.update(GPU_REFRESH_OUTPUT_ROOT=str(root), GPU_REFRESH_BUILD_ENV=str(MINI / "solver_cpp" / f"build_gpu_refresh_{build}" / "build.env"))
        env["GPU_REFRESH_AIX_RESOLUTION_LOG_ENV"] = str(out / "solver.ldd.txt")
        config = (MINI / env["CPP_ML_CONFIG_ENV"]).read_text()
        if provider == "PHYDLL":
            config = config.replace("batch_size = 500000", f"batch_size = {env['ML_BATCH_SIZE_ENV']}")
        (out / "config.toml").write_text(config)
        env["CPP_ML_CONFIG_ENV"] = str(out / "config.toml")
        (out / "runtime.env.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
        if "GPU_REFRESH_AIX_LIBRARY_DIR_ENV" in env:
            (out / "aix.READY").write_text(aix_ready(layout, build))
        driver = MINI / ("5r_driver_hetsurrogate.zsh" if "p" in group["hw"] else "5r_driver_single.zsh")
        (out / "driver.sha256").write_text(hashlib.sha256(driver.read_bytes()).hexdigest() + "\n")
        rc = 1
        safe = False
        try:
            clean_steps(job)
            with (out / "driver.log").open("w") as log:
                proc = subprocess.Popen(["zsh", str(MINI / "gpu_refresh_case.zsh"), str(driver)], cwd=MINI, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = proc.wait()
                finally:
                    # The driver's controller can survive an early solver failure.
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        time.sleep(1)
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            clean_steps(job)
            for name in (f"db_hostname_{run_id}.txt", f"close_driver_{run_id}.txt"):
                (MINI / name).unlink(missing_ok=True)
            if layouts:
                ready(build, layouts)
            else:
                ready(build)
            safe = True
        finally:
            expected_calls = int(env["TOTAL_STEPS_ENV"]) // 2
            log_path = out / "driver.log"
            text = log_path.read_text(errors="replace") if log_path.exists() else ""
            ml_steps = [int(step) for step in re.findall(r"STEP_TIMING step=(\d+) solver=ML step_ms=", text)]
            validation_error = ""
            if rc == 0 and ml_steps != list(range(2, int(env["TOTAL_STEPS_ENV"]) + 1, 2)):
                validation_error = f"Expected {expected_calls} unique ML calls, found steps {ml_steps}"
                rc = 1
            profiles = [p for p in (MINI / "scorep_runs").glob(f"gpu_refresh_{run_id}/**/*.cubex") if p.stat().st_size > 0]
            (out / "profiles.txt").write_text("".join(f"{p}\n" for p in profiles))
            if build == "scorep" and not profiles:
                validation_error += "; Missing nonempty Score-P profile"
                rc = rc or 1
            traces = sorted(p for p in (MINI / "scorep_runs").glob(f"gpu_refresh_{run_id}/**/traces.otf2")
                            if p.stat().st_size > 0)
            (out / "traces.txt").write_text("".join(f"{p}\n" for p in traces))
            if group["mode"] == "trace" and not traces:
                validation_error += "; Missing nonempty OTF2 trace"
                rc = rc or 1
            timelines = sorted((MINI / "logs").glob(f"aix_p2p_timeline_{run_id}/aix_p2p_timeline_rank_*.csv"))
            if provider == "AIX" and variant in ("p2p1", "p2p24") and not timelines:
                validation_error += "; Missing AIx P2P timeline CSV"
                rc = rc or 1
            rows.append(dict(run_id=run_id, model=model, provider=provider, variant=variant, layout=layout,
                             aix_library_dir=env.get("GPU_REFRESH_AIX_LIBRARY_DIR_ENV", ""), rc=rc, cleanup_safe=safe))
            rows[-1].update(group=group["id"], job_id=job, hardware=group["hw"], mode=group["mode"],
                             repeat=group["repeat"], order=index, steps=env["TOTAL_STEPS_ENV"], warm=1,
                             steady=int(env["TOTAL_STEPS_ENV"]) // 2 - 1, cap=env["ML_BATCH_SIZE_ENV"],
                             observed_ml_calls=len(ml_steps), validation_error=validation_error,
                            config=str(out / "config.toml"), profiles=str(out / "profiles.txt"),
                            solver_log=str(MINI / "logs" / f"mini_app_output_{run_id}.txt"),
                            controller_log=str(MINI / "logs" / f"mini_app_driver_{run_id}.txt"),
                            trajectory=str(out / "world_trajectory.h5"))
            write_csv(out / "manifest.csv", [rows[-1]], list(rows[-1]))
            write_csv(root / "manifest.csv", rows, list(rows[-1]))
            failed |= rc != 0 or not safe
        if not safe or (rc != 0 and group["id"].startswith("missing65_profile_")):
            break
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("--dry-run", "--submit", "--list", "--describe", "--run"))
    parser.add_argument("token", nargs="?")
    args = parser.parse_args(["--", *os.sys.argv[1:]])
    if args.action == "--list":
        print(json.dumps(groups(), indent=2))
        return 0
    if args.action in ("--describe", "--run"):
        authorized = (groups() + profile_retry_groups() + [profile_recovery_group()]
                       + thesis_native_groups() + thesis_trace_groups() + missing65_profile_groups()
                        + thesis_aix_trace_groups() + targeted_retest_groups())
        group = next((g for g in authorized if g["id"] == args.token), None)
        if group is None:
            parser.error("Unauthorized group")
        if args.action == "--describe":
            print(json.dumps([case_env(group, c) for c in group["cases"]], indent=2))
            return 0
        campaign = os.environ.get("GPU_REFRESH_CAMPAIGN_ENV", "")
    else:
        campaign = args.token or ""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", campaign):
        parser.error("Invalid campaign token")
    if args.action == "--run":
        return run(group, campaign)
    submit(args.action, campaign)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        os.sys.exit(f"ERROR: {error}")
