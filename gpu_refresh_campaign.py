#!/usr/bin/env python3
"""Authorized grouped GPU refresh campaign. Never submits by default."""
import argparse
import csv
import hashlib
import json
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


def minutes(group):
    if "minutes" in group:
        return group["minutes"]
    if group["mode"] == "profile":
        return 65 if "p" in group["hw"] else 55
    if group["cases"][0][0] == "giantmlp288":
        return 20
    return 75 if "p" in group["hw"] else (60 if group["hw"] == "24c1g" else 40)


def case_env(group, case):
    model, provider, variant = case
    ranks = 96 if group["hw"].startswith("96") else 24
    gpus = ranks // 24
    water = model == "watercnn8k"
    profile = group["mode"] == "profile"
    steps = 12 if profile or not water else 202
    cap = 20000000 if water else 500000
    build = "scorep" if profile else "plain"
    chain_count = {"shared_c1": "1", "shared_c3": "3"}.get(variant, "0")
    cmi = MINI.parent / "CPP-ML-Interface"
    env = dict(USE_SCOREP_ENV=str(int(profile)), USE_SCOREP=str(int(profile)),
               SCOREP_MPP_ENV="mpi", SKIP_COMPILE_ENV="1",
               COMPILE_OUTPUT_PATH_ENV=f"build_gpu_refresh_{build}",
               SOLVER_BIN=str(MINI / "solver_cpp" / f"build_gpu_refresh_{build}" / "terrain_solver"),
               PHYDLL_DL_BUILD_DIR=str(cmi / "dl_clients" / f"build-gpu-refresh-{build}"),
               PHYDLL_DL_CLIENT=str(cmi / "dl_clients" / f"build-gpu-refresh-{build}" / "phydll_dl_client"),
               AIX_INSTALL_PREFIX_ENV=str(cmi / "extern/AIxeleratorService" / ("INSTALL-SCOREP" if profile else "INSTALL")),
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
               NIC_SAMPLING_ENV="0", GPU_MEM_SAMPLING_ENV="0", AIX_P2P_TIMELINE_ENV="0",
               AIX_COMMUNICATION_MODE_ENV="collective", DB_NODES_ENV="1", DB_LAYOUT_ENV="shared",
               SMARTSIM_THREADS_PER_QUEUE_ENV="1", SMARTSIM_THREADS_PER_QUEUE="1",
                SMARTSIM_INTRA_OP_THREADS="8", SMARTSIM_INTER_OP_THREADS="1", SMARTSIM_MPI_SEQUENTIAL_PUT=chain_count,
               ML_DB_BATCH_SIZE_ENV=str(cap), PHYDLL_PIN_ENV="1", PHYDLL_PIN_LOG_ENV="1",
               PHYDLL_PIN_CPU_BUDGET_ENV=str(ranks), PHYDLL_DL_NODES_ENV="1",
               PHYDLL_NP_DL_ENV=str(gpus), PHYDLL_NP_DL=str(gpus), PHYDLL_DL_COUNT="1", PHYDLL_DL_FIELD_COUNT="1",
               PHYDLL_REBUILD_DL_CLIENT_ENV="0", PHYDLL_REBUILD_ENV="0", USE_PYTHON_DL_CLIENT="0",
               MLCOUPLING_INTRA_OP_THREADS="1", MLCOUPLING_INTER_OP_THREADS="1", OMP_NUM_THREADS="1",
               MLCOUPLING_MEM_LOG_VERBOSE="0", MLCOUPLING_MEM_LOG_COLLECTIVE="0", MLCOUPLING_STEP_TIMING_SYNC="1",
               SCOREP_ENABLE_TRACING_ENV="false", SCOREP_ENABLE_TRACING="false",
               SCOREP_ENABLE_PROFILING_ENV=str(profile).lower(), SCOREP_ENABLE_PROFILING=str(profile).lower(),
               SCOREP_SHARED_PROFILE_ENV="1", SKIP_PAPI_METRICS="1", SCOREP_METRIC_PAPI="",
               SCOREP_DIR_TAG_ENV="gpu_refresh", OVERWRITE_JOB_NAME_ENV="1", APPLY_CUDA_STUBS_ENV="0")
    env["CPP_ML_CONFIG_ENV"] = (f"config_phydll_gpu_refresh_{variant}.toml" if provider == "PHYDLL"
                                 else "config_aix.toml" if provider == "AIX" else "config.toml")
    return env


def ready(build):
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
    subprocess.run(["sha256sum", "--check", str(provenance / "source.sha256")], check=True, stdout=subprocess.DEVNULL)
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
    return marker


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def submit(mode, campaign, selected_groups=None, notifier="swatch"):
    selected_groups = groups() if selected_groups is None else selected_groups
    markers = {b: ready(b) for b in ("plain", "scorep")}
    root = MINI / "results/gpu_refresh" / campaign
    if root.exists():
        raise RuntimeError("Campaign already exists; use a fresh token")
    if notifier not in ("swatch", "snotify"):
        raise ValueError(f"Unknown notifier: {notifier}")
    watcher = os.environ.get(f"GPU_REFRESH_{notifier.upper()}", str(Path.home() / "scripts" / notifier))
    if mode == "--submit" and not os.access(watcher, os.X_OK):
        raise RuntimeError("Mandatory swatch executable missing")
    root.mkdir(parents=True)
    (root / "ready.json").write_text(json.dumps(markers, indent=2) + "\n")
    group_rows, case_rows, ids = [], [], []
    for group in selected_groups:
        cmd = ["sbatch", "--parsable", f"--time=00:{minutes(group):02d}:00" if minutes(group) < 60 else f"--time=01:{minutes(group)-60:02d}:00",
               f"--chdir={MINI}", f"--output={root}/{group['id']}_%j.out", f"--error={root}/{group['id']}_%j.err",
               f"--export=ALL,GPU_REFRESH_MINI_ENV={MINI},GPU_REFRESH_CAMPAIGN_ENV={campaign},GPU_REFRESH_BUILD_JOB_ID_ENV={os.environ['GPU_REFRESH_BUILD_JOB_ID_ENV']}",
               str(MINI / f"gpu_refresh_{group['hw']}.sbatch"), group["id"]]
        if "p" in group["hw"]:
            cmd[-2:-2] = [":", cmd[2]]
        print(shlex.join(cmd), flush=True)
        jid = "DRY_RUN"
        if mode == "--submit":
            reply = subprocess.check_output(cmd, text=True).strip()
            if not re.fullmatch(r"[0-9]+(?:;[A-Za-z0-9_.-]+)?", reply):
                raise RuntimeError(f"Unexpected sbatch reply: {reply}")
            jid = reply.split(";")[0]
            ids.append(jid)
        group_rows.append(dict(group=group["id"], mode=group["mode"], hardware=group["hw"], repeat=group["repeat"], minutes=minutes(group), job_id=jid))
        for index, case in enumerate(group["cases"], 1):
            env = case_env(group, case)
            case_rows.append(dict(group=group["id"], order=index, model=case[0], provider=case[1], variant=case[2], job_id=jid,
                                  steps=env["TOTAL_STEPS_ENV"], cap=env["ML_BATCH_SIZE_ENV"], warm=1,
                                  steady=int(env["TOTAL_STEPS_ENV"]) // 2 - 1))
        write_csv(root / "groups.csv", group_rows, list(group_rows[0]))
        write_csv(root / "cases.csv", case_rows, list(case_rows[0]))
    if ids:
        notify_args = (["--jobid"] if notifier == "swatch" else
                       ["-N", "GPU Refresh Profiles", "-m", f"{campaign}: {len(selected_groups)} allocations"])
        subprocess.run([watcher, *notify_args, *ids], check=True)
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
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job):
        raise RuntimeError("Slurm allocation required")
    ranks = 96 if group["hw"].startswith("96") else 24
    if "p" in group["hw"]:
        if os.environ.get("SLURM_HET_SIZE") != "2" or any(os.environ.get(f"SLURM_NTASKS_HET_GROUP_{i}") != str(ranks) or os.environ.get(f"SLURM_JOB_NUM_NODES_HET_GROUP_{i}") != "1" for i in (0, 1)):
            raise RuntimeError("Het allocation mismatch")
    elif os.environ.get("SLURM_NTASKS") != str(ranks) or os.environ.get("SLURM_JOB_NUM_NODES") != "1":
        raise RuntimeError("Single allocation mismatch")
    build = "scorep" if group["mode"] == "profile" else "plain"
    ready(build)
    root = MINI / "results/gpu_refresh" / campaign / f"{group['id']}_{job}"
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    failed = False
    for index, case in enumerate(group["cases"], 1):
        model, provider, variant = case
        run_id = f"gpu_refresh_{campaign}_{group['id']}_{job}_{index}_{model}_{provider.lower()}_{variant}"
        out = root / run_id
        out.mkdir()
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(("AIX_P2P_", "SMARTSIM_DB_", "SCOREP_")) or key == "SSDB":
                del env[key]
        env.update(case_env(group, case))
        env.update(RUN_ID_ENV=run_id, JOB_NAME_ENV=run_id, SMARTSIM_PORT_OFFSET_ENV=str(index * 31))
        env.update(GPU_REFRESH_OUTPUT_ROOT=str(root), GPU_REFRESH_BUILD_ENV=str(MINI / "solver_cpp" / f"build_gpu_refresh_{build}" / "build.env"))
        config = (MINI / env["CPP_ML_CONFIG_ENV"]).read_text()
        if provider == "PHYDLL":
            config = config.replace("batch_size = 500000", f"batch_size = {env['ML_BATCH_SIZE_ENV']}")
        (out / "config.toml").write_text(config)
        env["CPP_ML_CONFIG_ENV"] = str(out / "config.toml")
        (out / "runtime.env.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
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
            rows.append(dict(run_id=run_id, model=model, provider=provider, variant=variant, rc=rc, cleanup_safe=safe))
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
        if not safe:
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
        authorized = groups() + profile_retry_groups() + [profile_recovery_group()]
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
