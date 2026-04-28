#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Quick SmartSim/SmartRedis smoke test for Torch, ONNX, or TensorFlow model artifacts."
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Direct path to a model artifact. If omitted, resolve from --artifact-manifest + --model-id + --backend.",
    )
    p.add_argument(
        "--artifact-manifest",
        type=Path,
        default=Path("train_models/model_a/artifact_manifest_transformer_mlp.json"),
        help="Artifact manifest written by train.py.",
    )
    p.add_argument(
        "--model-id",
        default="transformer_mlp",
        help="Model name to resolve from the artifact manifest when --model is not given.",
    )
    p.add_argument(
        "--model-name",
        default="smoke_model",
        help="Model name to register in SmartRedis",
    )
    p.add_argument(
        "--device",
        choices=["CPU", "GPU", "cpu", "gpu"],
        default="CPU",
        help="SmartRedis model device",
    )
    p.add_argument(
        "--num-devices",
        type=int,
        default=1,
        help="Number of devices to use for model parallelism (default: 1)",
    )
    p.add_argument(
        "--backend",
        default="TORCH",
        help="Model backend for SmartRedis. Options: TF, TFLITE, TORCH, ONNX",
    )
    p.add_argument(
        "--model-io-layout",
        choices=["auto", "split_3x3", "flat_contiguous"],
        default="auto",
        help=(
            "Model input layout used for test tensors and run_model inputs. "
            "'auto' resolves from manifest io_layout (or falls back to split_3x3)."
        ),
    )
    p.add_argument(
        "--tf-inputs",
        nargs="*",
        default=None,
        help="Optional TensorFlow input node names. Defaults to the manifest values if present.",
    )
    p.add_argument(
        "--tf-outputs",
        nargs="*",
        default=None,
        help="Optional TensorFlow output node names. Defaults to the manifest values if present.",
    )
    p.add_argument(
        "--input-dim",
        type=int,
        default=10,
        help="Number of sample inputs to run through the model.",
    )
    p.add_argument(
        "--client-batch-size",
        type=int,
        default=1,
        help="Number of inputs to send in each client.run_model call (default: 1)",
    )
    p.add_argument(
        "--clients",
        type=int,
        default=1,
        help="Number of client threads to use (default: 1)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat the model run (default: 1)",
    )
    p.add_argument(
        "--server-side-batch-size",
        type=int,
        default=0,
        help="Optional server-side batch size for model execution in smartsim",
    )
    p.add_argument(
        "--server-side-min-batch-size",
        type=int,
        default=0,
        help="Optional server-side minimum batch size for model execution in smartsim",
    )
    p.add_argument(
        "--server-side-min-batch-timeout",
        type=int,
        default=0,
        help="Optional server-side batch timeout (ms) for model execution in smartsim",
    )
    p.add_argument(
        "--port",
        type=int,
        default=6789,
        help="Local DB port for the temporary SmartSim database",
    )
    p.add_argument(
        "--startup-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for DB startup",
    )
    p.add_argument(
        "--model-timeout-ms",
        type=int,
        default=None,
        help="Optional SmartRedis model load timeout in milliseconds (maps to SR_MODEL_TIMEOUT).",
    )
    p.add_argument(
        "--command-timeout-ms",
        type=int,
        default=None,
        help="Optional SmartRedis command timeout in milliseconds (maps to SR_CMD_TIMEOUT).",
    )
    p.add_argument(
        "--socket-timeout-ms",
        type=int,
        default=None,
        help="Optional SmartRedis socket timeout in milliseconds (maps to SR_SOCKET_TIMEOUT).",
    )
    p.add_argument(
        "--skip-backend-preflight",
        action="store_true",
        help="Skip backend shared-library preflight checks.",
    )
    p.add_argument(
        "--keep-exp-dir",
        action="store_true",
        help="Keep SmartSim experiment directory for debugging",
    )
    return p.parse_args()


def _artifact_manifest_was_explicitly_set(argv: list[str]) -> bool:
    return any(
        arg == "--artifact-manifest" or arg.startswith("--artifact-manifest=")
        for arg in argv
    )


def resolve_manifest_path(args) -> Path:
    manifest_path = args.artifact_manifest

    if args.model_io_layout == "flat_contiguous":
        flat_candidate = manifest_path.with_name(f"artifact_manifest_{args.model_id}_flat.json")
        if flat_candidate.exists():
            if flat_candidate != manifest_path:
                print(
                    f"Using flat artifact manifest for model_id='{args.model_id}': {flat_candidate}",
                    flush=True,
                )
            return flat_candidate.resolve()

    if _artifact_manifest_was_explicitly_set(sys.argv[1:]):
        return manifest_path.resolve()

    candidate_name = f"artifact_manifest_{args.model_id}.json"
    candidate_path = manifest_path.with_name(candidate_name)
    if candidate_path.exists():
        if candidate_path != manifest_path:
            print(
                f"Using model-specific artifact manifest for model_id='{args.model_id}': {candidate_path}",
                flush=True,
            )
        return candidate_path.resolve()

    return manifest_path.resolve()


def load_manifest(manifest_path: Path) -> dict:
    print(f"Loading artifact manifest from {manifest_path} ...", flush=True)
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "artifacts" not in payload:
        raise ValueError(f"Artifact manifest has unexpected format: {manifest_path}")
    return payload


def resolve_artifact_from_manifest(manifest: dict, model_id: str, backend: str) -> dict:
    backend = backend.upper()
    for artifact in manifest.get("artifacts", []):
        if artifact.get("model_name") == model_id and str(artifact.get("backend", "")).upper() == backend:
            return artifact
    raise KeyError(
        f"No artifact for model_id='{model_id}' and backend='{backend}' in manifest."
    )


def normalize_name_list(values):
    if values is None:
        return None
    return [value for value in values if value]


def normalize_tf_node_names(values):
    if values is None:
        return None
    normalized = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        normalized.append(text.split(":", 1)[0])
    return normalized


def configure_smartredis_timeouts(args, model_path: Path) -> None:
    explicit_timeouts = {
        "SR_MODEL_TIMEOUT": args.model_timeout_ms,
        "SR_CMD_TIMEOUT": args.command_timeout_ms,
        "SR_SOCKET_TIMEOUT": args.socket_timeout_ms,
    }
    for env_name, value in explicit_timeouts.items():
        if value is not None:
            if value < 1:
                raise ValueError(f"{env_name} must be >= 1 ms, got {value}")
            os.environ[env_name] = str(value)

    size_bytes = model_path.stat().st_size
    size_gib = size_bytes / float(1024 ** 3)
    print(
        f"[smoke] Model artifact size: {size_bytes} bytes ({size_gib:.2f} GiB)",
        flush=True,
    )

    # Large artifacts can legitimately take several minutes to upload and load
    # into RedisAI. If the caller did not set explicit timeouts, extend them.
    if size_bytes >= 1024 ** 3:
        auto_timeouts = {
            "SR_MODEL_TIMEOUT": 15 * 60 * 1000,
            "SR_CMD_TIMEOUT": 15 * 60 * 1000,
            "SR_SOCKET_TIMEOUT": 15 * 60 * 1000,
        }
        for env_name, value in auto_timeouts.items():
            if explicit_timeouts[env_name] is None and env_name not in os.environ:
                os.environ[env_name] = str(value)
                print(
                    f"[smoke] Auto-configured {env_name}={value} for large model artifact",
                    flush=True,
                )


def resolve_redisai_onnx_cuda_provider_path() -> Optional[Path]:
    spec = importlib.util.find_spec("smartsim")
    candidates = []
    if spec is not None and spec.origin is not None:
        smartsim_dir = Path(spec.origin).parent
        candidates.append(
            smartsim_dir
            / "_core"
            / "lib"
            / "backends"
            / "redisai_onnxruntime"
            / "lib"
            / "libonnxruntime_providers_cuda.so"
        )
        candidates.append(
            Path(spec.origin).resolve().parent
            / "_core"
            / "lib"
            / "backends"
            / "redisai_onnxruntime"
            / "lib"
            / "libonnxruntime_providers_cuda.so"
        )

    script_root = Path(__file__).resolve().parent
    candidates.append(
        script_root
        / "python"
        / "smartsim_cuda-12"
        / "lib"
        / "python3.9"
        / "site-packages"
        / "smartsim"
        / "_core"
        / "lib"
        / "backends"
        / "redisai_onnxruntime"
        / "lib"
        / "libonnxruntime_providers_cuda.so"
    )

    seen = set()
    for candidate in candidates:
        text = str(candidate)
        if text in seen:
            continue
        seen.add(text)
        if candidate.exists():
            return candidate
    return None


def check_onnx_gpu_backend_dependencies() -> None:
    provider = resolve_redisai_onnx_cuda_provider_path()
    if provider is None:
        print(
            "[smoke] Warning: Could not locate the RedisAI ONNX CUDA provider library; skipping ONNX GPU preflight.",
            flush=True,
        )
        return
    print(f"[smoke] ONNX GPU preflight provider: {provider}", flush=True)

    ldd = shutil.which("ldd")
    if ldd is None:
        print(
            "[smoke] Warning: 'ldd' is unavailable; skipping ONNX GPU shared-library preflight.",
            flush=True,
        )
        return

    try:
        proc = subprocess.run(
            [ldd, str(provider)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        print(
            f"[smoke] Warning: Failed to run ldd for ONNX GPU preflight: {exc}",
            flush=True,
        )
        return

    missing = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if "=> not found" in stripped:
            missing.append(stripped.split("=>", 1)[0].strip())

    if not missing:
        print("[smoke] ONNX GPU backend preflight passed", flush=True)
        return

    suggestions = [
        "The RedisAI ONNX CUDA provider has unresolved shared-library dependencies.",
        f"Provider: {provider}",
        "Missing libraries: " + ", ".join(missing),
        "This usually means LD_LIBRARY_PATH does not expose the CUDA/cuDNN runtime expected by ONNX Runtime.",
    ]
    if any(name == "libcudnn.so.8" for name in missing):
        suggestions.append(
            "Detected missing libcudnn.so.8. A cuDNN major-version mismatch is likely if your environment only provides libcudnn.so.9."
        )
    suggestions.append(
        "ONNX on CPU may still work, but ONNX on GPU is likely to hang or fail during set_model_from_file until this is fixed."
    )
    raise RuntimeError(" ".join(suggestions))


def resolve_tf_cuda_root() -> Optional[Path]:
    env_roots = [
        os.environ.get("EBROOTCUDA"),
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_ROOT"),
    ]

    candidate_roots = [Path(root) for root in env_roots if root]
    candidate_roots.extend(
        [
            Path("/usr/local/cuda"),
            Path("/usr/local/cuda-12.4"),
        ]
    )

    seen = set()
    for root in candidate_roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        libdevice = root / "nvvm" / "libdevice" / "libdevice.10.bc"
        if libdevice.exists():
            return root
    return None


def configure_tf_xla_cuda_data_dir(backend: str, device: str) -> None:
    if backend != "TF" or device != "GPU":
        return

    cuda_root = resolve_tf_cuda_root()
    if cuda_root is None:
        print(
            "[smoke] Warning: Could not locate CUDA libdevice. TF GPU XLA may fail with JIT errors (e.g. Rsqrt).",
            flush=True,
        )
        return

    existing = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_cuda_data_dir" not in existing:
        addon = f"--xla_gpu_cuda_data_dir={cuda_root}"
        os.environ["XLA_FLAGS"] = f"{existing} {addon}".strip()

    print(
        f"[smoke] TF XLA CUDA data dir: {cuda_root}",
        flush=True,
    )


def main():
    args = parse_args()

    backend = args.backend.upper()
    if backend not in {"TORCH", "ONNX", "TF", "TFLITE"}:
        raise ValueError(f"Unsupported backend: {args.backend}")

    artifact = None
    manifest = None
    if args.model is None:
        manifest_path = resolve_manifest_path(args)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")
        manifest = load_manifest(manifest_path)
        artifact = resolve_artifact_from_manifest(manifest, args.model_id, backend)
        model_path = Path(artifact["path"]).resolve()
        manifest_io_layout = str(artifact.get("io_layout", "")).strip().lower()
        if args.model_io_layout == "auto" and manifest_io_layout in {"split_3x3", "flat_contiguous"}:
            args.model_io_layout = manifest_io_layout
    else:
        model_path = args.model.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    configure_smartredis_timeouts(args, model_path)

    device = args.device.upper()
    if device not in {"CPU", "GPU"}:
        raise ValueError(f"Unsupported device: {args.device}")

    if (
        not args.skip_backend_preflight
        and backend == "ONNX"
        and device == "GPU"
    ):
        check_onnx_gpu_backend_dependencies()

    num_devices = args.num_devices
    if num_devices < 1:
        raise ValueError(f"num-devices must be >= 1, got {num_devices}")

    tf_inputs = normalize_name_list(args.tf_inputs)
    tf_outputs = normalize_name_list(args.tf_outputs)
    if args.model_io_layout == "auto":
        artifact_io_layout = "split_3x3"
        if artifact is not None:
            manifest_layout = str(artifact.get("io_layout", "")).strip().lower()
            if manifest_layout in {"split_3x3", "flat_contiguous"}:
                artifact_io_layout = manifest_layout
            else:
                manifest_inputs = artifact.get("inputs") or []
                artifact_io_layout = "flat_contiguous" if len(manifest_inputs) == 1 else "split_3x3"
    else:
        artifact_io_layout = args.model_io_layout
        if artifact is not None:
            manifest_layout = str(artifact.get("io_layout", "")).strip().lower()
            if manifest_layout in {"split_3x3", "flat_contiguous"} and manifest_layout != artifact_io_layout:
                print(
                    f"[smoke] Warning: --model-io-layout={artifact_io_layout} overrides manifest io_layout={manifest_layout}",
                    flush=True,
                )

    if artifact is not None and backend == "TF":
        if tf_inputs is None:
            tf_inputs = artifact.get("inputs")
        if tf_outputs is None:
            tf_outputs = artifact.get("outputs")
        tf_inputs = normalize_tf_node_names(tf_inputs)
        tf_outputs = normalize_tf_node_names(tf_outputs)
        if not tf_inputs or not tf_outputs:
            raise ValueError(
                "TensorFlow backend requires input and output node names. "
                "Provide them via the manifest or --tf-inputs/--tf-outputs."
            )

    if device == "GPU":
        print(f"Using device: {device} with num_devices={num_devices}")

    configure_tf_xla_cuda_data_dir(backend, device)

    from smartsim.experiment import Experiment
    from smartredis import Client

    os.environ["SR_DB_TYPE"] = "Standalone"

    exp_name = f"smartsim_smoke_{int(time.time())}"
    exp_dir = Path("smartsim_experiments") / exp_name
    exp = Experiment(name=str(exp_dir), launcher="local")

    db = exp.create_database(
        port=args.port,
        interface="lo",
        batch=False,
        single_cmd=True,
        db_nodes=1,
    )

    print(f"[smoke] Starting local SmartSim DB on port {args.port} ...", flush=True)
    exp.start(db, block=False, summary=False)

    try:
        start = time.time()
        address = None
        while time.time() - start < args.startup_timeout:
            try:
                address_list = db.get_address()
                if address_list and address_list[0]:
                    address = address_list[0]
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not address:
            raise TimeoutError(
                f"Timed out waiting for DB address after {args.startup_timeout:.1f}s"
            )

        os.environ["SSDB"] = address
        print(f"[smoke] DB address: {address}", flush=True)

        if args.clients < 1:
            raise ValueError(f"Number of clients must be >= 1, got {args.clients}")

        clients = []
        for i in range(args.clients):
            clients.append(Client(cluster=False, logger_name=f"smoke_client_{i}"))

        client = clients[0]

        print(
            f"[smoke] Loading model '{args.model_name}' from {model_path} with backend={backend} on device={device} ...",
            flush=True,
        )
        print(f"[smoke] Artifact model io_layout={artifact_io_layout}", flush=True)
        load_kwargs = {}
        if backend == "TF":
            load_kwargs["inputs"] = tf_inputs
            load_kwargs["outputs"] = tf_outputs
            print(f"[smoke] TensorFlow inputs={tf_inputs} outputs={tf_outputs}", flush=True)
        start_time = time.time()
        if device == "GPU" and num_devices > 1 and backend != "TF":
            client.set_model_from_file_multigpu(
                args.model_name,
                str(model_path),
                backend,
                first_gpu=0,
                num_gpus=num_devices,
                batch_size=args.server_side_batch_size,
                min_batch_size=args.server_side_min_batch_size,
                min_batch_timeout=args.server_side_min_batch_timeout,
                **load_kwargs,
            )
        else:
            client.set_model_from_file(
                args.model_name,
                str(model_path),
                backend,
                device,
                **load_kwargs,
            )
        print(f"[smoke] Model loaded in {time.time() - start_time:.2f} seconds", flush=True)
        input_dim = args.input_dim
        x_water = np.zeros((input_dim, 1, 3, 3), dtype=np.float32)
        x_terrain = np.zeros((input_dim, 1, 3, 3), dtype=np.float32)
        x_packed = np.zeros((input_dim, 18), dtype=np.float32)

        client_bs = int(math.ceil(input_dim / len(clients)))
        num_batches = int(math.ceil(input_dim / client_bs))
        
        # Overwrite for testing
        client_bs = input_dim
        num_batches = 1

        for i in range(input_dim):
            x_water[i:i + client_bs, :, :, :] = i % 100
            x_terrain[i:i + client_bs, :, :, :] = (i * 10) % 255
            x_packed[i, :9] = x_water[i, 0, :, :].reshape(-1)
            x_packed[i, 9:] = x_terrain[i, 0, :, :].reshape(-1)
        for i in range(num_batches):
            if artifact_io_layout == "flat_contiguous":
                client.put_tensor(f"smoke_x_packed_{i}", x_packed[i:i + client_bs, :])
            else:
                client.put_tensor(f"smoke_x_water_{i}", x_water[i:i + client_bs, :, :, :])
                client.put_tensor(f"smoke_x_terrain_{i}", x_terrain[i:i + client_bs, :, :, :])

        worker_count = len(clients)
        total_model_calls = num_batches * args.repeats
        print(
            f"[smoke] Running model {args.repeats} times ... for {input_dim} inputs in batches of {client_bs} on {worker_count} clients",
            flush=True,
        )

        worker_batches = [list(range(worker_id, num_batches, worker_count)) for worker_id in range(worker_count)]
        worker_errors = []
        worker_errors_lock = threading.Lock()

        def _worker_run(worker_id, worker_client):
            batches = worker_batches[worker_id]
            try:
                for _ in range(args.repeats):
                    for batch_id in batches:
                        if device == "GPU" and num_devices > 1 and backend != "TF":
                            worker_client.run_model_multigpu(
                                name=args.model_name,
                                offset=batch_id % num_devices,
                                first_gpu=0,
                                num_gpus=num_devices,
                                inputs=[f"smoke_x_packed_{batch_id}"] if artifact_io_layout == "flat_contiguous" else [f"smoke_x_water_{batch_id}", f"smoke_x_terrain_{batch_id}"],
                                outputs=[f"smoke_y_{batch_id}"],
                            )
                        else:
                            worker_client.run_model(
                                name=args.model_name,
                                inputs=[f"smoke_x_packed_{batch_id}"] if artifact_io_layout == "flat_contiguous" else [f"smoke_x_water_{batch_id}", f"smoke_x_terrain_{batch_id}"],
                                outputs=[f"smoke_y_{batch_id}"],
                            )
            except Exception as exc:
                with worker_errors_lock:
                    worker_errors.append((worker_id, exc))

        threads = []
        start_time = time.time()
        for worker_id, worker_client in enumerate(clients):
            thread = threading.Thread(target=_worker_run, args=(worker_id, worker_client), daemon=False)
            threads.append(thread)
            thread.start()
        launch_done = time.time()

        print(
            f"Started {len(threads)} worker threads in {launch_done - start_time:.2f} seconds. "
            f"Waiting for {total_model_calls} model runs to complete ...",
            flush=True,
        )
        for thread in threads:
            thread.join()

        if worker_errors:
            details = "; ".join(
                [f"worker={worker_id} error={exc}" for worker_id, exc in worker_errors]
            )
            raise RuntimeError(f"One or more worker threads failed: {details}")

        end_time = time.time()
        print(f"[smoke] Model run completed in {end_time - start_time:.2f} seconds", flush=True)

        y = np.zeros(input_dim, dtype=np.float32)
        c = 0
        for i in range(num_batches):
            fetched = client.get_tensor(f"smoke_y_{i}")
            for val in np.asarray(fetched).reshape(-1):
                y[c] = val
                c += 1
        print(f"[smoke] Success. Output shape={tuple(y.shape)}, dtype={y.dtype}", flush=True)
        # If y is large, print the first few and last few values to avoid flooding the console
        if len(y) > 10:
            print(f"[smoke] Output values={y[:5]} ... {y[-5:]}", flush=True)
        else:
            print(f"[smoke] Output values={y}", flush=True)
        print(f"[smoke] Output stats: len={len(y)} min={y.min()} max={y.max()} mean={y.mean():.4f}", flush=True)

    finally:
        print("[smoke] Stopping SmartSim DB ...", flush=True)
        try:
            exp.stop(db)
        except Exception as exc:
            print(f"[smoke] Warning: failed to stop DB cleanly: {exc}", file=sys.stderr, flush=True)

        if not args.keep_exp_dir:
            try:
                shutil.rmtree(exp_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
