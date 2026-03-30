#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path

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
        "--keep-exp-dir",
        action="store_true",
        help="Keep SmartSim experiment directory for debugging",
    )
    return p.parse_args()


def load_manifest(manifest_path: Path) -> dict:
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


def main():
    args = parse_args()

    backend = args.backend.upper()
    if backend not in {"TORCH", "ONNX", "TF", "TFLITE"}:
        raise ValueError(f"Unsupported backend: {args.backend}")

    artifact = None
    manifest = None
    if args.model is None:
        manifest_path = args.artifact_manifest.resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")
        manifest = load_manifest(manifest_path)
        artifact = resolve_artifact_from_manifest(manifest, args.model_id, backend)
        model_path = Path(artifact["path"]).resolve()
    else:
        model_path = args.model.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = args.device.upper()
    if device not in {"CPU", "GPU"}:
        raise ValueError(f"Unsupported device: {args.device}")

    num_devices = args.num_devices
    if num_devices < 1:
        raise ValueError(f"num-devices must be >= 1, got {num_devices}")

    tf_inputs = normalize_name_list(args.tf_inputs)
    tf_outputs = normalize_name_list(args.tf_outputs)
    if artifact is not None and backend == "TF":
        if tf_inputs is None:
            tf_inputs = artifact.get("inputs")
        if tf_outputs is None:
            tf_outputs = artifact.get("outputs")
        if not tf_inputs or not tf_outputs:
            raise ValueError(
                "TensorFlow backend requires input and output node names. "
                "Provide them via the manifest or --tf-inputs/--tf-outputs."
            )

    if device == "GPU":
        print(f"Using device: {device} with num_devices={num_devices}")

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

        client = Client(cluster=False)

        print(
            f"[smoke] Loading model '{args.model_name}' from {model_path} with backend={backend} on device={device} ...",
            flush=True,
        )
        load_kwargs = {}
        if backend == "TF":
            load_kwargs["inputs"] = tf_inputs
            load_kwargs["outputs"] = tf_outputs
            print(f"[smoke] TensorFlow inputs={tf_inputs} outputs={tf_outputs}", flush=True)

        if device == "GPU" and num_devices > 1 and backend != "TF":
            client.set_model_from_file_multigpu(
                args.model_name,
                str(model_path),
                backend,
                0,
                num_devices,
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

        input_dim = args.input_dim
        x_water = np.zeros((input_dim, 1, 3, 3), dtype=np.float32)
        x_terrain = np.zeros((input_dim, 1, 3, 3), dtype=np.float32)

        for i in range(int(math.ceil(input_dim / args.client_batch_size))):
            x_water[i:i+args.client_batch_size, :, :, :] = i % 100
            x_terrain[i:i+args.client_batch_size, :, :, :] = (i * 10) % 255
            client.put_tensor(f"smoke_x_water_{i}", x_water[i:i+args.client_batch_size, :, :, :])
            client.put_tensor(f"smoke_x_terrain_{i}", x_terrain[i:i+args.client_batch_size, :, :, :])

        print(f"[smoke] Running model once ... for {input_dim} inputs in batches of {args.client_batch_size}", flush=True)
        threads = []
        start_time = time.time()
        for i in range(int(math.ceil(input_dim / args.client_batch_size))):
            if device == "GPU" and num_devices > 1 and backend != "TF":
                t = threading.Thread(
                    target=client.run_model_multigpu,
                    kwargs={
                        "name": args.model_name,
                        "offset": i % num_devices,
                        "first_gpu": 0,
                        "num_gpus": num_devices,
                        "inputs": [f"smoke_x_water_{i}", f"smoke_x_terrain_{i}"],
                        "outputs": [f"smoke_y_{i}"],
                    }
                )
                threads.append(t)
                t.start()
            else:
                t = threading.Thread(
                    target=client.run_model,
                    kwargs={
                        "name": args.model_name,
                        "inputs": [f"smoke_x_water_{i}", f"smoke_x_terrain_{i}"],
                        "outputs": [f"smoke_y_{i}"],
                    }
                )
                threads.append(t)
                t.start()
        print(f"Waiting for {len(threads)} model runs to complete ...", flush=True)
        for t in threads:
            t.join()
        end_time = time.time()
        print(f"[smoke] Model run completed in {end_time - start_time:.2f} seconds", flush=True)

        y = np.zeros(input_dim, dtype=np.float32)
        c = 0
        for i in range(int(math.ceil(input_dim / args.client_batch_size))):
            fetched = client.get_tensor(f"smoke_y_{i}")
            for val in np.asarray(fetched).reshape(-1):
                y[c] = val
                c += 1
        print(f"[smoke] Success. Output shape={tuple(y.shape)}, dtype={y.dtype}", flush=True)
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
