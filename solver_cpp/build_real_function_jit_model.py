#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


TRAIN_MODEL_DIR = Path(__file__).resolve().parents[1] / "train_models" / "model_a"
sys.path.insert(0, str(TRAIN_MODEL_DIR))

from train import (  # noqa: E402
    RealFunctionModel,
    build_artifact_record,
    ensure_parent_dir,
    export_inference_model,
    export_onnx_model,
    export_tensorflow_frozen_model,
    write_artifact_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build export artifacts for the exact perfect-model update rule."
    )
    parser.add_argument(
        "--torch-output",
        type=Path,
        default=Path("train_models/model_a/real_function_jit.pt"),
        help="Output path for TorchScript model",
    )
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=Path("train_models/model_a/real_function.onnx"),
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--tf-output",
        type=Path,
        default=Path("train_models/model_a/real_function_tf.pb"),
        help="Output path for TensorFlow frozen graph",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=Path("train_models/model_a/artifact_manifest_perfect_model.json"),
        help="Output path for the artifact manifest",
    )
    parser.add_argument(
        "--export-backends",
        nargs="+",
        choices=["torch", "onnx", "tf"],
        default=["torch"],
        help="Backends to export",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device used during TorchScript export and self-check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = torch.device(args.device)
    model = RealFunctionModel().to(device).eval()
    artifacts = []

    if "torch" in args.export_backends:
        ensure_parent_dir(str(args.torch_output))
        export_inference_model(model, str(args.torch_output), device)
        artifacts.append(build_artifact_record("perfect_model", "TORCH", str(args.torch_output)))

    if "onnx" in args.export_backends:
        artifact = export_onnx_model(model, str(args.onnx_output))
        artifact["model_name"] = "perfect_model"
        artifacts.append(artifact)

    if "tf" in args.export_backends:
        artifact = export_tensorflow_frozen_model(model, str(args.tf_output))
        artifact["model_name"] = "perfect_model"
        artifacts.append(artifact)

    write_artifact_manifest(str(args.artifact_manifest), "perfect_model", artifacts)

    x_water = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=device)
    x_terrain = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(x_water, x_terrain)

    for artifact in artifacts:
        print(f"Saved {artifact['backend']} model to: {artifact['path']}")
    print(f"Saved artifact manifest to: {args.artifact_manifest}")
    print(f"Self-check output on zero input: {y.detach().cpu().numpy()}")


if __name__ == "__main__":
    main()
