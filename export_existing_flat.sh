#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/train_models/model_a"
PYTHON_BIN="python3"
DRY_RUN=0

MODELS=(perfect_model transformer_mlp watercnn benchmark_giant_mlp)
BACKENDS=(torch onnx tf)

usage() {
  cat <<'EOF'
Usage:
  ./export_existing_flat.sh [options]

Re-export existing model checkpoints/artifacts in flat_contiguous I/O layout
without retraining (uses train.py --export-only).

Options:
  --models "m1 m2 ..."       Space-separated model IDs
                              (default: perfect_model transformer_mlp watercnn benchmark_giant_mlp)
  --backends "b1 b2 ..."     Space-separated backends from: torch onnx tf
                              (default: "torch onnx tf")
  --python <exe>              Python executable (default: python3)
  --dry-run                   Print commands only
  -h, --help                  Show this help

Example:
  ./export_existing_flat.sh --models "transformer_mlp watercnn"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --models"; exit 1; }
      MODELS=(${=1})
      ;;
    --backends)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --backends"; exit 1; }
      BACKENDS=(${=1})
      ;;
    --python)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --python"; exit 1; }
      PYTHON_BIN="$1"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ ! -f "${MODEL_DIR}/train.py" ]]; then
  echo "Error: train.py not found at ${MODEL_DIR}"
  exit 1
fi

for backend in "${BACKENDS[@]}"; do
  case "$backend" in
    torch|onnx|tf) ;;
    *)
      echo "Error: invalid backend '$backend'. Allowed: torch onnx tf"
      exit 1
      ;;
  esac
done

echo "Export directory: ${MODEL_DIR}"
echo "Models: ${MODELS[*]}"
echo "Backends: ${BACKENDS[*]}"
echo "Layout: flat_contiguous"

cd "${MODEL_DIR}"

for model in "${MODELS[@]}"; do
  output_ckpt="best_model_${model}.pt"
  out_torch="best_model_jit_${model}_flat.pt"
  out_onnx="best_model_${model}_flat.onnx"
  out_tf="best_model_tf_${model}_flat.pb"
  out_manifest="artifact_manifest_${model}_flat.json"

  cmd=(
    "${PYTHON_BIN}" train.py
    --model "${model}"
    --export-only
    --output "${output_ckpt}"
    --export-backends "${BACKENDS[@]}"
    --export-io-layout flat_contiguous
    --inference-output "${out_torch}"
    --onnx-output "${out_onnx}"
    --tf-output "${out_tf}"
    --artifact-manifest "${out_manifest}"
  )

  echo
  echo "==> Exporting ${model}"
  echo "Command: ${cmd[*]}"

  if [[ ${DRY_RUN} -eq 0 ]]; then
    "${cmd[@]}"
  fi
done

echo
echo "Done. Flat manifests created in ${MODEL_DIR}:"
for model in "${MODELS[@]}"; do
  echo "  artifact_manifest_${model}_flat.json"
done
