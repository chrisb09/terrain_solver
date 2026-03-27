#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  make_dual_view_video.sh --top-pattern GLOB --slice-pattern GLOB --output FILE [options]

Required:
  --top-pattern GLOB      Glob for top-view frames (e.g. 'rendered_frames/frame_*.png')
  --slice-pattern GLOB    Glob for slice-view frames (e.g. 'rendered_slices/frame_*.png')
  --output FILE           Output video file (e.g. combined.mp4)

Options:
  --fps N                 Framerate (default: 60)
  --width N               Output width (default: 2160)
  --height N              Legacy total output height for symmetric split (default: 2160)
  --top-height N          Height of top view after scaling (overrides --height split)
  --bottom-height N       Height of slice view after scaling (overrides --height split)
  --codec NAME            Video codec (default: libx265)
  --pix-fmt NAME          Output pixel format (default: yuv420p)
  --crf N                 Constant Rate Factor (default: 23)
  --preset NAME           Encoder preset (default: slow)
  --threads N             Encoder threads (default: 16)

Example:
  ./make_dual_view_video.sh \
    --top-pattern 'rendered_frames/frame_*.png' \
    --slice-pattern 'rendered_slice_frames/frame_*.png' \
    --output dual_view.mp4 \
    --fps 60 --steps-per-frame 100 --threads 16
EOF
}

TOP_PATTERN=""
SLICE_PATTERN=""
OUTPUT=""
FPS=60
WIDTH=2160
HEIGHT=2160
TOP_HEIGHT=""
BOTTOM_HEIGHT=""
CODEC="libx265"
PIX_FMT="yuv420p"
CRF=23
PRESET="slow"
THREADS=16

while [[ $# -gt 0 ]]; do
  case "$1" in
    --top-pattern) TOP_PATTERN="${2:-}"; shift 2 ;;
    --slice-pattern) SLICE_PATTERN="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --fps) FPS="${2:-}"; shift 2 ;;
    --width) WIDTH="${2:-}"; shift 2 ;;
    --height) HEIGHT="${2:-}"; shift 2 ;;
    --top-height) TOP_HEIGHT="${2:-}"; shift 2 ;;
    --bottom-height) BOTTOM_HEIGHT="${2:-}"; shift 2 ;;
    --codec) CODEC="${2:-}"; shift 2 ;;
    --pix-fmt) PIX_FMT="${2:-}"; shift 2 ;;
    --crf) CRF="${2:-}"; shift 2 ;;
    --preset) PRESET="${2:-}"; shift 2 ;;
    --threads) THREADS="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$TOP_PATTERN" || -z "$SLICE_PATTERN" || -z "$OUTPUT" ]]; then
  echo "ERROR: --top-pattern, --slice-pattern and --output are required." >&2
  usage
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found in PATH." >&2
  exit 1
fi

if [[ -n "${TOP_HEIGHT}" || -n "${BOTTOM_HEIGHT}" ]]; then
  if [[ -z "${TOP_HEIGHT}" || -z "${BOTTOM_HEIGHT}" ]]; then
    echo "ERROR: when using custom split, provide both --top-height and --bottom-height." >&2
    exit 1
  fi
  TOP_H="${TOP_HEIGHT}"
  BOTTOM_H="${BOTTOM_HEIGHT}"
else
  TOP_H=$(( HEIGHT / 2 ))
  BOTTOM_H=$(( HEIGHT - TOP_H ))
fi

OUT_HEIGHT=$(( TOP_H + BOTTOM_H ))
ENCODE_WIDTH="${WIDTH}"
ENCODE_HEIGHT="${OUT_HEIGHT}"

FILTER_COMPLEX="[0:v]scale=${WIDTH}:${TOP_H}:flags=lanczos[top];\
[1:v]scale=${WIDTH}:${BOTTOM_H}:flags=lanczos[bottom];\
[top][bottom]vstack=inputs=2[stack]"

MAP_LABEL="[stack]"

# yuv420p (and many encoders) require even output dimensions.
if [[ "${PIX_FMT}" == "yuv420p" ]]; then
  PAD_W=$(( ENCODE_WIDTH % 2 ))
  PAD_H=$(( ENCODE_HEIGHT % 2 ))
  if [[ "${PAD_W}" -ne 0 || "${PAD_H}" -ne 0 ]]; then
    NEW_W=$(( ENCODE_WIDTH + PAD_W ))
    NEW_H=$(( ENCODE_HEIGHT + PAD_H ))
    FILTER_COMPLEX="${FILTER_COMPLEX};${MAP_LABEL}pad=${NEW_W}:${NEW_H}:0:0:color=black[enc]"
    MAP_LABEL="[enc]"
    ENCODE_WIDTH="${NEW_W}"
    ENCODE_HEIGHT="${NEW_H}"
    echo "INFO: padded output to even size ${ENCODE_WIDTH}x${ENCODE_HEIGHT} for pix_fmt=${PIX_FMT}."
  fi
fi

echo "Building dual-view video:"
echo "  top-pattern:   ${TOP_PATTERN}"
echo "  slice-pattern: ${SLICE_PATTERN}"
echo "  output:        ${OUTPUT}"
echo "  fps:           ${FPS}"
echo "  size:          ${WIDTH}x${OUT_HEIGHT} (top=${TOP_H}, bottom=${BOTTOM_H})"
echo "  encoded size:  ${ENCODE_WIDTH}x${ENCODE_HEIGHT} (pix_fmt=${PIX_FMT})"

ffmpeg -y \
  -framerate "${FPS}" -pattern_type glob -i "${TOP_PATTERN}" \
  -framerate "${FPS}" -pattern_type glob -i "${SLICE_PATTERN}" \
  -filter_complex "${FILTER_COMPLEX}" \
  -map "${MAP_LABEL}" \
  -c:v "${CODEC}" -crf "${CRF}" -preset "${PRESET}" -threads "${THREADS}" \
  -pix_fmt "${PIX_FMT}" \
  -shortest \
  "${OUTPUT}"
