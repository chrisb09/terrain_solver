#!/bin/bash
echo "Checking all available FFmpeg modules for drawtext support..."
module spider FFmpeg 2>&1 | grep -oP 'FFmpeg/[\d.]+' | sort -V | while read mod; do
  echo "Testing $mod..."
  module load $mod 2>/dev/null
  if ffmpeg -hide_banner -filters 2>&1 | grep -q drawtext; then
    echo "✓ drawtext available in $mod"
  else
    echo "✗ drawtext NOT available in $mod"
  fi
  module unload $mod 2>/dev/null
done