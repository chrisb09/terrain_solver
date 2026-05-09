#!/bin/sh

# pass this to cmake to enable the C++ ML interface in the mini-app
USE_CPP_ML_INTERFACE=ON

# Build the terrain solver

current_dir=$(pwd)

custom_build_dir="${1:-build}"
if [ -n "$custom_build_dir" ]; then
  echo "Using custom build directory: $custom_build_dir"
else
  echo "Using default build directory: build"
fi

# delete the old build directory if it exists to ensure a clean build
if [ -d "solver_cpp/${custom_build_dir}" ]; then
  echo "Removing old build directory: solver_cpp/${custom_build_dir}"
  rm -rf "solver_cpp/${custom_build_dir}"
fi

# create build directory if it doesn't exist
if [ ! -d "solver_cpp/${custom_build_dir}" ]; then
  mkdir -p "solver_cpp/${custom_build_dir}"
  echo "Created build directory: solver_cpp/${custom_build_dir}"
fi

script_dir=$(dirname "$0")
# get absolute path to the script directory
script_dir=$(cd "$script_dir" && pwd)

cd "$script_dir" || exit 1

# Record timestamp before build
build_start_seconds=$(date +%s)
build_start_date=$(date)
echo "Build started at: $build_start_date"

cd "solver_cpp/${custom_build_dir}" || { echo "Failed to change directory to solver_cpp/${custom_build_dir}"; cd "$current_dir"; exit 1; }


#rm ./* -r || { echo "Build failed"; cd "$current_dir"; exit 1; }
cmake -S .. -DCMAKE_BUILD_TYPE=Release -DUSE_CPP_ML_INTERFACE=${USE_CPP_ML_INTERFACE} || { echo "Build failed"; cd "$current_dir"; exit 1; }
cmake --build . || { echo "Build failed"; cd "$current_dir"; exit 1; }
echo "Build completed successfully"

# Get the build timestamp from the binary
echo ""
echo "Verifying build timestamp..."
binary_output=$(./terrain_solver --print-build-timestamp 2>&1)
build_timestamp_line=$(echo "$binary_output" | grep "Build timestamp:")

if [ -z "$build_timestamp_line" ]; then
    echo "ERROR: Could not extract build timestamp from binary"
    cd "$current_dir"
    exit 1
fi

echo "Binary reports: $build_timestamp_line"

# Extract the timestamp string (everything after "Build timestamp: ")
binary_timestamp_str=$(echo "$build_timestamp_line" | sed 's/Build timestamp: //')

# Convert binary timestamp to seconds since epoch for comparison
# Handle format like "Mar 15 2026 14:23:45"
binary_timestamp_seconds=$(date -d "$binary_timestamp_str" +%s 2>/dev/null)

if [ -z "$binary_timestamp_seconds" ]; then
    echo "ERROR: Could not parse binary timestamp"
    cd "$current_dir"
    exit 1
fi

# Compare timestamps
if [ "$binary_timestamp_seconds" -ge "$build_start_seconds" ]; then
    echo "✓ Binary timestamp verification PASSED"
    echo "  Build started at:  $(date -d @$build_start_seconds)"
    echo "  Binary built at:   $(date -d @$binary_timestamp_seconds)"
else
    echo "✗ Binary timestamp verification FAILED"
    echo "  Build started at:  $(date -d @$build_start_seconds)"
    echo "  Binary built at:   $(date -d @$binary_timestamp_seconds)"
    echo "  Binary timestamp is older than build start time!"
    cd "$current_dir"
    exit 1
fi

cd "$current_dir"