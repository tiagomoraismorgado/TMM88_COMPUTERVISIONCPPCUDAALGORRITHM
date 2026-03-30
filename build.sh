#!/usr/bin/env bash
set -euo pipefail

if ! command -v cmake &> /dev/null; then
  echo "cmake is not installed or not in PATH. Install CMake and retry." >&2
  exit 1
fi

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -- -j$(nproc)

if [ -f cuda_mad_exploits ]; then
  echo "build successful"
else
  echo "build failed" >&2
  exit 1
fi
