#!/usr/bin/env bash
# Builds and runs the host GoogleTest suites for the C++ kernels (cpp/tests).
#
# Needs cmake, a C++20 compiler and OpenCV's core + imgproc modules
# (`brew install opencv` / `apt-get install libopencv-core-dev libopencv-imgproc-dev`).
# GoogleTest is picked up if installed and fetched otherwise.
#
# Extra arguments go to ctest, e.g. `scripts/run-native-tests.sh -R CraftOpsTest`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$ROOT/cpp/tests"
BUILD_DIR="${RNOCR_TESTS_BUILD_DIR:-$SOURCE_DIR/build}"

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Debug}"
cmake --build "$BUILD_DIR" --parallel
ctest --test-dir "$BUILD_DIR" --output-on-failure "$@"
