# Maintenance Guide for cuda_mad_exploits

## Goals

- Keep code compilable across modern CUDA and OpenCV versions
- Keep baseline feature set stable (MAD + Log-Gabor evaluation)
- Keep repository idiomatic and easy for new contributors

## Build / CI

1. Ensure `CMakeLists.txt` has up-to-date minimum versions.
2. Keep `README.md` quick-start commands in sync:
   - `build.sh`, `build.ps1`
   - `ctest --output-on-failure`
3. Validate GitHub actions YAML on each major change.

## Code hygiene

- Run `clang-format -i` on modified files.
- Run `clang-tidy` with a lightweight profile (e.g., modernize and readability checks).
- Keep one linear `CMakeLists` without duplication.

## Repository structure

- `main.cpp` and `kernel.cu` are the core pipeline; refactor here first.
- `header.h` is currently a large concept file and can be broken into smaller module headers.
- The project’s code style uses `camelCase` and `constexpr` for compile-time constants.

## Adding features

1. Add new code in separate source file(s) under root or `src/`.
2. Add corresponding targets in CMake.
3. Add regression tests in `tests/` and wire them into `ctest`.
4. Update `README.md` and `MAINTENANCE.md`.

## Troubleshooting

- `cmake` not found: install CMake and ensure it is in PATH.
- OpenCV component missing: install `libopencv-dev` (Linux) or configure `OpenCV_DIR`.
- `CUDA` component missing: install CUDA Toolkit and validate `nvcc`.
