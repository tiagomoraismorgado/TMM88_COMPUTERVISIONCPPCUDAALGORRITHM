# Contributing to cuda_mad_exploits

Thank you for your interest in improving this research prototype.

## How to contribute

1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Add or update tests (if applicable).
4. Run the CI command:
   - `mkdir -p build && cd build`
   - `cmake .. -DCMAKE_BUILD_TYPE=Release`
   - `cmake --build . -- -j$(nproc)`
5. Create a pull request with a clear summary, motivation, and testing results.

## Coding standards

- Prefer C++17 modern style (no raw pointers where RAII can be used).
- Ensure CUDA kernels are boundary-checked and use error macros.
- Keep naming consistent: `camelCase` for functions, `kConstant` for compile-time constants.

## Issues

- Use issue templates to describe bug reports and feature requests.
- Include platform, CUDA version, OpenCV version, and driver version.
