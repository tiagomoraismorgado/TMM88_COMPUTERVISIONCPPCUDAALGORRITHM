# cuda_mad_exploits
**GPU-accelerated Most Apparent Distortion (MAD) variant with Log-Gabor filterbank**  
**Perceptual full-reference image quality assessment pipeline — extreme optimization testbed**

**Current status:** Experimental research prototype / optimization playground  
**Main purpose:**  
- Create a high-fidelity, numerically verifiable GPU implementation of a MAD-inspired perceptual metric  
- Serve as an aggressive testing ground for virtually every mid-to-low-level CUDA optimization technique that still makes sense in 2026  
- Enable side-by-side bit-exact / near-bit-exact comparison with reference CPU and MATLAB implementations  
- Remain **single-stream, fully synchronous** in the main branch to maximize debuggability and numerical reproducibility

**Important disclaimers**  
This is **not** production-grade code.  
It is intentionally kept simple structurally so that kernel modifications, PTX inspection, Nsight Compute experiments, numerical diffing and kernel fusion decisions remain as frictionless as possible.  
Many patterns used here would **not** be recommended in a latency-critical or multi-GPU production system.

**Reference date:** March 2026

## Core scientific / perceptual motivation

The project implements a perceptual full-reference (FR) image quality metric inspired by  
- **Most Apparent Distortion (MAD)** — Chandler & Hemami (2007–2010)  
- **CW-SSIM** structural similarity in complex wavelet domain  
- **Log-Gabor** filter banks (commonly used in texture analysis and bio-inspired vision models)  
- Multi-scale local contrast & higher-order statistics (variance, skewness, kurtosis)

The pipeline computes two complementary distortion maps:  
1. **Detection-based map** — emphasizes high-frequency, localized, clearly visible distortions  
2. **Appearance-based map** — captures low-frequency, global structural and contrast changes

These maps are then visibility-weighted, pooled using a MAD-like non-linear combination, and fused into a final scalar quality score.

The metric aims to correlate better than plain PSNR / SSIM / MS-SSIM with human perception — especially on:  
- blur + noise mixtures  
- JPEG2000 / HEIF / AVIF compression artifacts  
- GAN / diffusion model hallucinations  
- spatially-varying distortions (lens blur, chromatic aberrations, rendering errors)

## Current pipeline stages (March 2026)

1. **Optional CSF pre-filtering** (contrast sensitivity function in frequency domain)  
   - Uses either Barten 1999 or Movshon / Watson CSF model  
   - Implemented via cuFFT forward → pointwise multiply → inverse

2. **Log-Gabor filter bank decomposition**  
   - 4–6 scales × 4–8 orientations (configurable)  
   - Log-Gabor wavelets instead of classic Gabor (zero DC, better octave separation)  
   - Complex-valued filtering in frequency domain (most efficient path)

3. **Local statistic computation — two regimes**  
   - **High-frequency / detection branch** — small windows (3×3 to 7×7), emphasis on variance & kurtosis  
   - **Low-frequency / appearance branch** — larger windows (11×11 to 31×31), emphasis on mean, skew, structural preservation

4. **Distortion map generation**  
   - Detection map: visibility-weighted local energy differences  
   - Appearance map: local statistical divergence (MAD-style Minkowski-like pooling)

5. **Final score computation**  
   - Non-linear fusion of detection & appearance branches  
   - Optional masking (flat / texture regions)  
   - Scalar output ≈ 0 (excellent) … 1+ (very poor)

6. **Verification harness**  
   - Per-stage L2 / L∞ differences vs. golden reference  
   - Optional `--verify-bitexact` mode (requires float32 CPU reference)  
   - Tolerance maps saved as 16-bit PNG + difference heatmaps

## Active GPU optimization experiments (March 2026)

The repository is structured as a living diary of optimization ideas — most are benchmarked with Nsight Compute / Nsight Systems.

### Memory system exploits
- 2D vs. 1D shared memory tiling patterns (with/without halo)  
- Bank-conflict-free access patterns for filter coefficient loading  
- `__ldg()` + texture cache vs. direct global loads for read-only coefficient tables  
- Vectorized loads/stores (float4 / ushort4) at different pipeline stages  
- Pinned host memory + async memcpy overlap (disabled in main branch, exists in `async/`)

### Compute & occupancy tricks
- Register pressure vs. occupancy trade-off (via `--maxrregcount`)  
- Aggressive loop unrolling + `#pragma unroll` + template specialization on block size  
- Warp-level reductions using shuffle intrinsics / cooperative groups  
- Inline PTX for selected math functions (rcp, rsqrt, fma variants)  
- `__restrict__` + pointer aliasing hints everywhere reasonable

### cuFFT related
- In-place vs. out-of-place transforms  
- Different plan caching strategies  
- Batch mode vs. single-image mode performance cliffs  
- cuFFT callback API experiments (for pointwise operations without extra kernel)

### Algorithmic / numerical
- Single-precision vs. mixed f32/f16 computation (where quality degradation is acceptable)  
- Different normalization strategies for Log-Gabor energy  
- Approximate reciprocal square-root vs. true rsqrtf  
- Fused multiply-add density comparison across kernels

All major experiments are documented in `experiments/YYYY-MM/` folders with:  
- Markdown tables (kernel time, registers, occupancy, shared mem usage, achieved bandwidth)  
- Nsight Compute section screenshots  
- Numerical difference plots (per-pixel & aggregate)  
- PTX / SASS excerpts for critical sections

## Build & development environment (2026 edition)

### Minimum requirements
- CUDA Toolkit 12.4 – 12.8 (strongly recommended 12.6+)  
- NVIDIA driver ≥ 550.xx (RTX 40 & 50 series)  
- OpenCV 4.9+ or 4.10 (core, imgproc, imgcodecs) — preferably built with CUDA support  
- CMake 3.25+  
- Compiler: MSVC 2022 17.10+, GCC 12/13, Clang 16/17/18  
- Python 3.10+ (optional — used for reference metric & plotting scripts)

### Recommended development GPUs (March 2026)
- RTX 4090 / 5090 (24/32 GB) — best for large filter banks & batch experiments  
- RTX 4080 Super / 5080 — good price/performance  
- A6000 / L40 / L40S — if doing long unattended runs  
- RTX 3090 / 4060 Ti 16 GB — still usable but memory-constrained

### Quick build & test

```bash
git clone https://github.com/yourname/cuda_mad_exploits.git
cd cuda_mad_exploits

# Recommended: out-of-source build
mkdir -p build/release && cd build/release
cmake ../.. -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES="89;90" \
            -DOpenCV_DIR=/path/to/opencv/build

make -j$(nproc)

# Basic smoke test
./cuda_mad_exploits --ref ../data/reference/clean_building.png \
                    --dist ../data/distorted/building_jp2k_0.4bpp.png \
                    --output-score

# Full verification mode (slower, saves maps)
./cuda_mad_exploits --ref ref.png --dist dist.png \
                    --verify --save-maps --diff-tolerance 1.5e-5