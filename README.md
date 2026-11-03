# GEMM Project

An FP16 matrix-multiplication project with shape-specialized Tensor Core kernels, cuBLAS comparisons and measured performance results.

TL;DR:
 - GEMM source code changes are in [src](src/)
 - Benchmark data and analysis in [Performance](docs/PERFORMANCE.md)
 - Other files are setup code and test helpers.

## Operation

```text
A[M,K] × B[K,N] → C[M,N]
C = half(alpha × FP32_accumulate(A × B) + beta × float(C))
```

A, B and C are contiguous row-major FP16 device buffers. The public C entry point is `gemm_launcher(A, B, C, M, N, K, alpha, beta)`. It launches on the default CUDA stream; the caller synchronizes before reading C. The launcher selects a kernel using dimensions and pointer alignment.

## Third-party code

The bundled challenge test cases come from AlphaGPU GPU Challenges and are licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0). See [NOTICE.md](NOTICE.md) and [licenses/AlphaGPU.txt](licenses/AlphaGPU.txt) for attribution and license details.

## Requirements

- NVIDIA GPU and a CUDA toolkit supporting the target GPU; the default target is SM120.
- CUDA-enabled PyTorch, a C++ compiler and cuBLAS.
- Nsight Compute for cold-input profiling.

The scripts discover the existing WSL installation. Set `CUDA_HOME`, `CUDA_ARCH` or `PYTHON_BIN` to select another installation.

## Quick start

```bash
cd "/path/to/GEMM Project"
./scripts/build.sh
./scripts/test.sh
./scripts/benchmark.sh --suite current
```

Use an otherwise idle GPU for tests and measurements. GPU commands check for other CUDA processes and serialize this project's jobs using a shared lock.

## Commands

| Command | Action |
|---|---|
| `./scripts/build.sh` | Build the current GEMM library. |
| `./scripts/test.sh` | Run six bundled, 84 extended and five precision tests against the public launcher. |
| `./scripts/benchmark.sh --suite current` | Compare the current implementation with cuBLAS using ordinary launches and CUDA events. |
| `./scripts/update_results.sh` | Regenerate the performance guides and tables from packaged measurement records. |
| `./scripts/check_project.sh` | Check table calculations, documentation links and script syntax. |

Build and temporary test/benchmark output are in `build/`, with each run under `build/runs/`. Only the evidence backing the current performance report is retained in `results/reference/`. Successful benchmark runs refresh their corresponding retained dataset. Scripts accept `--help` and work when invoked by absolute path from another directory.

Shared reference records omit hostnames, account-specific paths and device identifiers. Native profiler captures and build output can contain machine/network information; keep them private and exclude them from shared copies. Performance tables use the retained CSV/JSON measurements.

## Read next

- [Implementation](docs/IMPLEMENTATION.md): layouts, kernel selection and optimizations.
- [Performance](docs/PERFORMANCE.md): current measurements, saved tables and measurement methods.
- [Results index](results/reference/manifest.json): retained datasets and file checksums.

## Layout

```text
src/                   Current launcher and kernel headers
tests/                 Consolidated correctness suite and numerical checks
scripts/               Build, test, measurement and table commands
tools/                 Project command and data-processing implementation
docs/                  Implementation, testing and performance guides
results/reference/     De-identified measurements, CSV exports and checksums
build/                 Current library and temporary build files
build/runs/            Temporary test and benchmark output
```
