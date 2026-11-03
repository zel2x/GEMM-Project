# Implementation

## Public interface

```cpp
extern "C" void gemm_launcher(
    const half* A, const half* B, half* C,
    int M, int N, int K, float alpha, float beta);
```

A has M rows and K columns; B has K rows and N columns; C has M rows and N columns. K is the reduction dimension. The implementation accumulates products in FP32, applies alpha and beta, then stores FP16 output. C is both input and output when beta is nonzero. The caller supplies device storage and waits for default-stream work to finish before consuming C.

## Kernel selection

The launcher first tries shape-specialized kernels. A single row can use an FP32 GEMV path. Small batches use narrow-row Tensor Core tiles. Large, eligible matrices use larger tiles and grouped block ordering.

If the specialized selection does not accept a shape, an aligned 64×64 Tensor Core path is tried. Full aligned tiles can use direct-global WMMA. Boundary cases and misaligned pointers use shared-memory WMMA with zero-filled input edges and checked output stores. Empty output dimensions return without launching work.

## Optimizations

- **Shared operand reuse:** several warps consume one staged A tile instead of independently fetching it for every output-column tile.
- **Asynchronous input staging:** `cp.async` moves global-memory operands into shared memory without intermediate data registers. Multiple staging buffers overlap loading with matrix multiplication.
- **Register accumulation:** each warp keeps multiple output fragments in FP32 registers across the K loop.
- **Shape-specific tiles:** narrow M tiles reduce wasted rows during decoding; larger output tiles amortize address and staging work for larger inputs.
- **Grouped block ordering:** nearby blocks process related matrix regions to encourage cache reuse.
- **Checked fallbacks:** dimension and pointer checks choose implementations that can safely handle the supplied layout.

Larger tiles also use more shared memory and produce fewer blocks. Better input reuse can therefore trade off against the number of independent warps available to cover memory waits. The performance guide compares complete operations rather than assuming any one optimization always wins.

## Source organization

- `src/gemm.cu`: host launcher and shape-based selection.
- `src/lib/gemm_progressive_kernel.cuh`: scalar baseline, shared-memory FP32 FMA variants, then WMMA and pipelined/swizzled Tensor Core variants. Includes both direct-global and boundary-safe shared-memory WMMA kernels. The opening catalog describes each implementation and its relation to the others.
- `src/lib/gemm_llm_kernels.cuh`: single-row `gemm_gemv_kernel` followed by multi-row `gemm_mma_llm_kernel`, plus common constants and async-copy/MMA helpers. These kernels serve different workloads, not successive optimization steps.
- `src/lib/gemm_cublas.cuh`: cuBLAS comparator.

The progressive and LLM headers use the same anonymous namespace within each translation unit. `gemm_wmma_swizzle_kernel` uses explicit `ldmatrix` and `mma.sync` instructions; its name does not imply use of the C++ WMMA API.

The current benchmark calls the public launcher directly. Numerical reference calculations for the tests are kept in the test support module.
