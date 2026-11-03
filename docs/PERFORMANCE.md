# Performance

## 1. Test methods

All matrices are contiguous row-major FP16, with FP32 accumulation, alpha=1 and beta=0. Times are GPU microseconds per complete GEMM. **Time/cuBLAS** means kernel time divided by cuBLAS time for the same measurement method; a value below 1 favors the named kernel.

Build and pass the correctness tests before collecting timings:

```bash
./scripts/build.sh
./scripts/test.sh
./scripts/benchmark.sh --suite current
./scripts/benchmark_square.sh
```

### Warm-input tests

Reuse A and B without explicit cache eviction. Submit 20 warmup calls per implementation and wait for completion. Collect 11 trials, alternating the named kernel and cuBLAS. CUDA events bracket one ordinary default-stream GEMM per sample; report the median. Allocation, input initialization and C reset finish before timing. Reused inputs do not imply that an entire matrix fits in cache.

### Cold-input tests

Submit ten warmup GEMMs per implementation and wait for completion. Before each sample, restore C and read a separate 256 MiB pseudorandom eviction buffer using `ld.global.ca`. Wait for that sweep to finish, then capture one complete GEMM inside the `gemm-profile` NVTX range.

```cpp
// Initialize C before warmup.
d.reset(v);

// 1. Submit ten complete GEMM calls.
for (int w = 0; w < 10; ++w)
    launch(d.s, v, d.a, d.b, d.c);

// 2. Finish warmup.
CK(cudaDeviceSynchronize());

// 3. Restore C so warmup does not change the profiled input.
d.reset(v);

// 4. Evict cached operands; this helper waits for its sweep to finish.
d.cold();

// 5. Capture one complete GEMM, including any internal reduction kernels.
nvtxRangePushA("gemm-profile");
launch(d.s, v, d.a, d.b, d.c);
CK(cudaDeviceSynchronize());
nvtxRangePop();
```

## 2. Final implementation versus cuBLAS

Each row identifies the specialization selected by `gemm_launcher` for that shape. All five shapes execute at their requested M without padding. Warm values are medians of 11 ordinary-launch trials; cold values are medians of nine complete-call profiler trials.

| M,N,K | Selected kernel | Warm µs | Warm cuBLAS µs | Warm time/cuBLAS | Cold µs | Cold cuBLAS µs | Cold time/cuBLAS |
|---|---|---:|---:|---:|---:|---:|---:|
| 1,6144,4096 | `gemm_gemv_kernel<8, 16, 4>` | 192.832 | 188.544 | 1.023 | 189.312 | 205.408 | 0.922 |
| 8,6144,4096 | `gemm_mma_llm_kernel<16, 128, 3>` | 193.792 | 250.016 | 0.775 | 182.144 | 253.888 | 0.717 |
| 32,6144,4096 | `gemm_mma_llm_kernel<32, 128, 3>` | 190.912 | 195.008 | 0.979 | 185.792 | 198.368 | 0.937 |
| 2048,6144,4096 | `gemm_mma_llm_kernel<64, 128, 3, 32>` | 2892.544 | 2731.104 | 1.059 | 3406.112 | 3109.952 | 1.095 |
| 2048,4096,14336 | `gemm_mma_llm_kernel<128, 64, 4, 4>` | 7348.096 | 6608.704 | 1.112 | 7808.032 | 7072.896 | 1.104 |

![Warm and cold time/cuBLAS ratios for the final kernels](figures/final-warm-cold.svg)

Note:
mma_llm<16, 128, 3> is ~28% faster than cublas for the following reasons:
  1. efficient memory load. direct cp.async from gmem to smem, where cublas use intermediate regsters.
  2. cublas use split k and merging, this has additonal merge kernel cost, but this is small 9us/253us
  3. cublas's launch has less warp shared tile. it use 1 warp/block to compute A(16 * 64) matmul B (64 * 16), where mma_llm use 4 warps/block to compute A(16*32) matmul B(16*64)
     l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum shows 25% less traffic, comes from A load 24MB (cublas) -> 6MB (mma_llm)
the speedup comes from no split k (9us) and more efficient memory staging (4 warps per block, direct gmem to smem copy, skipping reg intermediate)

## 3. Progressive kernel comparisons

Rows follow the source's teaching order: scalar global loads, shared-memory FP32 FMA, WMMA, then asynchronous/swizzled Tensor Core staging. A later row is not necessarily faster. Every implementation is named explicitly; cuBLAS is called through `gemm_cublas`.

**Executed M** is the physical number of rows passed to the implementation. Padded rows perform extra work and are diagnostic controls, not equal-work speedup claims. Time/cuBLAS uses the cuBLAS control at the requested M; the M=32 table also retains a cuBLAS M=64 padding control. Grid and block dimensions are (x,y), with z=1.

### M,N,K = 8,6144,4096

| Kernel | Executed M | Grid (x,y) | Block (x,y) | GPU µs | Time/cuBLAS |
|---|---:|---|---|---:|---:|
| `gemm_base_kernel` | 8 | (192,1) | (32,8) | 818.112 | 3.222 |
| `gemm_smem_kernel` | 8 | (192,1) | (32,32) | 2438.368 | 9.604 |
| `gemm_smem_reg_kernel` | 8 | (192,1) | (16,16) | 799.616 | 3.149 |
| `gemm_smem_reg_half2_kernel` | 8 | (192,1) | (16,16) | 721.216 | 2.841 |
| `gemm_wmma_smem_kernel` | 8 | (384,1) | (32,1) | 515.744 | 2.031 |
| `gemm_wmma_gmem_kernel` | 16 (padded) | (384,1) | (32,1) | 247.104 | 0.973 |
| `gemm_wmma_pipeline_kernel<3>` | 64 (padded) | (96,1) | (128,1) | 210.912 | 0.831 |
| `gemm_wmma_swizzle_kernel<3>` | 64 (padded) | (96,1) | (128,1) | 215.136 | 0.847 |
| `gemm_mma_llm_kernel<16, 128, 3>` | 8 | (48,1) | (128,1) | 182.144 | 0.717 |
| cuBLAS (`gemm_cublas`) | 8 | library | library | 253.888 | 1.000 |

![GPU execution times for M,N,K = 8,6144,4096](figures/progressive-m8.svg)

`gemm_smem_kernel` is slower than base because it uses a 32-row tile while there are only eight real output rows, so 24 tile rows do not contribute stored outputs.

### M,N,K = 32,6144,4096

| Kernel | Executed M | Grid (x,y) | Block (x,y) | GPU µs | Time/cuBLAS |
|---|---:|---|---|---:|---:|
| `gemm_base_kernel` | 32 | (192,4) | (32,8) | 2051.648 | 10.754 |
| `gemm_smem_kernel` | 32 | (192,1) | (32,32) | 1769.376 | 9.274 |
| `gemm_smem_reg_kernel` | 32 | (192,1) | (16,16) | 598.656 | 3.138 |
| `gemm_smem_reg_half2_kernel` | 32 | (192,1) | (16,16) | 578.400 | 3.032 |
| `gemm_f4_kernel` | 32 | (192,1) | (4,32) | 857.376 | 4.494 |
| `gemm_wmma_smem_kernel` | 32 | (384,2) | (32,1) | 831.968 | 4.361 |
| `gemm_wmma_gmem_kernel` | 32 | (384,2) | (32,1) | 496.000 | 2.600 |
| `gemm_wmma_smem_reg_kernel` | 32 | (192,1) | (32,1) | 1244.544 | 6.523 |
| `gemm_wmma_pipeline_kernel<3>` | 64 (padded) | (96,1) | (128,1) | 197.696 | 1.036 |
| `gemm_wmma_swizzle_kernel<3>` | 64 (padded) | (96,1) | (128,1) | 202.112 | 1.059 |
| `gemm_mma_llm_kernel<32, 128, 3>` | 32 | (48,1) | (128,1) | 184.672 | 0.968 |
| cuBLAS (`gemm_cublas`) | 32 | library | library | 190.784 | 1.000 |
| cuBLAS (`gemm_cublas`) | 64 (padded) | library | library | 191.776 | 1.005 |

![GPU execution times for M,N,K = 32,6144,4096](figures/progressive-m32.svg)

### M,N,K = 1024,1024,1024

| Kernel | Executed M | Grid (x,y) | Block (x,y) | GPU µs | Time/cuBLAS |
|---|---:|---|---|---:|---:|
| `gemm_base_kernel` | 1024 | (32,128) | (32,8) | 3672.064 | 42.126 |
| `gemm_smem_kernel` | 1024 | (32,32) | (32,32) | 2870.944 | 32.936 |
| `gemm_smem_reg_kernel` | 1024 | (32,32) | (16,16) | 1010.048 | 11.587 |
| `gemm_smem_reg_half2_kernel` | 1024 | (32,32) | (16,16) | 878.368 | 10.077 |
| `gemm_wmma_smem_kernel` | 1024 | (64,64) | (32,1) | 1048.832 | 12.032 |
| `gemm_wmma_gmem_kernel` | 1024 | (64,64) | (32,1) | 496.672 | 5.698 |
| `gemm_wmma_pipeline_kernel<3>` | 1024 | (16,16) | (128,1) | 125.952 | 1.445 |
| `gemm_mma_llm_kernel<64, 128, 3, 32>` | 1024 | (128,1) | (128,1) | 97.216 | 1.115 |
| `gemm_wmma_swizzle_kernel<3>` | 1024 | (16,16) | (128,1) | 95.552 | 1.096 |
| cuBLAS (`gemm_cublas`) | 1024 | library | library | 87.168 | 1.000 |

![GPU execution times for M,N,K = 1024,1024,1024](figures/progressive-square.svg)
