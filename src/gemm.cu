#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mma.h>
#include "lib/gemm_progressive_kernel.cuh"
#include "lib/gemm_cublas.cuh"
#include "lib/gemm_llm_kernels.cuh"

void log_gemm_launch(const char* kernel, int M, int N, int K, dim3 grid, dim3 block) {
    printf("Launching %s: M=%d N=%d K=%d, grid=(%u,%u,%u), block=(%u,%u,%u)\n", kernel, M, N, K,
           grid.x, grid.y, grid.z, block.x, block.y, block.z);
    fflush(stdout);
}

bool launch_gemm_llm(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                     float beta) {
    const auto addresses = reinterpret_cast<std::uintptr_t>(A) |
                           reinterpret_cast<std::uintptr_t>(B) |
                           reinterpret_cast<std::uintptr_t>(C);
    if (M <= 0 || N < 512 || K < 512 || (addresses & 15u) != 0 || N % 64 != 0)
        return false;

    // A has a single row
    if (M == 1) {
        if (N >= 2048 && N % 256 == 0) {
            // Short-decode attention/QKV [2026-10-19], Llama (1,6144,4096):
            // Optimized 189.312 us vs cuBLAS 205.408 us; Opt/CU=0.922.
            log_gemm_launch("gemm_gemv_kernel<8, 16, 4>", M, N, K, N / 256, 512);
            gemm_gemv_kernel<8, 16, 4><<<N / 256, 512>>>(A, B, C, N, K, alpha, beta);
        } else {
            log_gemm_launch("gemm_gemv_kernel<2, 16, 4>", M, N, K, N / 64, 512);
            gemm_gemv_kernel<2, 16, 4><<<N / 64, 512>>>(A, B, C, N, K, alpha, beta);
        }
        return true;
    }

    // A is small batch
    if (K % PIPE_K != 0)
        return false;
    if (M < SMALL_M_LIMIT) {
        // Zero-fill missing A rows and mask their C stores.
        if (M <= 16) {
            if (N % 128 == 0) {
                // Small batch/16-row CTA, Llama QKV (8,6144,4096): 182.144 / 253.888 us; Opt/CU=0.717.
                log_gemm_launch("gemm_mma_llm_kernel<16, 128, 3>", M, N, K, N / 128, PIPE_THREADS);
                gemm_mma_llm_kernel<16, 128, 3>
                    <<<N / 128, PIPE_THREADS>>>(A, B, C, M, N, K, alpha, beta);
            } else {
                log_gemm_launch("gemm_mma_llm_kernel<16, 64, 3>", M, N, K, N / 64, PIPE_THREADS);
                gemm_mma_llm_kernel<16, 64, 3>
                    <<<N / 64, PIPE_THREADS>>>(A, B, C, M, N, K, alpha, beta);
            }
        } else {
            if (N % 128 == 0) {
                // Small batch/32-row CTA, Llama QKV (32,6144,4096): 185.792 / 198.368 us; Opt/CU=0.937.
                log_gemm_launch("gemm_mma_llm_kernel<32, 128, 3>", M, N, K,
                                (N / 128) * ceil_div(M, 32), PIPE_THREADS);
                gemm_mma_llm_kernel<32, 128, 3>
                    <<<(N / 128) * ceil_div(M, 32), PIPE_THREADS>>>(A, B, C, M, N, K, alpha, beta);
            } else {
                log_gemm_launch("gemm_mma_llm_kernel<32, 64, 3>", M, N, K,
                                (N / 64) * ceil_div(M, 32), PIPE_THREADS);
                gemm_mma_llm_kernel<32, 64, 3>
                    <<<(N / 64) * ceil_div(M, 32), PIPE_THREADS>>>(A, B, C, M, N, K, alpha, beta);
            }
        }
        return true;
    }

    // Large matmul
    const size_t weight_bytes = static_cast<size_t>(K) * N * sizeof(half);
    // Leave M=64 and cache-fitting weights to the WMMA fallbacks.
    if (M < 128 || M % 64 != 0 || weight_bytes <= 32u * 1024u * 1024u)
        return false;
    if (K >= 8192 && M % 128 == 0) {
        // Long-K prefill, Llama down (2048,4096,14336): 7808.032 / 7072.896 us; Opt/CU=1.104.
        log_gemm_launch("gemm_mma_llm_kernel<128, 64, 4, 4>", M, N, K, (M / 128) * (N / 64),
                        PIPE_THREADS);
        gemm_mma_llm_kernel<128, 64, 4, 4>
            <<<(M / 128) * (N / 64), PIPE_THREADS>>>(A, B, C, M, N, K, alpha, beta);
    } else if (N % 128 == 0) {
        if (K >= 8192) {
            log_gemm_launch("gemm_mma_llm_kernel<64, 128, 4, 8>", M, N, K, (M / 64) * (N / 128),
                            PIPE_THREADS);
            gemm_mma_llm_kernel<64, 128, 4, 8>
                <<<(M / 64) * (N / 128), PIPE_THREADS>>>(A, B, C, M, N, K, alpha, beta);
        } else {
            // Long-prefill attention/QKV [2026-10-19], Llama (2048,6144,4096):
            // Optimized 3406.112 us vs cuBLAS 3109.952 us; Opt/CU=1.095.
            log_gemm_launch("gemm_mma_llm_kernel<64, 128, 3, 32>", M, N, K, (M / 64) * (N / 128),
                            PIPE_THREADS);
            gemm_mma_llm_kernel<64, 128, 3, 32>
                <<<(M / 64) * (N / 128), PIPE_THREADS>>>(A, B, C, M, N, K, alpha, beta);
        }
    } else {
        return false;
    }
    return true;
}

extern "C" void gemm_launcher(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
                      float beta) {
    if (M <= 0 || N <= 0) {
        return;
    }
    if (launch_gemm_llm(A, B, C, M, N, K, alpha, beta))
        return;

    const auto input_address =
        reinterpret_cast<std::uintptr_t>(A) | reinterpret_cast<std::uintptr_t>(B);
    // Swizzled mma.sync interior fast path. Other shapes/offsets retain the original
    // direct-global or boundary-safe shared WMMA implementations below.
    const auto pipeline_address = input_address | reinterpret_cast<std::uintptr_t>(C);
    if ((pipeline_address & 15u) == 0 && M % PIPE_M == 0 && N % PIPE_N == 0 &&
        K > 0 && K % PIPE_K == 0) {
        const dim3 pipe_blocks(N / PIPE_N, M / PIPE_M);
        log_gemm_launch("gemm_wmma_swizzle_kernel<3>", M, N, K, pipe_blocks, PIPE_THREADS);
        gemm_wmma_swizzle_kernel<MMA_STAGES><<<pipe_blocks, PIPE_THREADS>>>(A, B, C, M, N, K, alpha, beta);
        return;
    }
    // Direct WMMA for full aligned tiles; shared WMMA for boundaries/offsets.
    dim3 blocks(ceil_div(N, TILE), ceil_div(M, TILE));
    if (((input_address & 31u) == 0) && (K % TILE == 0) && (M % TILE == 0) && (N % TILE == 0)) {
        log_gemm_launch("gemm_wmma_gmem_kernel", M, N, K, blocks, WARP_SIZE);
        // 507 us
        gemm_wmma_gmem_kernel<<<blocks, WARP_SIZE>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        log_gemm_launch("gemm_wmma_smem_kernel", M, N, K, blocks, WARP_SIZE);
        gemm_wmma_smem_kernel<<<blocks, WARP_SIZE>>>(A, B, C, M, N, K, alpha, beta);
    }
}
