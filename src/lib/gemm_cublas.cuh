#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace {
void check_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status));
        std::abort();
    }
}

struct CublasContext {
    cublasHandle_t handle;
    CublasContext() {
        check_cublas(cublasCreate(&handle));
        // Keep split reductions in FP32 even though C is FP16.
        check_cublas(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    }
    ~CublasContext() { cublasDestroy(handle); }
};

__global__ void gemm_cublas_scale_c(half* C, size_t count, float beta) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float previous = beta != 0.0f ? __half2float(C[i]) : 0.0f;
        C[i] = __float2half_rn(beta * previous);
    }
}
}  // namespace

// NVIDIA library version:
// Link starter.cu with -lcublas. Uses the default stream and a cached handle.
extern "C" void gemm_cublas(const half* A, const half* B, half* C, int M, int N, int K,
                            float alpha, float beta, bool column_major = false) {
    if (M <= 0 || N <= 0) return;
    if (K == 0) {
        const size_t count = static_cast<size_t>(M) * N;
        gemm_cublas_scale_c<<<(count + 255) / 256, 256>>>(C, count, beta);
        return;
    }
    static thread_local CublasContext context;  // One fixed GPU per host thread.
    if (column_major) {
        check_cublas(cublasGemmEx(context.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  M, N, K, &alpha,
                                  A, CUDA_R_16F, M,
                                  B, CUDA_R_16F, K,
                                  &beta, C, CUDA_R_16F, M,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    } else {
        // Column-major C^T = B^T*A^T implements row-major C = A*B without copies.
        check_cublas(cublasGemmEx(context.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K, &alpha,
                                  B, CUDA_R_16F, N,
                                  A, CUDA_R_16F, K,
                                  &beta, C, CUDA_R_16F, N,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
}
