// Stream-aware inference launcher. The standalone source is not modified.
#include "gemm_llm_kernels.cuh"

extern "C" bool launchGemmSample(const half* a, const half* b, half* c,
                              int m, int n, int k, cudaStream_t stream) {
    const auto addresses = reinterpret_cast<std::uintptr_t>(a)
                         | reinterpret_cast<std::uintptr_t>(b)
                         | reinterpret_cast<std::uintptr_t>(c);
    if (m < 1 || m > 16 || n < 512 || n % 64 || k < 512 || k % 32
        || (addresses & 15u))
        return false;
    if (m == 1) {
        if (n >= 2048 && n % 256 == 0)
            gemm_gemv_kernel<8, 16, 4><<<n / 256, 512, 0, stream>>>(a, b, c, n, k, 1.0f, 0.0f);
        else
            gemm_gemv_kernel<2, 16, 4><<<n / 64, 512, 0, stream>>>(a, b, c, n, k, 1.0f, 0.0f);
    } else if (n % 128 == 0) {
        gemm_mma_llm_kernel<16, 128, 3><<<n / 128, PIPE_THREADS, 0, stream>>>(a, b, c, m, n, k, 1.0f, 0.0f);
    } else {
        gemm_mma_llm_kernel<16, 64, 3><<<n / 64, PIPE_THREADS, 0, stream>>>(a, b, c, m, n, k, 1.0f, 0.0f);
    }
    return true;
}
