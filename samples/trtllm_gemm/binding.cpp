// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" bool launchGemmSample(half const* a, half const* b, half* c,
    int m, int n, int k, cudaStream_t stream);

namespace
{
at::Tensor gemm(at::Tensor const& a, at::Tensor const& b)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device() == b.device(), "CUDA device mismatch");
    TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf, "FP16 required");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0), "Invalid GEMM dimensions");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Contiguous operands required");
    auto const m = a.size(0);
    auto const k = a.size(1);
    auto const n = b.size(1);
    TORCH_CHECK(m >= 1 && m <= 16 && n >= 512 && n % 64 == 0 && k >= 512 && k % 32 == 0,
        "Shape outside sample dispatch");
    TORCH_CHECK(((reinterpret_cast<std::uintptr_t>(a.data_ptr())
        | reinterpret_cast<std::uintptr_t>(b.data_ptr())) & 15U) == 0, "16-byte alignment required");
    c10::cuda::CUDAGuard const guard(a.device());
    auto output = at::empty({m, n}, a.options());
    auto const stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
    bool const launched = launchGemmSample(
        reinterpret_cast<half const*>(a.const_data_ptr<at::Half>()),
        reinterpret_cast<half const*>(b.const_data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.mutable_data_ptr<at::Half>()),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), stream);
    TORCH_CHECK(launched, "Supported dispatch did not launch");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

at::Tensor gemmMeta(at::Tensor const& a, at::Tensor const& b)
{
    return at::empty({a.size(0), b.size(1)}, a.options());
}
} // namespace

TORCH_LIBRARY(trtllm_gemm_sample, m)
{
    m.def("gemm(Tensor a, Tensor b) -> Tensor");
}
TORCH_LIBRARY_IMPL(trtllm_gemm_sample, CUDA, m)
{
    m.impl("gemm", &gemm);
}
TORCH_LIBRARY_IMPL(trtllm_gemm_sample, Meta, m)
{
    m.impl("gemm", &gemmMeta);
}
