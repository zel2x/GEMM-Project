"""Illustrative FP16 linear replacement; stock execution is the fallback."""
import functools
import os
import torch

_INSTALLED = False
_ROLES = {"qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"}


def signature(weight):
    # Inference weights are immutable; repack after reloading or modifying them.
    version = None if torch.is_inference(weight) else weight._version
    return weight.data_ptr(), tuple(weight.shape), weight.dtype, weight.device, version


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    mode = os.environ.get("GEMM_SAMPLE_MODE", "stock")
    if mode not in ("stock", "qkv_only", "all_dense"):
        raise ValueError("GEMM_SAMPLE_MODE must be stock, qkv_only or all_dense")
    if mode == "stock":
        return

    # Load an already compiled operator, never compile during inference.
    torch.ops.load_library(os.environ["GEMM_SAMPLE_LIBRARY"])
    from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    original_load = ModelLoader.load
    original_apply = UnquantizedLinearMethod.apply

    @functools.wraps(original_load)
    def load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        model = result[0]  # This is the evaluated runtime's ModelLoader return format.
        with torch.no_grad():
            for name, module in model.named_modules():
                role = name.rsplit(".", 1)[-1]
                selected = role in _ROLES and (mode == "all_dense" or role == "qkv_proj")
                if not selected or not isinstance(module, Linear):
                    continue
                if not isinstance(module.quant_method, UnquantizedLinearMethod):
                    continue
                w = module.weight
                if module.bias is not None or w.dtype != torch.float16 or not w.is_cuda or not w.is_contiguous():
                    continue
                # Keep W[N,K] for stock fallback and preserve tied embeddings.
                module.register_buffer("_sample_gemm_b", w.T.contiguous(), persistent=False)
                module._sample_gemm_signature = signature(w)
        # Startup only: finish packing before warmup and graph capture.
        torch.cuda.synchronize()
        return result

    @functools.wraps(original_apply)
    def apply(self, module, x, bias):
        b = getattr(module, "_sample_gemm_b", None)
        w = module.weight
        if b is not None and isinstance(x, torch.Tensor) and x.ndim >= 2:
            n, k = w.shape
            m = x.numel() // k if x.shape[-1] == k else 0
            eligible = (
                bias is None and x.dtype == w.dtype == b.dtype == torch.float16
                and x.is_cuda and w.is_cuda and b.is_cuda
                and x.device == w.device == b.device
                and x.is_contiguous() and w.is_contiguous() and b.is_contiguous()
                and ((x.data_ptr() | w.data_ptr() | b.data_ptr()) & 15) == 0
                and 1 <= m <= 16 and n >= 512 and n % 64 == 0
                and k >= 512 and k % 32 == 0
                and module._sample_gemm_signature == signature(w)
            )
            if eligible:
                y = torch.ops.trtllm_gemm_sample.gemm(x.view(-1, k), b)
                return y.view(*x.shape[:-1], n)
        return original_apply(self, module, x, bias)

    ModelLoader.load = load
    UnquantizedLinearMethod.apply = apply
    _INSTALLED = True
