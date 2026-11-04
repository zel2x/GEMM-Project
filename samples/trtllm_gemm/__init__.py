"""Sample entry point for TRT-LLM's --custom_module_dirs option.

Build binding.cpp and launcher.cu as a PyTorch CUDA extension, including the
existing src/lib directory. Point GEMM_SAMPLE_LIBRARY at that shared library.
Use GEMM_SAMPLE_MODE=all_dense or qkv_only to enable replacement; stock is default.
Import this package before model loading, warmup and CUDA graph capture.
"""
from .adapter import install

install()

# Pass-through registration carries this package into spawned model workers.
from tensorrt_llm._torch.models.modeling_llama import LlamaForCausalLM
from tensorrt_llm._torch.models.modeling_utils import register_auto_model


@register_auto_model("LlamaForCausalLM")
class SampleLlamaForCausalLM(LlamaForCausalLM):
    pass
