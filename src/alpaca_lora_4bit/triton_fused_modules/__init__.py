# Directly copy from https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/fastest-inference-4bit
from .fused_attn import QuantLlamaAttention, make_quant_attn
from .fused_mlp import QuantLlamaMLP, make_fused_mlp, autotune_warmup_fused
