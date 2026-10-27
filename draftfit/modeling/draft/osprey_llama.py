"""Llama-source Osprey backbones reuse the shared attention/TTT stack without Q/K norms."""
import torch.nn as nn
from transformers import LlamaConfig
from .osprey_qwen3 import OspreyQwen3DraftModel
from .registry import register_draft


@register_draft
class OspreyLlamaDraftModel(OspreyQwen3DraftModel):
    config_class = LlamaConfig

    def __init__(self, config, **kwargs):
        if not hasattr(config, "head_dim"):
            config.head_dim = config.hidden_size // config.num_attention_heads
        rope = getattr(config, "rope_parameters", None) or {}
        if rope.get("rope_type") == "llama3":
            config.rope_scaling = dict(rope)
        super().__init__(config, **kwargs)
        layers = self.layers if hasattr(self, "layers") else [self.midlayer]
        for layer in layers:
            layer.self_attn.q_norm = nn.Identity()
            layer.self_attn.k_norm = nn.Identity()
