"""Optional CUDA arithmetic matching the official EAGLE4 row implementation.

The vLLM RMSNorm forward is paired with the analytic RMSNorm derivative. No
eagle-harness code or weights are included. Parameter names remain unchanged.
"""

import importlib

import torch
from torch.nn import functional as F

from .llama3_eagle import LlamaAttention, LlamaMLP, LlamaRMSNorm, rotate_half


class _NativeRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, epsilon):
        if value.device.type != "cuda":
            raise RuntimeError("eagle4_precision=vllm requires CUDA for forward")
        try:
            from vllm import ir

            importlib.import_module("vllm.kernels")
        except ImportError as exc:
            raise RuntimeError(
                "eagle4_precision=vllm requires a compatible vLLM CUDA runtime"
            ) from exc
        ctx.save_for_backward(value, weight)
        ctx.epsilon = epsilon
        with ir.ops.rms_norm.set_priority(["vllm_c", "native"]):
            output = ir.ops.rms_norm(
                value.reshape(-1, value.shape[-1]).contiguous(), weight, epsilon
            )
        return output.reshape(value.shape)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        value, weight = ctx.saved_tensors
        x, w, grad = value.float(), weight.float(), grad_output.float()
        scale = torch.rsqrt(x.square().mean(-1, keepdim=True) + ctx.epsilon)
        weighted = grad * w
        dx = scale * (
            weighted - x * scale.square() * (weighted * x).mean(-1, keepdim=True)
        )
        dw = (grad * x * scale).reshape(-1, x.shape[-1]).sum(0)
        return dx.to(value.dtype), dw.to(weight.dtype), None


class Eagle4NativeRMSNorm(LlamaRMSNorm):
    def forward(self, hidden_states):
        return _NativeRMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)


class Eagle4NativeAttention(LlamaAttention):
    def project_qkv(self, hidden_states):
        # Grouping changes BF16 GEMM rounding; keep separate checkpoint keys.
        weight = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0
        )
        projected = F.linear(
            hidden_states.reshape(-1, hidden_states.shape[-1]), weight
        ).reshape(*hidden_states.shape[:-1], -1)
        return projected.split(
            (
                self.num_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ),
            dim=-1,
        )

    def apply_rotary(self, q, k, cos, sin, position_ids):
        cos = cos.squeeze(1).squeeze(0)[position_ids].unsqueeze(1).float()
        sin = sin.squeeze(1).squeeze(0)[position_ids].unsqueeze(1).float()
        q32, k32 = q.float(), k.float()
        return (
            (q32 * cos + rotate_half(q32) * sin).to(q.dtype),
            (k32 * cos + rotate_half(k32) * sin).to(k.dtype),
        )


class Eagle4NativeMLP(LlamaMLP):
    def forward(self, value):
        weight = torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0)
        projected = F.linear(value.reshape(-1, value.shape[-1]), weight).reshape(
            *value.shape[:-1], -1
        )
        gate, up = projected.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
