# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone, inference-only VeOmni v0.1.0 fused-MoE compatibility shim.

This module preserves the ``veomni.ops.fused_moe_forward`` call signature used
by VeOmni v0.1.0 while removing VeOmni's training, Expert Parallelism (EP), NPU,
and Seed-kernel dependencies. It is intended for single-device inference only.

The CUDA fast path uses a small Triton grouped-linear kernel. If Triton is not
available, the tensors are not on CUDA, or ``LLADA_MOE_BACKEND=eager`` is set,
the implementation falls back to ordinary PyTorch operations.

Replace the original model-code import with, for example,
``from .fused_moe_v010 import fused_moe_forward``.

Derived from ByteDance-Seed/VeOmni v0.1.0.post1:
https://github.com/ByteDance-Seed/VeOmni/tree/v0.1.0.post1
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # The eager fallback does not require Triton.
    triton = None
    tl = None


_SUPPORTED_TRITON_DTYPES = (torch.float16, torch.bfloat16)


if triton is not None:

    @triton.jit
    def _grouped_linear_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        expert_cumsum_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute per-expert ``input @ weight.T`` for contiguous tensors."""
        block_m = tl.program_id(axis=0)
        block_n = tl.program_id(axis=1)
        expert = tl.program_id(axis=2)

        expert_start = tl.load(expert_cumsum_ptr + expert - 1, mask=expert > 0, other=0)
        expert_end = tl.load(expert_cumsum_ptr + expert)
        expert_tokens = expert_end - expert_start

        if block_m * BLOCK_M >= expert_tokens:
            return

        row_offsets = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offsets = tl.arange(0, BLOCK_K)

        input_ptrs = (
            input_ptr
            + (expert_start + row_offsets[:, None]) * K
            + k_offsets[None, :]
        )
        weight_ptrs = (
            weight_ptr
            + expert * N * K
            + col_offsets[None, :] * K
            + k_offsets[:, None]
        )

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_block in range(0, tl.cdiv(K, BLOCK_K)):
            remaining_k = K - k_block * BLOCK_K
            inputs = tl.load(
                input_ptrs,
                mask=(row_offsets[:, None] < expert_tokens) & (k_offsets[None, :] < remaining_k),
                other=0.0,
            )
            weights = tl.load(
                weight_ptrs,
                mask=(col_offsets[None, :] < N) & (k_offsets[:, None] < remaining_k),
                other=0.0,
            )
            accumulator += tl.dot(inputs, weights)
            input_ptrs += BLOCK_K
            weight_ptrs += BLOCK_K

        output_ptrs = (
            output_ptr
            + (expert_start + row_offsets[:, None]) * N
            + col_offsets[None, :]
        )
        tl.store(
            output_ptrs,
            accumulator,
            mask=(row_offsets[:, None] < expert_tokens) & (col_offsets[None, :] < N),
        )


def _validate_inputs(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
) -> None:
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if torch.is_grad_enabled():
        raise RuntimeError(
            "This standalone fused_moe_forward is inference-only. Call it under "
            "torch.no_grad() or torch.inference_mode()."
        )
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must have shape [tokens, hidden], got {tuple(hidden_states.shape)}")
    if routing_weights.ndim != 2 or selected_experts.shape != routing_weights.shape:
        raise ValueError(
            "routing_weights and selected_experts must have the same [tokens, top_k] shape, got "
            f"{tuple(routing_weights.shape)} and {tuple(selected_experts.shape)}"
        )
    if routing_weights.shape[1] == 0:
        raise ValueError("top_k must be positive")
    if routing_weights.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing_weights and hidden_states must contain the same number of tokens")
    if selected_experts.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"selected_experts must be int32 or int64, got {selected_experts.dtype}")
    if fc1_1_weight.ndim != 3 or fc1_2_weight.ndim != 3 or fc2_weight.ndim != 3:
        raise ValueError("expert weights must be rank-3 tensors")
    if fc1_1_weight.shape != fc1_2_weight.shape:
        raise ValueError("fc1_1_weight and fc1_2_weight must have identical shapes")

    experts, intermediate_size, hidden_size = fc1_1_weight.shape
    expected_fc2_shape = (experts, hidden_size, intermediate_size)
    if experts != num_experts:
        raise ValueError(f"num_experts={num_experts}, but the weights contain {experts} experts")
    if hidden_states.shape[1] != hidden_size:
        raise ValueError(f"hidden size is {hidden_states.shape[1]}, but the weights expect {hidden_size}")
    if tuple(fc2_weight.shape) != expected_fc2_shape:
        raise ValueError(f"fc2_weight must have shape {expected_fc2_shape}, got {tuple(fc2_weight.shape)}")
    if selected_experts.numel():
        # These scalar checks synchronize CUDA once, before launching harder-to-debug kernels.
        min_expert = int(selected_experts.min().item())
        max_expert = int(selected_experts.max().item())
        if min_expert < 0 or max_expert >= num_experts:
            raise ValueError(f"selected expert IDs must be in [0, {num_experts}), got [{min_expert}, {max_expert}]")

    devices = {
        hidden_states.device,
        routing_weights.device,
        selected_experts.device,
        fc1_1_weight.device,
        fc1_2_weight.device,
        fc2_weight.device,
    }
    if len(devices) != 1:
        raise ValueError(f"all inputs and weights must be on one device, got {sorted(map(str, devices))}")


def _route_tokens(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort routed token copies by expert and return the inverse permutation."""
    top_k = selected_experts.shape[1]
    flat_experts = selected_experts.reshape(-1).to(torch.int64)
    order = torch.argsort(flat_experts, stable=True)
    sorted_hidden_states = hidden_states[torch.div(order, top_k, rounding_mode="floor")].contiguous()
    sorted_routing_weights = routing_weights.reshape(-1)[order].contiguous()
    tokens_per_expert = torch.bincount(flat_experts, minlength=num_experts)
    expert_cumsum = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32).contiguous()
    return sorted_hidden_states, sorted_routing_weights, expert_cumsum, order


def _unroute_tokens(
    sorted_outputs: torch.Tensor,
    order: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    restored = torch.empty_like(sorted_outputs)
    restored[order] = sorted_outputs
    # VeOmni's v0.1.0 gather kernel accumulates the top-k outputs in FP32.
    return restored.view(num_tokens, top_k, -1).sum(dim=1, dtype=torch.float32).to(sorted_outputs.dtype)


def _grouped_linear_triton(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    expert_cumsum: torch.Tensor,
) -> torch.Tensor:
    if triton is None:  # pragma: no cover - guarded by the caller
        raise RuntimeError("Triton is not available")
    if not inputs.is_contiguous() or not weights.is_contiguous():
        raise ValueError("the Triton path requires contiguous inputs and expert weights")

    num_experts, output_size, input_size = weights.shape
    if inputs.shape[1] != input_size:
        raise ValueError(f"input width is {inputs.shape[1]}, but the weights expect {input_size}")

    output = torch.empty((inputs.shape[0], output_size), dtype=inputs.dtype, device=inputs.device)
    block_m, block_n, block_k = 128, 128, 32
    grid = (
        triton.cdiv(inputs.shape[0], block_m),
        triton.cdiv(output_size, block_n),
        num_experts,
    )
    with torch.cuda.device(inputs.device):
        _grouped_linear_kernel[grid](
            inputs,
            weights,
            output,
            expert_cumsum,
            N=output_size,
            K=input_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=8,
            num_stages=3,
        )
    return output


def _triton_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
) -> torch.Tensor:
    sorted_hidden, sorted_routing, expert_cumsum, order = _route_tokens(
        num_experts, routing_weights, selected_experts, hidden_states
    )
    gate = _grouped_linear_triton(sorted_hidden, fc1_1_weight, expert_cumsum)
    up = _grouped_linear_triton(sorted_hidden, fc1_2_weight, expert_cumsum)
    intermediate = F.silu(gate) * up
    intermediate.mul_(sorted_routing.unsqueeze(-1))
    sorted_outputs = _grouped_linear_triton(intermediate.contiguous(), fc2_weight, expert_cumsum)
    return _unroute_tokens(sorted_outputs, order, hidden_states.shape[0], selected_experts.shape[1])


def _eager_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
) -> torch.Tensor:
    sorted_hidden, sorted_routing, expert_cumsum, order = _route_tokens(
        num_experts, routing_weights, selected_experts, hidden_states
    )
    expert_ends = expert_cumsum.to(device="cpu", dtype=torch.int64).tolist()
    outputs: list[torch.Tensor] = []
    start = 0
    for expert, end in enumerate(expert_ends):
        if end > start:
            expert_inputs = sorted_hidden[start:end]
            gate = F.linear(expert_inputs, fc1_1_weight[expert])
            up = F.linear(expert_inputs, fc1_2_weight[expert])
            intermediate = F.silu(gate) * up
            intermediate.mul_(sorted_routing[start:end].unsqueeze(-1))
            outputs.append(F.linear(intermediate, fc2_weight[expert]))
        start = end

    sorted_outputs = torch.cat(outputs, dim=0) if outputs else hidden_states.new_empty((0, hidden_states.shape[1]))
    return _unroute_tokens(sorted_outputs, order, hidden_states.shape[0], selected_experts.shape[1])


def fused_moe_forward(
    module: torch.nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
) -> torch.Tensor:
    """Run the VeOmni v0.1.0 split-weight MoE operation for inference.

    ``module`` is retained for call-site compatibility. Like VeOmni's original
    non-EP implementation, this function does not use it.

    Set ``LLADA_MOE_BACKEND`` to ``auto`` (default), ``triton``, or ``eager``.
    The ``triton`` setting fails loudly if its requirements are not met;
    ``auto`` falls back to the PyTorch implementation.
    """
    del module
    _validate_inputs(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    )

    backend = os.getenv("LLADA_MOE_BACKEND", "auto").lower()
    if backend not in {"auto", "triton", "eager"}:
        raise ValueError(f"LLADA_MOE_BACKEND must be auto, triton, or eager; got {backend!r}")

    compute_dtype = fc1_1_weight.dtype
    if fc1_2_weight.dtype != compute_dtype or fc2_weight.dtype != compute_dtype:
        raise TypeError("all expert weights must have the same dtype")
    hidden_states = hidden_states.to(dtype=compute_dtype)
    routing_weights = routing_weights.to(dtype=compute_dtype)

    if hidden_states.shape[0] == 0:
        return hidden_states

    can_use_triton = (
        triton is not None
        and hidden_states.is_cuda
        and compute_dtype in _SUPPORTED_TRITON_DTYPES
        and fc1_1_weight.is_contiguous()
        and fc1_2_weight.is_contiguous()
        and fc2_weight.is_contiguous()
    )
    if backend == "triton" and not can_use_triton:
        raise RuntimeError(
            "The Triton backend requires Triton, CUDA tensors, contiguous expert weights, "
            "and float16 or bfloat16 weights."
        )
    if backend != "eager" and can_use_triton:
        return _triton_moe_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
        )
    return _eager_moe_forward(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    )


__all__ = ["fused_moe_forward"]
