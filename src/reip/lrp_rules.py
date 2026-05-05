"""
LRP Propagation Rules for ReIP (Relevance Patching).

This module implements five custom PyTorch autograd.Function subclasses that
replace standard gradient computation with Layer-wise Relevance Propagation
(LRP) rules derived from Deep Taylor Decomposition (DTD).

Each rule enforces the Relevance Conservation principle:
    sum(R^{l-1}) == sum(R^{l})

Rules implemented:
    - LNRule       : For LayerNorm / RMSNorm layers
    - IdentityRule : For non-linear activations (GELU, SiLU, etc.)
    - AHRule       : For Attention Mechanism
    - HalfRule     : For Multiplicative Gate mechanisms (SwiGLU, GeGLU)
    - ZeroRule     : For standard Linear / FFN layers (gradient × input)

References:
    - RelP paper: https://github.com/FarnoushRJ/RelP
    - Circuit Insights: https://arxiv.org/html/2510.14936v2
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# LN-rule: LayerNorm / RMSNorm
# ---------------------------------------------------------------------------

class _LNRuleFunction(Function):
    """
    Custom autograd function implementing the LN-rule for LayerNorm/RMSNorm.

    During the backward pass, the centering operation and the variance-based
    scaling factor are treated as constants, cutting the gradient path through
    variance computation and preventing 'Relevance Collapse' in deep networks.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        eps: float,
        is_rms: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.is_rms = is_rms

        if is_rms:
            # RMSNorm: normalize by root mean square only
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
            x_norm = x / rms
        else:
            # LayerNorm: center then normalize
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / (var + eps).sqrt()

        out = x_norm * weight if weight is not None else x_norm
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], None, None]:
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        is_rms = ctx.is_rms

        # Treat normalization statistics as constants (LN-rule core)
        if is_rms:
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
            # Propagate relevance through the constant scaling factor
            if weight is not None:
                grad_x = grad_output * weight / rms
            else:
                grad_x = grad_output / rms
        else:
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            std = (var + eps).sqrt()
            if weight is not None:
                grad_x = grad_output * weight / std
            else:
                grad_x = grad_output / std

        grad_weight = (grad_output * x).sum(dim=list(range(grad_output.dim() - 1))) if weight is not None else None
        return grad_x, grad_weight, None, None, None


class LNRule(nn.Module):
    """
    Wrapper module that applies the LN-rule to a LayerNorm or RMSNorm layer.

    Usage:
        ln_rule = LNRule(original_layernorm, is_rms=False)
        output = ln_rule(x)  # Uses LN-rule in backward pass
    """

    def __init__(self, original_norm: nn.Module, is_rms: bool = False):
        super().__init__()
        self.original_norm = original_norm
        self.is_rms = is_rms
        self.eps = getattr(original_norm, "eps", 1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = getattr(self.original_norm, "weight", None)
        bias = getattr(self.original_norm, "bias", None)
        return _LNRuleFunction.apply(x, weight, bias, self.eps, self.is_rms)


# ---------------------------------------------------------------------------
# Identity-rule: Non-linear activations (GELU, SiLU, ReLU, etc.)
# ---------------------------------------------------------------------------

class _IdentityRuleFunction(Function):
    """
    Custom autograd function implementing the Identity-rule.

    During the backward pass, the derivative of the non-linear activation
    function is forced to 1 (treated as a linear identity function).
    This ensures that feature relevance signals are strictly conserved
    when crossing non-linear boundaries.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, activation_fn) -> torch.Tensor:
        ctx.save_for_backward(x)
        return activation_fn(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Identity-rule: derivative = 1, pass gradient unchanged
        return grad_output, None


class IdentityRule(nn.Module):
    """
    Wrapper that applies Identity-rule to any non-linear activation function.

    Usage:
        identity_rule = IdentityRule(torch.nn.functional.gelu)
        output = identity_rule(x)
    """

    def __init__(self, activation_fn=None):
        super().__init__()
        self.activation_fn = activation_fn or torch.nn.functional.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _IdentityRuleFunction.apply(x, self.activation_fn)


# ---------------------------------------------------------------------------
# AH-rule: Attention Mechanism
# ---------------------------------------------------------------------------

class _AHRuleFunction(Function):
    """
    Custom autograd function implementing the AH-rule for Attention.

    The attention weight matrix (softmax output) is treated as a constant
    during backpropagation, linearizing the attention head computation.
    This ensures relevance scores are precisely distributed to Key, Query,
    and Value streams without distortion from the softmax non-linearity.
    """

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(value, attn_weights)
        # Standard attention output: attn_weights @ value
        return torch.matmul(attn_weights, value)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[None, None, torch.Tensor, None]:
        value, attn_weights = ctx.saved_tensors
        # AH-rule: treat attn_weights as constant; propagate only through value
        grad_value = torch.matmul(attn_weights.transpose(-2, -1), grad_output)
        return None, None, grad_value, None


class AHRule(nn.Module):
    """
    Module that applies the AH-rule during attention backward pass.

    This is used primarily for GPT-2 and other early architectures.
    For modern architectures with multiplicative gates, use HalfRule instead.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        return _AHRuleFunction.apply(query, key, value, attn_weights)


# ---------------------------------------------------------------------------
# Half-rule: Multiplicative Gate Mechanisms (SwiGLU, GeGLU)
# ---------------------------------------------------------------------------

class _HalfRuleFunction(Function):
    """
    Custom autograd function implementing the Half-rule for multiplicative gates.

    For modern architectures (Qwen2, Gemma2) that use gated MLP variants
    (SwiGLU, GeGLU), relevance entering a multiplicative branch is forced to
    be split equally (50/50) between the two branches. This prevents spurious
    relevance amplification during backward propagation through element-wise
    multiplication.

    Given: output = gate(x) * linear(x)
    Half-rule: grad_gate = 0.5 * grad_output * linear(x)
               grad_linear = 0.5 * grad_output * gate(x)
    """

    @staticmethod
    def forward(
        ctx,
        gate_output: torch.Tensor,
        linear_output: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(gate_output, linear_output)
        return gate_output * linear_output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_output, linear_output = ctx.saved_tensors
        # Half-rule: split relevance equally between both branches
        grad_gate = 0.5 * grad_output * linear_output
        grad_linear = 0.5 * grad_output * gate_output
        return grad_gate, grad_linear


class HalfRule(nn.Module):
    """
    Module that applies the Half-rule to gated MLP computations.

    Usage:
        half_rule = HalfRule()
        output = half_rule(gate_output, linear_output)
    """

    def forward(
        self,
        gate_output: torch.Tensor,
        linear_output: torch.Tensor,
    ) -> torch.Tensor:
        return _HalfRuleFunction.apply(gate_output, linear_output)


# ---------------------------------------------------------------------------
# 0-rule (Zero-rule): Standard Linear / FFN layers
# ---------------------------------------------------------------------------

class _ZeroRuleFunction(Function):
    """
    Custom autograd function implementing the 0-rule (Zero-rule) for linear layers.

    Mathematically equivalent to Gradient × Input (GI), this is the default
    LRP mode for all standard feed-forward neural network layers and linear
    projection layers, ensuring lossless energy transfer.

    R_i^{l-1} = a_i * (dL/da_i)
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        out = torch.nn.functional.linear(x, weight, bias)
        return out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x, weight = ctx.saved_tensors
        # Standard linear backward (0-rule = standard gradient)
        grad_x = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.reshape(-1, grad_output.shape[-1]).t(),
                                   x.reshape(-1, x.shape[-1]))
        grad_bias = grad_output.sum(dim=list(range(grad_output.dim() - 1)))
        return grad_x, grad_weight, grad_bias


class ZeroRule(nn.Module):
    """
    Module that applies the 0-rule to a linear layer.

    This is the default rule for all Linear layers (W_Q, W_K, W_V, W_O,
    W_in, W_out in MLP blocks).
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ZeroRuleFunction.apply(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Rule Registry
# ---------------------------------------------------------------------------

LRP_RULE_REGISTRY = {
    "ln": LNRule,
    "identity": IdentityRule,
    "ah": AHRule,
    "half": HalfRule,
    "zero": ZeroRule,
}
"""
Registry mapping rule name strings to their corresponding module classes.

Supported keys:
    "ln"       -> LNRule       (LayerNorm / RMSNorm)
    "identity" -> IdentityRule (non-linear activations)
    "ah"       -> AHRule       (attention mechanism)
    "half"     -> HalfRule     (multiplicative gates)
    "zero"     -> ZeroRule     (linear layers, default)
"""
