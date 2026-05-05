"""Core LRP rule implementations used by ReIP.

These are lightweight, framework-agnostic helpers that can be wrapped by
PyTorch hooks/autograd functions in integration code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class LRPRule(str, Enum):
    LN = "ln"
    IDENTITY = "identity"
    AH = "ah"
    HALF = "half"
    ZERO = "zero"


@dataclass(frozen=True)
class LRPContext:
    eps: float = 1e-12


def ln_rule(grad: torch.Tensor, *, ctx: LRPContext | None = None) -> torch.Tensor:
    """LayerNorm relevance rule: pass-through with nan/inf guard."""
    ctx = ctx or LRPContext()
    out = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return out.clamp(min=-1e6, max=1e6)


def identity_rule(grad: torch.Tensor) -> torch.Tensor:
    """Identity-rule: treat nonlinearity derivative as 1."""
    return grad


def ah_rule(grad: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
    """Attention-head rule using detached attention weights."""
    return grad * attn_weights.detach()


def half_rule(grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split relevance equally across multiplicative branches."""
    half = grad * 0.5
    return half, half


def zero_rule(grad: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """0-rule equivalent to Gradient x Input."""
    return grad * inp
