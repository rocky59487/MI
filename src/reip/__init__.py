"""
ReIP: Relevance-Patching-based dynamic causal circuit localization engine.

This module implements Layer-wise Relevance Propagation (LRP) rules as custom
PyTorch autograd functions and integrates them with TransformerLens backward
hooks to replace standard gradients with relevance-conserving propagation
coefficients.

Key components:
    - lrp_rules: LN-rule, Identity-rule, AH-rule, Half-rule, 0-rule
    - backward_hooks: TransformerLens add_hook(dir="bwd") integration
    - pipeline: End-to-end ReIP execution pipeline
    - pruning: Sparse causal topology graph pruning algorithm
"""

from .lrp_rules import (
    LNRule,
    IdentityRule,
    AHRule,
    HalfRule,
    ZeroRule,
    LRP_RULE_REGISTRY,
)
from .backward_hooks import ReIPHookManager
from .pipeline import ReIPPipeline
from .pruning import TopologyPruner

__all__ = [
    "LNRule",
    "IdentityRule",
    "AHRule",
    "HalfRule",
    "ZeroRule",
    "LRP_RULE_REGISTRY",
    "ReIPHookManager",
    "ReIPPipeline",
    "TopologyPruner",
]
