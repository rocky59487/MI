"""
TransformerLens Backward Hook Integration for ReIP.

This module provides the ReIPHookManager class, which intercepts standard
gradients in TransformerLens models via add_hook(dir="bwd") and injects
custom LRP propagation rules.

The hook manager supports activation via:
    model.cfg.use_lrp = True
    model.cfg.LRP_rules = ["ln", "identity", "ah", "half", "zero"]

Architecture-specific default rule sets are provided for:
    - GPT-2 / Pythia
    - Qwen2
    - Gemma2-2B / Llama-3.x
"""

from __future__ import annotations

import torch
from typing import Any, Callable, Dict, List, Optional, Tuple
from contextlib import contextmanager

from .lrp_rules import (
    LNRule,
    IdentityRule,
    AHRule,
    HalfRule,
    ZeroRule,
    LRP_RULE_REGISTRY,
)

# ---------------------------------------------------------------------------
# Architecture-specific default LRP rule configurations
# ---------------------------------------------------------------------------

ARCH_DEFAULT_RULES: Dict[str, List[str]] = {
    "gpt2":    ["ln", "identity", "ah",   "zero"],
    "pythia":  ["ln", "identity", "ah",   "zero"],
    "qwen2":   ["ln", "identity", "half", "zero"],
    "gemma2":  ["ln", "identity", "half", "zero"],
    "llama":   ["ln", "identity", "half", "zero"],
}
"""
Default LRP rule sets per model architecture family.
Keys are lowercase architecture name prefixes.
"""


def _get_default_rules(model_name: str) -> List[str]:
    """Infer default LRP rules from the model name string."""
    model_name_lower = model_name.lower()
    for arch_key, rules in ARCH_DEFAULT_RULES.items():
        if arch_key in model_name_lower:
            return rules
    # Fallback: conservative default set
    return ["ln", "identity", "zero"]


# ---------------------------------------------------------------------------
# Hook functions injected into TransformerLens backward pass
# ---------------------------------------------------------------------------

def _make_ln_backward_hook() -> Callable:
    """
    Returns a backward hook that sanitizes LayerNorm gradients for numeric stability.

    The hook preserves gradient direction and only removes NaN/Inf values.
    This is a stability guard, not a full LN-rule reparameterization.
    """
    def hook_fn(grad: torch.Tensor) -> torch.Tensor:
        # We cannot recover LayerNorm forward statistics at this hook point.
        # Keep directionality intact and only sanitize numerical pathologies
        # to avoid injecting arbitrary rescaling into relevance attribution.
        return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return hook_fn


def _make_identity_backward_hook() -> Callable:
    """Returns a backward hook that applies Identity-rule (passes grad unchanged)."""
    def hook_fn(grad: torch.Tensor) -> torch.Tensor:
        return grad  # Identity: derivative = 1
    return hook_fn


def _make_half_backward_hook(split_factor: float = 0.5) -> Callable:
    """
    Returns a backward hook that applies Half-rule to multiplicative gates.

    The gradient is scaled by split_factor (default 0.5) to enforce equal
    relevance distribution between the two branches of a gated MLP.
    """
    def hook_fn(grad: torch.Tensor) -> torch.Tensor:
        return grad * split_factor
    return hook_fn


# ---------------------------------------------------------------------------
# ReIPHookManager
# ---------------------------------------------------------------------------

class ReIPHookManager:
    """
    Manages the registration and cleanup of LRP backward hooks in a
    TransformerLens HookedTransformer model.

    This class provides a context manager interface for clean hook lifecycle
    management, ensuring no VRAM leaks occur during extended analysis sessions.

    Example usage::

        from transformer_lens import HookedTransformer
        from src.reip import ReIPHookManager

        model = HookedTransformer.from_pretrained("gpt2")
        hook_manager = ReIPHookManager(model, rules=["ln", "identity", "zero"])

        with hook_manager.active():
            logits, cache = model.run_with_cache(tokens)
            loss = logits[0, -1, target_token]
            loss.backward()
            # Gradients in cache are now LRP-modified relevance scores
    """

    def __init__(
        self,
        model: Any,  # transformer_lens.HookedTransformer
        rules: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        verbose: bool = False,
        strict: bool = False,
    ):
        """
        Args:
            model: A TransformerLens HookedTransformer instance.
            rules: List of rule names to apply. If None, inferred from model_name.
            model_name: Model name string used to infer default rules.
            verbose: If True, print hook registration details.
        """
        self.model = model
        self.verbose = verbose
        self.strict = strict
        self._hook_handles: List[Any] = []
        self._grad_stores: Dict[str, torch.Tensor] = {}

        # Resolve rules
        if rules is not None:
            self.rules = rules
        elif model_name is not None:
            self.rules = _get_default_rules(model_name)
        else:
            cfg_name = getattr(getattr(model, "cfg", None), "model_name", "")
            self.rules = _get_default_rules(cfg_name)

        if verbose:
            print(f"[ReIPHookManager] Active LRP rules: {self.rules}")

    # ------------------------------------------------------------------
    # Hook point name resolution helpers
    # ------------------------------------------------------------------

    def _get_ln_hook_names(self) -> List[str]:
        """Return TransformerLens hook point names for LayerNorm outputs."""
        n_layers = self.model.cfg.n_layers
        names = []
        for layer in range(n_layers):
            names.append(f"blocks.{layer}.ln1.hook_normalized")
            names.append(f"blocks.{layer}.ln2.hook_normalized")
        names.append("ln_final.hook_normalized")
        return names

    def _get_mlp_act_hook_names(self) -> List[str]:
        """Return hook point names for MLP activation outputs."""
        n_layers = self.model.cfg.n_layers
        return [f"blocks.{layer}.mlp.hook_post" for layer in range(n_layers)]

    def _get_attn_pattern_hook_names(self) -> List[str]:
        """Return hook point names for attention pattern (softmax output)."""
        n_layers = self.model.cfg.n_layers
        return [f"blocks.{layer}.attn.hook_pattern" for layer in range(n_layers)]

    def _get_mlp_gate_hook_names(self) -> List[str]:
        """Return hook point names for gated MLP (SwiGLU/GeGLU) outputs."""
        n_layers = self.model.cfg.n_layers
        return [f"blocks.{layer}.mlp.hook_pre_linear" for layer in range(n_layers)]

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_ln_hooks(self) -> None:
        """Register LN-rule backward hooks on all LayerNorm hook points."""
        for hook_name in self._get_ln_hook_names():
            try:
                hook_fn = _make_ln_backward_hook()
                handle = self.model.add_hook(
                    hook_name,
                    hook_fn,
                    dir="bwd",
                    is_permanent=False,
                )
                self._hook_handles.append(handle)
                if self.verbose:
                    print(f"[ReIPHookManager] Registered LN-rule hook: {hook_name}")
            except Exception as exc:
                if self.strict:
                    raise RuntimeError(f"Failed to register LN hook: {hook_name}") from exc

    def _register_identity_hooks(self) -> None:
        """Register Identity-rule backward hooks on MLP activation hook points."""
        hook_fn = _make_identity_backward_hook()
        for hook_name in self._get_mlp_act_hook_names():
            try:
                handle = self.model.add_hook(
                    hook_name,
                    hook_fn,
                    dir="bwd",
                    is_permanent=False,
                )
                self._hook_handles.append(handle)
            except Exception as exc:
                if self.strict:
                    raise RuntimeError(f"Failed to register identity hook: {hook_name}") from exc

    def _register_ah_hooks(self) -> None:
        """Register AH-rule backward hooks on attention pattern hook points."""
        for hook_name in self._get_attn_pattern_hook_names():
            try:
                # AH-rule: treat attention weights as constants
                # We zero out the gradient flowing through the pattern
                def ah_hook(grad: torch.Tensor) -> torch.Tensor:
                    return torch.zeros_like(grad)
                handle = self.model.add_hook(
                    hook_name,
                    ah_hook,
                    dir="bwd",
                    is_permanent=False,
                )
                self._hook_handles.append(handle)
            except Exception as exc:
                if self.strict:
                    raise RuntimeError(f"Failed to register AH hook: {hook_name}") from exc

    def _register_half_hooks(self) -> None:
        """Register Half-rule backward hooks on gated MLP hook points."""
        hook_fn = _make_half_backward_hook(split_factor=0.5)
        for hook_name in self._get_mlp_gate_hook_names():
            try:
                handle = self.model.add_hook(
                    hook_name,
                    hook_fn,
                    dir="bwd",
                    is_permanent=False,
                )
                self._hook_handles.append(handle)
            except Exception as exc:
                if self.strict:
                    raise RuntimeError(f"Failed to register half hook: {hook_name}") from exc

    def register_all_hooks(self) -> None:
        """Register all LRP backward hooks according to the configured rule set."""
        if "ln" in self.rules:
            self._register_ln_hooks()
        if "identity" in self.rules:
            self._register_identity_hooks()
        if "ah" in self.rules:
            self._register_ah_hooks()
        if "half" in self.rules:
            self._register_half_hooks()
        # 0-rule (zero) requires no special hook; standard gradient = GI

    def remove_all_hooks(self) -> None:
        """Remove all registered backward hooks and clear VRAM."""
        self.model.reset_hooks(including_permanent=False)
        self._hook_handles.clear()
        self._grad_stores.clear()
        if self.verbose:
            print("[ReIPHookManager] All hooks removed.")

    # ------------------------------------------------------------------
    # Context manager interface
    # ------------------------------------------------------------------

    @contextmanager
    def active(self):
        """
        Context manager that registers hooks on entry and removes them on exit.

        Guarantees clean hook lifecycle even if an exception occurs during
        the analysis, preventing VRAM leaks in long-running sessions.
        """
        self.register_all_hooks()
        try:
            yield self
        finally:
            self.remove_all_hooks()

    # ------------------------------------------------------------------
    # Gradient extraction
    # ------------------------------------------------------------------

    def extract_relevance_scores(
        self,
        cache: Any,  # transformer_lens.ActivationCache
        component: str = "mlp_out",
    ) -> Dict[str, torch.Tensor]:
        """
        Extract LRP relevance scores from the activation cache after backward pass.

        Args:
            cache: TransformerLens ActivationCache populated by run_with_cache.
            component: Component type to extract scores for.
                       Options: "mlp_out", "attn_out", "resid_post"

        Returns:
            Dictionary mapping hook point names to relevance score tensors.
        """
        scores: Dict[str, torch.Tensor] = {}
        n_layers = self.model.cfg.n_layers

        for layer in range(n_layers):
            if component == "mlp_out":
                key = f"blocks.{layer}.hook_mlp_out"
            elif component == "attn_out":
                key = f"blocks.{layer}.hook_attn_out"
            elif component == "resid_post":
                key = f"blocks.{layer}.hook_resid_post"
            else:
                continue

            if key in cache:
                tensor = cache[key]
                if tensor.grad is not None:
                    scores[key] = tensor.grad.detach().clone()

        return scores
