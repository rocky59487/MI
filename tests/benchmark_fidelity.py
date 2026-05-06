"""
Fidelity Benchmark: ReIP Score vs Activation Patching Score Correlation.

This module implements the ground-truth fidelity validation loop that
measures the Pearson/Spearman correlation and Top-k overlap between
ReIP attribution scores and exhaustive Activation Patching (AP) scores.

The benchmark proves that ReIP scores are a faithful approximation of
the causal effect measured by AP, not merely that "intervention has an effect."

Protocol:
    1. For each component (node/edge) in the model's computation graph:
       a. Compute the AP score via single-component patching (ground truth).
       b. Compute the ReIP score via the grad_delta * act_delta approximation.
    2. Compute Pearson correlation coefficient (PCC) between the two score vectors.
    3. Compute Spearman rank correlation coefficient.
    4. Compute Top-k overlap (k=5, 10, 20) between highest-scoring components.

Targets:
    - PCC(ReIP, AP) — to be measured (no pre-claimed threshold)
    - Spearman(ReIP, AP) — to be measured
    - Top-5 overlap — to be measured
    - Top-10 overlap — to be measured

NOTE: This benchmark requires a real model and GPU to produce meaningful
results. The unit test version uses synthetic data to validate the
benchmark infrastructure itself.
"""

from __future__ import annotations

import time
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class FidelityResult:
    """
    Container for fidelity benchmark results.

    Attributes:
        model_name: Name of the model tested.
        task_name: Name of the evaluation task (e.g., "IOI").
        n_components: Number of components compared.
        reip_scores: ReIP attribution scores for each component.
        ap_scores: Activation Patching scores for each component (ground truth).
        pearson_pcc: Pearson correlation coefficient.
        spearman_rho: Spearman rank correlation coefficient.
        top_k_overlaps: Dict mapping k to overlap ratio.
        reip_runtime_s: Time to compute all ReIP scores.
        ap_runtime_s: Time to compute all AP scores.
        scoring_formula: Name of the scoring formula used.
    """
    model_name: str
    task_name: str
    n_components: int
    reip_scores: np.ndarray
    ap_scores: np.ndarray
    pearson_pcc: float = 0.0
    spearman_rho: float = 0.0
    top_k_overlaps: Dict[int, float] = field(default_factory=dict)
    reip_runtime_s: float = 0.0
    ap_runtime_s: float = 0.0
    scoring_formula: str = "grad_delta_x_act_delta"

    def compute_metrics(self) -> None:
        """Compute all correlation metrics from score vectors."""
        from scipy.stats import pearsonr, spearmanr

        if len(self.reip_scores) < 3:
            return

        self.pearson_pcc, _ = pearsonr(self.reip_scores, self.ap_scores)
        self.spearman_rho, _ = spearmanr(self.reip_scores, self.ap_scores)

        for k in [5, 10, 20]:
            if k > len(self.reip_scores):
                continue
            reip_top_k = set(np.argsort(np.abs(self.reip_scores))[-k:])
            ap_top_k = set(np.argsort(np.abs(self.ap_scores))[-k:])
            overlap = len(reip_top_k & ap_top_k) / k
            self.top_k_overlaps[k] = overlap

    def summary(self) -> str:
        """Return a human-readable summary of the benchmark results."""
        lines = [
            f"=== Fidelity Benchmark: {self.model_name} / {self.task_name} ===",
            f"Scoring formula: {self.scoring_formula}",
            f"Components tested: {self.n_components}",
            f"Pearson PCC:  {self.pearson_pcc:.4f}",
            f"Spearman rho: {self.spearman_rho:.4f}",
        ]
        for k, overlap in sorted(self.top_k_overlaps.items()):
            lines.append(f"Top-{k} overlap: {overlap:.2%}")
        lines.append(f"ReIP runtime:  {self.reip_runtime_s:.3f}s")
        lines.append(f"AP runtime:    {self.ap_runtime_s:.3f}s")
        if self.ap_runtime_s > 0:
            speedup = self.ap_runtime_s / max(self.reip_runtime_s, 1e-6)
            lines.append(f"Speedup:       {speedup:.1f}x")
        return "\n".join(lines)


def compute_activation_patching_scores(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    target_token_idx: int,
    target_token_id: int,
    hook_points: List[str],
) -> Tuple[np.ndarray, float]:
    """
    Compute ground-truth Activation Patching scores for each hook point.

    For each component (hook point), replaces the clean activation with
    the corrupted activation and measures the change in logit difference.

    Args:
        model: TransformerLens HookedTransformer instance.
        clean_tokens: Clean input token tensor, shape (1, seq_len).
        corrupted_tokens: Corrupted input token tensor, shape (1, seq_len).
        target_token_idx: Position index of the target token prediction.
        target_token_id: Token ID of the target token.
        hook_points: List of hook point names to patch.

    Returns:
        Tuple of (scores array, runtime in seconds).
    """
    start_time = time.time()

    # Get clean and corrupted caches
    _, clean_cache = model.run_with_cache(clean_tokens)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    # Get clean logit for target token
    clean_logits = model(clean_tokens)
    clean_target_logit = clean_logits[0, target_token_idx, target_token_id].item()

    scores = np.zeros(len(hook_points))

    for i, hook_name in enumerate(hook_points):
        if hook_name not in clean_cache or hook_name not in corrupted_cache:
            continue

        def patch_hook(activation, hook, corrupted_act=corrupted_cache[hook_name]):
            return corrupted_act

        # Run with the single component patched
        patched_logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(hook_name, patch_hook)],
        )
        patched_target_logit = patched_logits[0, target_token_idx, target_token_id].item()

        # AP score = change in target logit when component is patched
        scores[i] = clean_target_logit - patched_target_logit

    runtime = time.time() - start_time
    return scores, runtime


def compute_reip_scores(
    model: Any,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    target_token_idx: int,
    target_token_id: int,
    hook_points: List[str],
    scoring_formula: str = "grad_delta_x_act_delta",
) -> Tuple[np.ndarray, float]:
    """
    Compute ReIP attribution scores using the specified scoring formula.

    Supported formulas:
        - "grad_delta_x_act_delta": (grad_clean - grad_corrupted) * (act_clean - act_corrupted)
        - "corr_grad_x_act_delta": grad_corrupted * (act_clean - act_corrupted)
        - "clean_grad_x_act_delta": grad_clean * (act_clean - act_corrupted)
        - "half_sum_x_act_delta": 0.5 * (grad_clean + grad_corrupted) * (act_clean - act_corrupted)

    Args:
        model: TransformerLens HookedTransformer instance.
        clean_tokens: Clean input token tensor.
        corrupted_tokens: Corrupted input token tensor.
        target_token_idx: Position index of the target token prediction.
        target_token_id: Token ID of the target token.
        hook_points: List of hook point names to score.
        scoring_formula: Which formula to use for scoring.

    Returns:
        Tuple of (scores array, runtime in seconds).
    """
    start_time = time.time()

    # Use run_with_hooks + retain_grad to properly capture gradients
    captured_clean = {}
    captured_corrupted = {}

    def _make_retain_hook(store, name):
        def hook_fn(activation, hook):
            activation.requires_grad_(True)
            activation.retain_grad()
            store[name] = activation
            return activation
        return hook_fn

    # Clean forward pass with gradient-capturing hooks
    fwd_hooks_clean = [(hp, _make_retain_hook(captured_clean, hp)) for hp in hook_points]
    model.zero_grad()
    clean_logits = model.run_with_hooks(clean_tokens, return_type="logits", fwd_hooks=fwd_hooks_clean)
    target_logit_clean = clean_logits[0, target_token_idx, target_token_id]
    target_logit_clean.backward()

    # Collect clean gradients
    clean_grads = {}
    for hook_name in hook_points:
        if hook_name in captured_clean and captured_clean[hook_name].grad is not None:
            clean_grads[hook_name] = captured_clean[hook_name].grad.clone()

    # Corrupted forward pass with gradient-capturing hooks
    fwd_hooks_corrupted = [(hp, _make_retain_hook(captured_corrupted, hp)) for hp in hook_points]
    model.zero_grad()
    corrupted_logits = model.run_with_hooks(corrupted_tokens, return_type="logits", fwd_hooks=fwd_hooks_corrupted)
    target_logit_corrupted = corrupted_logits[0, target_token_idx, target_token_id]
    target_logit_corrupted.backward()

    corrupted_grads = {}
    for hook_name in hook_points:
        if hook_name in captured_corrupted and captured_corrupted[hook_name].grad is not None:
            corrupted_grads[hook_name] = captured_corrupted[hook_name].grad.clone()

    # Compute scores based on formula
    scores = np.zeros(len(hook_points))

    for i, hook_name in enumerate(hook_points):
        if hook_name not in captured_clean or hook_name not in captured_corrupted:
            continue
        act_clean = captured_clean[hook_name].detach()
        act_corrupted = captured_corrupted[hook_name].detach()
        act_delta = act_clean - act_corrupted

        grad_clean = clean_grads.get(hook_name, torch.zeros_like(act_clean))
        grad_corrupted = corrupted_grads.get(hook_name, torch.zeros_like(act_clean))

        if scoring_formula == "grad_delta_x_act_delta":
            grad_delta = grad_clean - grad_corrupted
            score_tensor = (grad_delta * act_delta).sum()
        elif scoring_formula == "corr_grad_x_act_delta":
            score_tensor = (grad_corrupted * act_delta).sum()
        elif scoring_formula == "clean_grad_x_act_delta":
            score_tensor = (grad_clean * act_delta).sum()
        elif scoring_formula == "half_sum_x_act_delta":
            grad_avg = 0.5 * (grad_clean + grad_corrupted)
            score_tensor = (grad_avg * act_delta).sum()
        else:
            raise ValueError(f"Unknown scoring formula: {scoring_formula}")

        scores[i] = score_tensor.item()

    runtime = time.time() - start_time
    return scores, runtime


def run_fidelity_benchmark(
    model: Any,
    clean_prompt: str,
    corrupted_prompt: str,
    target_token: str,
    scoring_formula: str = "grad_delta_x_act_delta",
    model_name: str = "unknown",
    task_name: str = "IOI",
) -> FidelityResult:
    """
    Run the complete fidelity benchmark comparing ReIP to AP.

    This is the main entry point for the benchmark. It:
    1. Tokenizes prompts
    2. Identifies target token position and ID
    3. Enumerates all hook points (MLP out, attention out per layer)
    4. Computes AP scores (ground truth)
    5. Computes ReIP scores (approximation)
    6. Computes correlation metrics

    Args:
        model: TransformerLens HookedTransformer instance.
        clean_prompt: Clean input prompt string.
        corrupted_prompt: Corrupted input prompt string.
        target_token: Target token string (e.g., " Mary").
        scoring_formula: ReIP scoring formula to use.
        model_name: Model name for reporting.
        task_name: Task name for reporting.

    Returns:
        FidelityResult with all metrics computed.
    """
    # Tokenize
    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    target_token_id = model.to_single_token(target_token)
    target_token_idx = clean_tokens.shape[1] - 1  # Predict next token

    # Enumerate hook points
    n_layers = model.cfg.n_layers
    hook_points = []
    for layer in range(n_layers):
        hook_points.append(f"blocks.{layer}.hook_mlp_out")
        hook_points.append(f"blocks.{layer}.hook_attn_out")

    # Compute AP scores (ground truth)
    ap_scores, ap_runtime = compute_activation_patching_scores(
        model, clean_tokens, corrupted_tokens,
        target_token_idx, target_token_id, hook_points
    )

    # Compute ReIP scores
    reip_scores, reip_runtime = compute_reip_scores(
        model, clean_tokens, corrupted_tokens,
        target_token_idx, target_token_id, hook_points,
        scoring_formula=scoring_formula,
    )

    # Build result
    result = FidelityResult(
        model_name=model_name,
        task_name=task_name,
        n_components=len(hook_points),
        reip_scores=reip_scores,
        ap_scores=ap_scores,
        reip_runtime_s=reip_runtime,
        ap_runtime_s=ap_runtime,
        scoring_formula=scoring_formula,
    )
    result.compute_metrics()
    return result


# ---------------------------------------------------------------------------
# Unit tests for benchmark infrastructure (synthetic data)
# ---------------------------------------------------------------------------

class TestFidelityBenchmarkInfrastructure(unittest.TestCase):
    """
    Tests for the fidelity benchmark infrastructure using synthetic data.

    These tests validate that the benchmark code correctly computes
    correlation metrics and top-k overlaps, without requiring a real model.
    """

    def test_perfect_correlation(self):
        """Identical score vectors should yield PCC = 1.0."""
        scores = np.random.randn(50)
        result = FidelityResult(
            model_name="test",
            task_name="synthetic",
            n_components=50,
            reip_scores=scores,
            ap_scores=scores,
        )
        result.compute_metrics()
        self.assertAlmostEqual(result.pearson_pcc, 1.0, places=4)
        self.assertAlmostEqual(result.spearman_rho, 1.0, places=4)
        self.assertEqual(result.top_k_overlaps[5], 1.0)

    def test_anti_correlation(self):
        """Negated score vectors should yield PCC = -1.0."""
        scores = np.random.randn(50)
        result = FidelityResult(
            model_name="test",
            task_name="synthetic",
            n_components=50,
            reip_scores=scores,
            ap_scores=-scores,
        )
        result.compute_metrics()
        self.assertAlmostEqual(result.pearson_pcc, -1.0, places=4)

    def test_random_correlation_near_zero(self):
        """Independent random vectors should have PCC near 0."""
        np.random.seed(42)
        result = FidelityResult(
            model_name="test",
            task_name="synthetic",
            n_components=1000,
            reip_scores=np.random.randn(1000),
            ap_scores=np.random.randn(1000),
        )
        result.compute_metrics()
        self.assertAlmostEqual(result.pearson_pcc, 0.0, delta=0.1)

    def test_top_k_overlap_computation(self):
        """Top-k overlap should be correctly computed."""
        # Scores where top-5 are the same
        base = np.zeros(20)
        base[0:5] = np.array([10, 9, 8, 7, 6])
        noise = base + np.random.randn(20) * 0.01
        noise[0:5] = base[0:5]  # Keep top-5 identical

        result = FidelityResult(
            model_name="test",
            task_name="synthetic",
            n_components=20,
            reip_scores=base,
            ap_scores=noise,
        )
        result.compute_metrics()
        self.assertEqual(result.top_k_overlaps[5], 1.0)

    def test_summary_format(self):
        """Summary should be a non-empty string with key metrics."""
        result = FidelityResult(
            model_name="gpt2",
            task_name="IOI",
            n_components=24,
            reip_scores=np.random.randn(24),
            ap_scores=np.random.randn(24),
            reip_runtime_s=0.05,
            ap_runtime_s=5.0,
        )
        result.compute_metrics()
        summary = result.summary()
        self.assertIn("Pearson PCC", summary)
        self.assertIn("Spearman rho", summary)
        self.assertIn("Speedup", summary)

    def test_insufficient_data_graceful(self):
        """Benchmark should handle very small component counts gracefully."""
        result = FidelityResult(
            model_name="test",
            task_name="synthetic",
            n_components=2,
            reip_scores=np.array([1.0, 2.0]),
            ap_scores=np.array([1.0, 2.0]),
        )
        # Should not raise
        result.compute_metrics()


if __name__ == "__main__":
    unittest.main(verbosity=2)
