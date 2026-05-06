"""
ReIP Scoring Formula Ablation Study.

This module implements a systematic ablation experiment comparing 4 different
scoring formulas for the ReIP attribution pipeline, measuring which formula
achieves the highest correlation with ground-truth Activation Patching scores.

Formulas under test:
    1. grad_delta_x_act_delta:  (grad_clean - grad_corrupted) * (act_clean - act_corrupted)
    2. corr_grad_x_act_delta:   grad_corrupted * (act_clean - act_corrupted)
    3. clean_grad_x_act_delta:  grad_clean * (act_clean - act_corrupted)
    4. half_sum_x_act_delta:    0.5 * (grad_clean + grad_corrupted) * (act_clean - act_corrupted)

Formula (2) is the standard Attribution Patching (AtP) formula, known to
suffer from gradient pathology. Formula (1) is the "gradient delta" variant
that may partially mitigate this issue. Formula (4) is a trapezoidal
approximation. Formula (3) uses only the clean gradient.

Usage:
    # Run ablation on a real model (requires GPU + transformer-lens):
    python -m tests.ablation_scoring --model gpt2 --task IOI

    # Run synthetic ablation (no GPU required):
    python -m tests.ablation_scoring --synthetic
"""

from __future__ import annotations

import argparse
import sys
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from tests.benchmark_fidelity import (
    FidelityResult,
    compute_activation_patching_scores,
    compute_reip_scores,
)


SCORING_FORMULAS = [
    "grad_delta_x_act_delta",
    "corr_grad_x_act_delta",
    "clean_grad_x_act_delta",
    "half_sum_x_act_delta",
]


@dataclass
class AblationResult:
    """
    Container for ablation study results across all scoring formulas.

    Attributes:
        model_name: Name of the model tested.
        task_name: Name of the evaluation task.
        results: Dict mapping formula name to FidelityResult.
        best_formula: Formula with highest Pearson PCC.
        ranking: List of (formula, pcc) sorted by PCC descending.
    """
    model_name: str
    task_name: str
    results: Dict[str, FidelityResult] = field(default_factory=dict)
    best_formula: str = ""
    ranking: List[tuple] = field(default_factory=list)

    def compute_ranking(self) -> None:
        """Compute the ranking of formulas by Pearson PCC."""
        self.ranking = sorted(
            [(name, r.pearson_pcc) for name, r in self.results.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        if self.ranking:
            self.best_formula = self.ranking[0][0]

    def summary(self) -> str:
        """Return a formatted summary table of the ablation results."""
        lines = [
            f"=== Scoring Formula Ablation: {self.model_name} / {self.task_name} ===",
            "",
            f"{'Formula':<30} {'PCC':>8} {'Spearman':>10} {'Top-5':>8} {'Top-10':>8}",
            "-" * 70,
        ]
        for name, pcc in self.ranking:
            r = self.results[name]
            top5 = r.top_k_overlaps.get(5, 0.0)
            top10 = r.top_k_overlaps.get(10, 0.0)
            marker = " <-- BEST" if name == self.best_formula else ""
            lines.append(
                f"{name:<30} {pcc:>8.4f} {r.spearman_rho:>10.4f} "
                f"{top5:>7.2%} {top10:>7.2%}{marker}"
            )
        lines.append("")
        lines.append(f"Best formula: {self.best_formula}")
        return "\n".join(lines)


def run_ablation_real_model(
    model: Any,
    clean_prompt: str,
    corrupted_prompt: str,
    target_token: str,
    model_name: str = "unknown",
    task_name: str = "IOI",
) -> AblationResult:
    """
    Run the ablation study on a real model, comparing all 4 scoring formulas
    against ground-truth Activation Patching.

    Args:
        model: TransformerLens HookedTransformer instance.
        clean_prompt: Clean input prompt.
        corrupted_prompt: Corrupted input prompt.
        target_token: Target token string.
        model_name: Model name for reporting.
        task_name: Task name for reporting.

    Returns:
        AblationResult with all formulas compared.
    """
    from tests.benchmark_fidelity import run_fidelity_benchmark

    ablation = AblationResult(model_name=model_name, task_name=task_name)

    for formula in SCORING_FORMULAS:
        result = run_fidelity_benchmark(
            model=model,
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            target_token=target_token,
            scoring_formula=formula,
            model_name=model_name,
            task_name=task_name,
        )
        ablation.results[formula] = result

    ablation.compute_ranking()
    return ablation


def run_ablation_synthetic(n_components: int = 100) -> AblationResult:
    """
    Run a synthetic ablation study to validate the infrastructure.

    Simulates different formula behaviors:
    - grad_delta_x_act_delta: moderate noise (simulates partial pathology fix)
    - corr_grad_x_act_delta: high noise (simulates AtP gradient pathology)
    - clean_grad_x_act_delta: moderate-low noise
    - half_sum_x_act_delta: low noise (simulates trapezoidal approximation)

    Returns:
        AblationResult with synthetic data.
    """
    np.random.seed(42)

    # Ground truth AP scores (sparse distribution)
    ap_scores = np.random.exponential(scale=0.1, size=n_components)
    top_indices = np.random.choice(n_components, size=10, replace=False)
    ap_scores[top_indices] = np.random.uniform(0.5, 1.0, size=10)

    # Simulate different formula noise levels
    formula_noise = {
        "grad_delta_x_act_delta": 0.15,
        "corr_grad_x_act_delta": 0.80,   # High noise = AtP pathology
        "clean_grad_x_act_delta": 0.20,
        "half_sum_x_act_delta": 0.10,
    }

    ablation = AblationResult(model_name="synthetic", task_name="synthetic_IOI")

    for formula, noise_level in formula_noise.items():
        np.random.seed(hash(formula) % (2**32))
        noise = np.random.randn(n_components) * noise_level
        reip_scores = ap_scores + noise

        result = FidelityResult(
            model_name="synthetic",
            task_name="synthetic_IOI",
            n_components=n_components,
            reip_scores=reip_scores,
            ap_scores=ap_scores,
            scoring_formula=formula,
        )
        result.compute_metrics()
        ablation.results[formula] = result

    ablation.compute_ranking()
    return ablation


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestAblationInfrastructure(unittest.TestCase):
    """Tests for the ablation study infrastructure."""

    def test_synthetic_ablation_runs(self):
        """Synthetic ablation should complete without errors."""
        result = run_ablation_synthetic(n_components=50)
        self.assertEqual(len(result.results), 4)
        self.assertIn(result.best_formula, SCORING_FORMULAS)

    def test_ranking_is_sorted(self):
        """Ranking should be sorted by PCC descending."""
        result = run_ablation_synthetic(n_components=100)
        pccs = [pcc for _, pcc in result.ranking]
        self.assertEqual(pccs, sorted(pccs, reverse=True))

    def test_corr_grad_has_lowest_pcc(self):
        """
        In synthetic simulation, corr_grad (standard AtP) should have the
        lowest PCC due to simulated gradient pathology.
        """
        result = run_ablation_synthetic(n_components=200)
        corr_grad_pcc = result.results["corr_grad_x_act_delta"].pearson_pcc
        best_pcc = result.results[result.best_formula].pearson_pcc
        self.assertLess(
            corr_grad_pcc, best_pcc,
            "Standard AtP formula should perform worse than alternatives "
            "due to gradient pathology simulation."
        )

    def test_summary_format(self):
        """Summary should contain all formula names."""
        result = run_ablation_synthetic()
        summary = result.summary()
        for formula in SCORING_FORMULAS:
            self.assertIn(formula, summary)
        self.assertIn("BEST", summary)

    def test_all_formulas_present(self):
        """All 4 formulas should be present in results."""
        result = run_ablation_synthetic()
        for formula in SCORING_FORMULAS:
            self.assertIn(formula, result.results)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ReIP Scoring Formula Ablation Study")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--task", type=str, default="IOI", help="Task name")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic ablation")
    args = parser.parse_args()

    if args.synthetic:
        print("Running synthetic ablation study...")
        result = run_ablation_synthetic(n_components=200)
        print(result.summary())
    else:
        print(f"Running real ablation on {args.model}...")
        print("NOTE: Requires GPU and transformer-lens installed.")
        try:
            from transformer_lens import HookedTransformer
            model = HookedTransformer.from_pretrained(args.model)
            result = run_ablation_real_model(
                model=model,
                clean_prompt="When Mary and John went to the store, John gave a drink to",
                corrupted_prompt="When Mary and John went to the store, Mary gave a drink to",
                target_token=" Mary",
                model_name=args.model,
                task_name=args.task,
            )
            print(result.summary())
        except ImportError:
            print("ERROR: transformer-lens not installed. Use --synthetic for testing.")
            sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        unittest.main(verbosity=2)
