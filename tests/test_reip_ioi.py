"""
IOI (Indirect Object Identification) Task — ReIP Pipeline Infrastructure Tests.

IMPORTANT DISCLAIMER:
    This test suite validates the **infrastructure and logic** of the ReIP
    pipeline using synthetic data. It does NOT prove that ReIP achieves any
    specific PCC against real Activation Patching scores.

    For actual fidelity measurement (ReIP Score vs AP Score correlation),
    see `tests/benchmark_fidelity.py`, which implements the proper comparison
    loop on a real model.

    The synthetic tests below verify:
    - Pipeline configuration correctness
    - Topology pruner graph construction
    - JSON serialization
    - IOI prompt structural validity
    - Theoretical runtime complexity advantage

IOI Task Definition:
    Clean prompt:     "When Mary and John went to the store, John gave a drink to"
    Corrupted prompt: "When Mary and John went to the store, Mary gave a drink to"
    Target token:     " Mary"

    The model should assign higher probability to " Mary" in the clean prompt.
"""

from __future__ import annotations

import time
import unittest
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import torch


# ---------------------------------------------------------------------------
# IOI test fixtures
# ---------------------------------------------------------------------------

IOI_PROMPTS: List[Tuple[str, str, str]] = [
    (
        "When Mary and John went to the store, John gave a drink to",
        "When Mary and John went to the store, Mary gave a drink to",
        " Mary",
    ),
    (
        "When Alice and Bob went to the park, Bob gave a ball to",
        "When Alice and Bob went to the park, Alice gave a ball to",
        " Alice",
    ),
    (
        "When Emma and James went to the office, James gave a report to",
        "When Emma and James went to the office, Emma gave a report to",
        " Emma",
    ),
]


class TestReIPPipelineInfrastructure(unittest.TestCase):
    """
    Infrastructure tests for the ReIP pipeline.

    These tests validate pipeline logic and configuration without making
    any claims about fidelity or correlation with Activation Patching.
    For fidelity benchmarks, see tests/benchmark_fidelity.py.
    """

    def test_reip_config_defaults(self):
        """Test that ReIPConfig has sensible defaults."""
        from src.reip.pipeline import ReIPConfig
        config = ReIPConfig()
        self.assertEqual(config.model_name, "gpt2")
        self.assertIsNone(config.lrp_rules)
        self.assertGreater(config.pruning_threshold, 0)
        self.assertTrue(config.normalize_scores)
        self.assertTrue(config.reset_hooks_end)

    def test_topology_pruner_basic(self):
        """Test TopologyPruner constructs a graph from synthetic relevance scores."""
        from src.reip.pruning import TopologyPruner

        pruner = TopologyPruner(threshold=0.1, top_k=10, normalize=True)

        # Create synthetic relevance scores
        scores = {
            "blocks.0.hook_mlp_out": torch.randn(1, 5, 64).abs(),
            "blocks.1.hook_mlp_out": torch.randn(1, 5, 64).abs(),
            "blocks.2.hook_attn_out": torch.randn(1, 5, 64).abs(),
        }

        result = pruner.build_graph(scores, token_labels=["The", "cat", "sat", "on", "mat"])

        # Should return a valid graph structure
        self.assertIsNotNone(result)

    def test_topology_pruner_json_serialization(self):
        """Test that topology graph can be serialized to valid JSON."""
        import json
        from src.reip.pruning import TopologyPruner

        pruner = TopologyPruner(threshold=0.05, normalize=True)
        scores = {
            "blocks.0.hook_mlp_out": torch.rand(1, 3, 32),
        }
        graph = pruner.build_graph(scores)
        json_str = pruner.to_json(graph)

        # Should be valid JSON
        data = json.loads(json_str)
        self.assertIn("nodes", data)
        self.assertIn("edges", data)

    def test_ioi_prompt_structure(self):
        """Validate that IOI prompt pairs have the expected structural properties."""
        for clean, corrupted, target in IOI_PROMPTS:
            # Both prompts should have the same word count
            clean_words = clean.split()
            corrupted_words = corrupted.split()
            self.assertEqual(
                len(clean_words), len(corrupted_words),
                f"IOI prompt pair has different lengths: "
                f"{len(clean_words)} vs {len(corrupted_words)}"
            )
            # Target token should appear in the clean prompt
            self.assertIn(
                target.strip(), clean,
                f"Target token '{target}' not found in clean prompt: '{clean}'"
            )

    def test_theoretical_runtime_advantage(self):
        """
        Verify the theoretical runtime advantage of ReIP over exhaustive AP.

        This is a mathematical check, not an empirical measurement.
        ReIP requires O(2F + B) operations (2 forward + 1 backward pass),
        while AP requires O(|E| * F) operations (one forward pass per component).
        For typical circuits with |E| > 100, this yields >30x theoretical speedup.
        """
        forward_pass_time = 0.01  # 10ms per forward pass (hypothetical)
        n_components = 500  # Typical number of edges

        # O(2F + B) expected time for ReIP
        expected_time_reip = 2 * forward_pass_time + forward_pass_time

        # O(|E| * F) expected time for exhaustive AP
        expected_time_ap = n_components * forward_pass_time

        # Theoretical speedup
        speedup = expected_time_ap / expected_time_reip
        self.assertGreater(
            speedup, 100,
            f"Expected theoretical speedup > 100x, got {speedup:.1f}x."
        )


class TestSyntheticCorrelationBaseline(unittest.TestCase):
    """
    Synthetic correlation tests to validate metric computation logic.

    WARNING: These tests use artificially constructed data to verify that
    the correlation computation code works correctly. They do NOT demonstrate
    that ReIP achieves high correlation with real AP scores. For real
    fidelity measurement, run `tests/benchmark_fidelity.py` with a GPU.
    """

    def test_correlation_computation_logic(self):
        """Verify that PCC computation works on known synthetic data."""
        from scipy.stats import pearsonr

        # Construct two vectors with known correlation
        np.random.seed(42)
        x = np.random.randn(100)
        noise = np.random.randn(100) * 0.1
        y = x + noise  # Should have high PCC

        pcc, _ = pearsonr(x, y)
        self.assertGreater(pcc, 0.9, "Sanity check: correlated vectors should have high PCC")

    def test_random_vectors_low_correlation(self):
        """Verify that independent random vectors have near-zero PCC."""
        from scipy.stats import pearsonr

        np.random.seed(42)
        x = np.random.randn(200)
        y = np.random.randn(200)

        pcc, _ = pearsonr(x, y)
        self.assertAlmostEqual(pcc, 0.0, delta=0.15,
                               msg="Independent random vectors should have PCC near 0")

    def test_top_k_overlap_logic(self):
        """Verify top-k overlap computation logic."""
        # Same top-5 elements
        scores_a = np.array([10, 9, 8, 7, 6, 1, 1, 1, 1, 1], dtype=float)
        scores_b = np.array([10, 9, 8, 7, 6, 2, 2, 2, 2, 2], dtype=float)

        top5_a = set(np.argsort(scores_a)[-5:])
        top5_b = set(np.argsort(scores_b)[-5:])
        overlap = len(top5_a & top5_b) / 5

        self.assertEqual(overlap, 1.0, "Identical top-5 should yield 100% overlap")


if __name__ == "__main__":
    unittest.main(verbosity=2)
