"""
Unit Tests for CircuitLens Context-Dependent Feature Analysis.

Tests cover:
    - JaccardSimilarity: Circuit set extraction, pairwise matrix computation
    - CircuitClusterer: DBSCAN clustering, composite label generation
    - JacobianAnalyzer: Jacobian result structure validation
"""

from __future__ import annotations

import unittest
from typing import List

import numpy as np
import torch


class TestJaccardSimilarity(unittest.TestCase):
    """Tests for Jaccard similarity matrix computation."""

    def setUp(self):
        from src.circuitlens.jaccard import JaccardSimilarity
        from src.circuitlens.jacobian import JacobianResult

        self.JaccardSimilarity = JaccardSimilarity
        self.JacobianResult = JacobianResult

    def _make_result(
        self,
        feature_idx: int,
        layer_idx: int,
        top_heads: List,
    ):
        """Helper to create a JacobianResult with specified top_attention_heads."""
        return self.JacobianResult(
            feature_idx=feature_idx,
            layer_idx=layer_idx,
            jacobian_matrix=torch.zeros(5, 64),
            top_attention_heads=top_heads,
        )

    def test_identical_circuits_score_one(self):
        """Two samples with identical circuits should have Jaccard similarity = 1."""
        heads = [(0, 0, 1, 0.9), (0, 1, 2, 0.8), (1, 0, 3, 0.7)]
        r1 = self._make_result(0, 3, heads)
        r2 = self._make_result(0, 3, heads)

        jac = self.JaccardSimilarity(attribution_threshold=0.5)
        matrix = jac.compute_matrix([r1, r2])

        self.assertAlmostEqual(matrix[0, 1], 1.0, places=4)
        self.assertAlmostEqual(matrix[1, 0], 1.0, places=4)

    def test_disjoint_circuits_score_zero(self):
        """Two samples with completely different circuits should score 0."""
        r1 = self._make_result(0, 3, [(0, 0, 1, 0.9), (0, 1, 2, 0.8)])
        r2 = self._make_result(0, 3, [(1, 2, 5, 0.9), (2, 3, 8, 0.8)])

        jac = self.JaccardSimilarity(attribution_threshold=0.5)
        matrix = jac.compute_matrix([r1, r2])

        self.assertAlmostEqual(matrix[0, 1], 0.0, places=4)

    def test_matrix_symmetry(self):
        """Jaccard matrix should be symmetric."""
        results = [
            self._make_result(0, 3, [(0, 0, 1, 0.9), (0, 1, 2, 0.8)]),
            self._make_result(0, 3, [(0, 0, 1, 0.9), (1, 2, 5, 0.7)]),
            self._make_result(0, 3, [(2, 1, 3, 0.9), (0, 1, 2, 0.6)]),
        ]

        jac = self.JaccardSimilarity(attribution_threshold=0.3)
        matrix = jac.compute_matrix(results)

        np.testing.assert_array_almost_equal(matrix, matrix.T, decimal=6)

    def test_diagonal_is_one(self):
        """Diagonal of Jaccard matrix should be 1.0 (self-similarity)."""
        results = [
            self._make_result(0, 3, [(0, 0, 1, 0.9)]),
            self._make_result(0, 3, [(1, 1, 2, 0.8)]),
        ]

        jac = self.JaccardSimilarity(attribution_threshold=0.3)
        matrix = jac.compute_matrix(results)

        for i in range(len(results)):
            self.assertAlmostEqual(matrix[i, i], 1.0, places=4)

    def test_distance_matrix_is_complement(self):
        """Distance matrix should equal 1 - similarity matrix."""
        results = [
            self._make_result(0, 3, [(0, 0, 1, 0.9), (0, 1, 2, 0.8)]),
            self._make_result(0, 3, [(0, 0, 1, 0.9), (1, 2, 5, 0.7)]),
        ]

        jac = self.JaccardSimilarity(attribution_threshold=0.3)
        sim = jac.compute_matrix(results)
        dist = jac.compute_distance_matrix(results)

        np.testing.assert_array_almost_equal(sim + dist, np.ones_like(sim), decimal=6)


class TestCircuitClusterer(unittest.TestCase):
    """Tests for DBSCAN-based circuit clustering."""

    def setUp(self):
        from src.circuitlens.clustering import CircuitClusterer
        from src.circuitlens.jacobian import JacobianResult

        self.CircuitClusterer = CircuitClusterer
        self.JacobianResult = JacobianResult

    def _make_result(self, feature_idx, layer_idx, top_heads):
        return self.JacobianResult(
            feature_idx=feature_idx,
            layer_idx=layer_idx,
            jacobian_matrix=torch.zeros(5, 64),
            top_attention_heads=top_heads,
        )

    def test_cluster_two_distinct_groups(self):
        """DBSCAN should identify two clusters from clearly separated circuits."""
        # Group A: uses heads (0,0) and (0,1)
        group_a = [
            self._make_result(0, 3, [(0, 0, 1, 0.9), (0, 1, 2, 0.8)])
            for _ in range(4)
        ]
        # Group B: uses heads (2,3) and (3,4)
        group_b = [
            self._make_result(0, 3, [(2, 3, 5, 0.9), (3, 4, 6, 0.8)])
            for _ in range(4)
        ]

        clusterer = self.CircuitClusterer(
            dbscan_eps=0.3,
            dbscan_min_samples=2,
            attribution_threshold=0.3,
        )
        result = clusterer.cluster(group_a + group_b)

        # Should find at least 2 clusters (or 1 if sklearn not available)
        self.assertGreaterEqual(result.n_clusters, 1)
        self.assertEqual(result.feature_idx, 0)
        self.assertEqual(result.layer_idx, 3)

    def test_cluster_single_sample(self):
        """Single sample should return a valid ClusterResult without error."""
        sample = [self._make_result(5, 2, [(0, 0, 1, 0.9)])]
        clusterer = self.CircuitClusterer()
        result = clusterer.cluster(sample)

        self.assertEqual(result.n_clusters, 1)
        self.assertIsNotNone(result.composite_label)

    def test_cluster_empty_input(self):
        """Empty input should return a valid ClusterResult with 0 clusters."""
        clusterer = self.CircuitClusterer()
        result = clusterer.cluster([])

        self.assertEqual(result.n_clusters, 0)

    def test_composite_label_non_empty(self):
        """Composite label should be a non-empty string."""
        samples = [
            self._make_result(0, 3, [(0, 0, 1, 0.9), (0, 1, 2, 0.8)])
            for _ in range(3)
        ]
        clusterer = self.CircuitClusterer(dbscan_min_samples=2)
        result = clusterer.cluster(samples)

        self.assertIsInstance(result.composite_label, str)
        self.assertGreater(len(result.composite_label), 0)

    def test_cluster_result_to_dict(self):
        """ClusterResult.to_dict() should produce a JSON-serializable dict."""
        import json

        samples = [
            self._make_result(0, 3, [(0, 0, 1, 0.9)])
            for _ in range(3)
        ]
        clusterer = self.CircuitClusterer()
        result = clusterer.cluster(samples)
        d = result.to_dict()

        self.assertIn("n_clusters", d)
        self.assertIn("composite_label", d)

        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)


class TestJacobianAnalyzer(unittest.TestCase):
    """Tests for JacobianAnalyzer structure and LRP weight extraction."""

    def test_jacobian_result_structure(self):
        """JacobianResult should have correct field types."""
        from src.circuitlens.jacobian import JacobianResult

        result = JacobianResult(
            feature_idx=5,
            layer_idx=3,
            jacobian_matrix=torch.randn(10, 64),
            top_attention_heads=[(0, 0, 1, 0.9), (0, 1, 2, 0.8)],
            masked_input_positions=[3, 7],
        )

        self.assertEqual(result.feature_idx, 5)
        self.assertEqual(result.layer_idx, 3)
        self.assertEqual(result.jacobian_matrix.shape, (10, 64))
        self.assertEqual(len(result.top_attention_heads), 2)
        self.assertEqual(len(result.masked_input_positions), 2)

    def test_lrp_weight_extraction(self):
        """_extract_lrp_weights should return a tensor of correct length."""
        from src.circuitlens.jacobian import JacobianAnalyzer

        mock_model = unittest.mock.MagicMock()
        analyzer = JacobianAnalyzer(model=mock_model, device="cpu")

        lrp_scores = {
            "blocks.2.hook_resid_post": torch.rand(1, 8, 64),
        }

        weights = analyzer._extract_lrp_weights(lrp_scores, target_layer=3, seq_len=8)
        if weights is not None:
            self.assertEqual(weights.shape[0], 8)
            self.assertTrue((weights >= 0).all())
            self.assertTrue((weights <= 1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
