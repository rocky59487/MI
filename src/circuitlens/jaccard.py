"""
Jaccard Similarity Matrix for Circuit-Based Clustering.

This module computes pairwise Jaccard similarity between input samples based
on the set of attention head-token pairs that activate a target Transcoder
feature. The resulting similarity matrix quantifies how much different input
samples share the same underlying computational circuit when triggering the
same feature.

Jaccard similarity between samples A and B:
    J(A, B) = |circuit(A) ∩ circuit(B)| / |circuit(A) ∪ circuit(B)|

Where circuit(X) = set of (attention_head, token_position) pairs with
attribution score above threshold for sample X.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np

from .jacobian import JacobianResult


class JaccardSimilarity:
    """
    Computes pairwise Jaccard similarity matrices from Jacobian analysis results.

    The similarity matrix is used as input to the DBSCAN clustering algorithm
    in CircuitClusterer to decompose polysemantic features into monosemantic
    sub-clusters.

    Args:
        attribution_threshold: Minimum normalized attribution score for a
                               (head, position) pair to be included in a
                               sample's circuit set.
        position_bins: If set, discretize token positions into bins to improve
                       robustness to minor positional variations.
    """

    def __init__(
        self,
        attribution_threshold: float = 0.1,
        position_bins: Optional[int] = None,
    ):
        self.attribution_threshold = attribution_threshold
        self.position_bins = position_bins

    def compute_matrix(
        self,
        jacobian_results: List[JacobianResult],
    ) -> np.ndarray:
        """
        Compute the pairwise Jaccard similarity matrix for a list of samples.

        Args:
            jacobian_results: List of JacobianResult instances, one per input sample.
                              Each result contains top_attention_heads which defines
                              the circuit set for that sample.

        Returns:
            Symmetric numpy array of shape (n_samples, n_samples) with values in [0, 1].
            Entry [i, j] = Jaccard similarity between samples i and j.
            Diagonal entries are 1.0.
        """
        n = len(jacobian_results)
        if n == 0:
            return np.array([[]])

        # Extract circuit sets for each sample
        circuit_sets: List[Set[Tuple]] = []
        for result in jacobian_results:
            circuit_set = self._extract_circuit_set(result)
            circuit_sets.append(circuit_set)

        # Compute pairwise Jaccard similarity
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            matrix[i, i] = 1.0
            for j in range(i + 1, n):
                sim = self._jaccard(circuit_sets[i], circuit_sets[j])
                matrix[i, j] = sim
                matrix[j, i] = sim

        return matrix

    def compute_distance_matrix(
        self,
        jacobian_results: List[JacobianResult],
    ) -> np.ndarray:
        """
        Compute pairwise Jaccard distance matrix (1 - Jaccard similarity).

        This is the format expected by DBSCAN with metric="precomputed".

        Args:
            jacobian_results: List of JacobianResult instances.

        Returns:
            Symmetric numpy array of shape (n_samples, n_samples) with values in [0, 1].
            Entry [i, j] = 0 means identical circuits; 1 means no overlap.
        """
        sim_matrix = self.compute_matrix(jacobian_results)
        return 1.0 - sim_matrix

    def _extract_circuit_set(self, result: JacobianResult) -> Set[Tuple]:
        """
        Extract the circuit set (frozenset of relevant head-token pairs) from
        a JacobianResult.

        The circuit set is defined as all (layer, head, position) tuples whose
        attribution score exceeds the threshold, after normalization.
        """
        if not result.top_attention_heads:
            return set()

        # Normalize scores within this sample
        scores = [abs(score) for _, _, _, score in result.top_attention_heads]
        max_score = max(scores) if scores else 1.0
        if max_score == 0:
            return set()

        circuit_set = set()
        for layer, head, position, score in result.top_attention_heads:
            normalized_score = abs(score) / max_score
            if normalized_score >= self.attribution_threshold:
                # Optionally bin positions for robustness
                if self.position_bins is not None and position >= 0:
                    binned_pos = position // self.position_bins
                else:
                    binned_pos = position
                circuit_set.add((layer, head, binned_pos))

        return circuit_set

    @staticmethod
    def _jaccard(set_a: Set, set_b: Set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set_a and not set_b:
            return 1.0  # Both empty: identical circuits
        if not set_a or not set_b:
            return 0.0  # One empty: no overlap

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def get_circuit_overlap_details(
        self,
        result_a: JacobianResult,
        result_b: JacobianResult,
    ) -> dict:
        """
        Compute detailed circuit overlap statistics between two samples.

        Returns:
            Dictionary with keys: jaccard_score, shared_pairs, unique_to_a,
            unique_to_b, total_pairs_a, total_pairs_b.
        """
        set_a = self._extract_circuit_set(result_a)
        set_b = self._extract_circuit_set(result_b)

        shared = set_a & set_b
        unique_a = set_a - set_b
        unique_b = set_b - set_a

        return {
            "jaccard_score": self._jaccard(set_a, set_b),
            "shared_pairs": len(shared),
            "unique_to_a": len(unique_a),
            "unique_to_b": len(unique_b),
            "total_pairs_a": len(set_a),
            "total_pairs_b": len(set_b),
        }
