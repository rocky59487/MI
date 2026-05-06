"""
DBSCAN-Based Circuit Clustering for Polysemanticity Decomposition.

This module applies DBSCAN (Density-Based Spatial Clustering of Applications
with Noise) to the Jaccard distance matrix to decompose polysemantic
Transcoder features into monosemantic sub-clusters.

Each sub-cluster represents a distinct semantic context in which the feature
is activated, characterized by a unique underlying computational circuit
(set of attention head-token pairs).

Workflow:
    1. Collect Jacobian results for N input samples that activate the feature.
    2. Compute Jaccard distance matrix (N x N).
    3. Apply DBSCAN with metric="precomputed" to identify dense circuit clusters.
    4. For each cluster, generate an independent semantic description via WeightLens.
    5. Combine sub-cluster descriptions into a unified composite feature label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .jacobian import JacobianResult
from .jaccard import JaccardSimilarity


@dataclass
class ClusterResult:
    """
    Result of DBSCAN clustering for a single Transcoder feature.

    Attributes:
        feature_idx: Target feature index.
        layer_idx: Target layer index.
        n_clusters: Number of identified monosemantic sub-clusters.
        n_noise: Number of samples classified as noise (outliers).
        cluster_labels: Array of cluster assignments, shape (n_samples,).
                        -1 indicates noise.
        cluster_sample_indices: Dict mapping cluster_id to list of sample indices.
        cluster_descriptions: Dict mapping cluster_id to semantic description string.
        composite_label: Combined label merging all sub-cluster descriptions.
        jaccard_matrix: The Jaccard similarity matrix used for clustering.
    """
    feature_idx: int
    layer_idx: int
    n_clusters: int
    n_noise: int
    cluster_labels: np.ndarray
    cluster_sample_indices: Dict[int, List[int]] = field(default_factory=dict)
    cluster_descriptions: Dict[int, str] = field(default_factory=dict)
    composite_label: str = ""
    jaccard_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "feature_idx": self.feature_idx,
            "layer_idx": self.layer_idx,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "cluster_labels": self.cluster_labels.tolist(),
            "cluster_sample_indices": {int(k): v for k, v in self.cluster_sample_indices.items()},
            "cluster_descriptions": {int(k): v for k, v in self.cluster_descriptions.items()},
            "composite_label": self.composite_label,
        }


class CircuitClusterer:
    """
    Decomposes polysemantic Transcoder features into monosemantic sub-clusters
    using DBSCAN on Jaccard circuit similarity.

    Args:
        dbscan_eps: DBSCAN epsilon parameter (maximum distance between two
                    samples to be considered neighbors). Range: [0, 1].
                    Lower values produce tighter, more specific clusters.
        dbscan_min_samples: Minimum number of samples in a neighborhood for
                            a point to be considered a core point.
        attribution_threshold: Threshold for circuit set extraction in Jaccard.
        position_bins: Position binning for Jaccard similarity robustness.
    """

    def __init__(
        self,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 3,
        attribution_threshold: float = 0.1,
        position_bins: Optional[int] = None,
    ):
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.jaccard_computer = JaccardSimilarity(
            attribution_threshold=attribution_threshold,
            position_bins=position_bins,
        )

    def cluster(
        self,
        jacobian_results: List[JacobianResult],
        sample_labels: Optional[List[str]] = None,
    ) -> ClusterResult:
        """
        Perform circuit-based clustering on a set of Jacobian analysis results.

        Args:
            jacobian_results: List of JacobianResult instances for the same
                              target feature across different input samples.
            sample_labels: Optional string labels for each sample (e.g., input text).

        Returns:
            ClusterResult with cluster assignments and descriptions.
        """
        feature_idx = jacobian_results[0].feature_idx if jacobian_results else -1
        layer_idx = jacobian_results[0].layer_idx if jacobian_results else -1
        n_samples = len(jacobian_results)

        if n_samples < 2:
            # Cannot cluster with fewer than 2 samples
            return ClusterResult(
                feature_idx=feature_idx,
                layer_idx=layer_idx,
                n_clusters=1 if n_samples == 1 else 0,
                n_noise=0,
                cluster_labels=np.zeros(n_samples, dtype=int),
                cluster_sample_indices={0: list(range(n_samples))},
                composite_label="Insufficient samples for clustering",
            )

        # Step 1: Compute Jaccard distance matrix
        distance_matrix = self.jaccard_computer.compute_distance_matrix(jacobian_results)

        # Step 2: Apply DBSCAN
        if HAS_SKLEARN:
            cluster_labels = self._run_dbscan(distance_matrix)
        else:
            # Fallback: simple threshold-based clustering
            cluster_labels = self._simple_threshold_clustering(distance_matrix)

        # Step 3: Organize results by cluster
        unique_labels = set(cluster_labels)
        n_noise = int(np.sum(cluster_labels == -1))
        n_clusters = len(unique_labels - {-1})

        cluster_sample_indices: Dict[int, List[int]] = {}
        for sample_idx, label in enumerate(cluster_labels):
            if label not in cluster_sample_indices:
                cluster_sample_indices[label] = []
            cluster_sample_indices[label].append(sample_idx)

        # Step 4: Generate cluster descriptions
        cluster_descriptions = self._generate_cluster_descriptions(
            cluster_sample_indices,
            jacobian_results,
            sample_labels,
        )

        # Step 5: Build composite label
        composite_label = self._build_composite_label(
            cluster_descriptions, cluster_sample_indices
        )

        return ClusterResult(
            feature_idx=feature_idx,
            layer_idx=layer_idx,
            n_clusters=n_clusters,
            n_noise=n_noise,
            cluster_labels=cluster_labels,
            cluster_sample_indices=cluster_sample_indices,
            cluster_descriptions=cluster_descriptions,
            composite_label=composite_label,
            jaccard_matrix=1.0 - distance_matrix,  # Store similarity, not distance
        )

    def _run_dbscan(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Run scikit-learn DBSCAN with precomputed distance matrix."""
        dbscan = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="precomputed",
        )
        return dbscan.fit_predict(distance_matrix)

    def _simple_threshold_clustering(
        self, distance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Simple greedy threshold-based clustering fallback when sklearn is unavailable.

        Assigns samples to clusters based on distance threshold, with no
        noise detection (all samples assigned to a cluster).
        """
        n = distance_matrix.shape[0]
        labels = np.full(n, -1, dtype=int)
        current_cluster = 0

        for i in range(n):
            if labels[i] != -1:
                continue
            labels[i] = current_cluster
            for j in range(i + 1, n):
                if labels[j] == -1 and distance_matrix[i, j] <= self.dbscan_eps:
                    labels[j] = current_cluster
            current_cluster += 1

        return labels

    def _generate_cluster_descriptions(
        self,
        cluster_sample_indices: Dict[int, List[int]],
        jacobian_results: List[JacobianResult],
        sample_labels: Optional[List[str]],
    ) -> Dict[int, str]:
        """Generate a textual description for each cluster."""
        descriptions = {}
        for cluster_id, sample_indices in cluster_sample_indices.items():
            if cluster_id == -1:
                descriptions[-1] = f"Noise ({len(sample_indices)} outlier samples)"
                continue

            # Aggregate top positions across cluster samples
            position_counts: Dict[int, int] = {}
            for idx in sample_indices:
                result = jacobian_results[idx]
                for _, _, pos, score in result.top_attention_heads:
                    if abs(score) > 0:
                        position_counts[pos] = position_counts.get(pos, 0) + 1

            top_positions = sorted(
                position_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]

            # Build description from sample labels if available
            if sample_labels:
                cluster_samples = [
                    sample_labels[i] for i in sample_indices
                    if i < len(sample_labels)
                ][:3]
                sample_summary = "; ".join(f'"{s[:30]}"' for s in cluster_samples)
            else:
                sample_summary = f"{len(sample_indices)} samples"

            pos_summary = ", ".join(f"pos_{p}" for p, _ in top_positions)
            descriptions[cluster_id] = (
                f"Cluster {cluster_id}: {len(sample_indices)} samples "
                f"(key positions: {pos_summary or 'N/A'}) | "
                f"Examples: {sample_summary}"
            )

        return descriptions

    def _build_composite_label(
        self,
        cluster_descriptions: Dict[int, str],
        cluster_sample_indices: Dict[int, List[int]],
    ) -> str:
        """Build a unified composite label from all sub-cluster descriptions."""
        valid_clusters = {
            cid: desc for cid, desc in cluster_descriptions.items()
            if cid != -1
        }

        if not valid_clusters:
            return "No valid clusters identified"

        if len(valid_clusters) == 1:
            return list(valid_clusters.values())[0]

        # Sort clusters by size (largest first)
        sorted_clusters = sorted(
            valid_clusters.items(),
            key=lambda x: len(cluster_sample_indices.get(x[0], [])),
            reverse=True,
        )

        parts = [f"[{desc}]" for _, desc in sorted_clusters]
        return f"Polysemantic feature with {len(valid_clusters)} contexts: " + " | ".join(parts)
