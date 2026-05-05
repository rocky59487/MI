"""
CircuitLens: Context-dependent feature analysis and polysemanticity decomposition.

This module extends WeightLens with dynamic Jacobian-based attribution to handle
context-dependent features in middle layers of modern architectures (RoPE models
like Llama and Gemma), where static weight analysis alone is insufficient.

Key components:
    - jacobian: Jacobian matrix computation with cross-layer interference isolation
    - jaccard: Jaccard similarity matrix for circuit-based clustering
    - clustering: DBSCAN-based polysemanticity decomposition into monosemantic sub-clusters
"""

from .jacobian import JacobianAnalyzer
from .jaccard import JaccardSimilarity
from .clustering import CircuitClusterer

__all__ = [
    "JacobianAnalyzer",
    "JaccardSimilarity",
    "CircuitClusterer",
]
