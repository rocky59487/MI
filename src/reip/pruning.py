"""Graph pruning helpers for ReIP relevance outputs."""

from __future__ import annotations


def prune_edges(edges: list[dict], threshold: float) -> list[dict]:
    """Keep only edges with absolute score >= threshold."""
    return [e for e in edges if abs(float(e.get("score", 0.0))) >= threshold]
