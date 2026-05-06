from __future__ import annotations

import torch

from src.reip.pruning import TopologyPruner


def _edge_count(graph):
    if hasattr(graph, "edges"):
        return len(list(graph.edges()))
    return len(graph["edges"])


def test_pruner_connects_consecutive_layers_only():
    pruner = TopologyPruner(threshold=0.0, top_k=None, normalize=False)
    scores = {
        "blocks.0.hook_mlp_out": torch.ones(1, 2, 4),
        "blocks.1.hook_mlp_out": torch.ones(1, 2, 4),
        "blocks.3.hook_mlp_out": torch.ones(1, 2, 4),
    }
    graph = pruner.build_graph(scores)

    # two positions, only layer0->layer1 and no layer1->layer3 since layer2 absent
    assert _edge_count(graph) == 2


def test_pruner_top_k_applies_after_build():
    pruner = TopologyPruner(threshold=0.0, top_k=1, normalize=False)
    scores = {
        "blocks.0.hook_mlp_out": torch.ones(1, 2, 4),
        "blocks.1.hook_mlp_out": torch.ones(1, 2, 4) * 2,
    }
    graph = pruner.build_graph(scores)
    assert _edge_count(graph) == 1
