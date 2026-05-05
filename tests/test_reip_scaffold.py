from src.reip.pipeline import ReIPConfig, ReIPPipeline
from src.reip.pruning import prune_edges


def test_prune_edges_threshold():
    edges = [
        {"source": "a", "target": "b", "score": 0.01},
        {"source": "b", "target": "c", "score": -0.2},
    ]
    pruned = prune_edges(edges, threshold=0.05)
    assert len(pruned) == 1
    assert pruned[0]["source"] == "b"


def test_pipeline_run():
    pipe = ReIPPipeline(ReIPConfig(prune_threshold=0.1))
    out = pipe.run([
        {"source": "x", "target": "y", "score": 0.09},
        {"source": "x", "target": "z", "score": 0.11},
    ])
    assert len(out["edges"]) == 1
    assert out["edges"][0]["target"] == "z"
