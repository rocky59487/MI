"""End-to-end ReIP orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .pruning import prune_edges


@dataclass
class ReIPConfig:
    prune_threshold: float = 0.05
    rules: list[str] = field(default_factory=lambda: ["ln", "identity", "zero"])


class ReIPPipeline:
    def __init__(self, config: ReIPConfig | None = None) -> None:
        self.config = config or ReIPConfig()

    def run(self, raw_edges: list[dict]) -> dict:
        """Run lightweight pruning pipeline on precomputed relevance edges."""
        pruned = prune_edges(raw_edges, self.config.prune_threshold)
        return {"rules": self.config.rules, "edges": pruned}
