"""
Sparse Causal Topology Graph Pruning Algorithm.

After LRP backward propagation, this module prunes nodes and edges whose
causal contribution scores fall below a user-defined relevance threshold,
producing a highly sparse, high-fidelity causal circuit topology graph.

Output format: NetworkX DiGraph or JSON-serializable dictionary.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class TopologyPruner:
    """
    Prunes a dense causal attribution graph to a sparse circuit topology.

    The pruner accepts raw relevance scores per component (keyed by hook
    point names) and constructs a directed acyclic graph (DAG) retaining
    only the edges and nodes whose absolute causal contribution exceeds
    the specified threshold.

    Args:
        threshold: Minimum absolute relevance score to retain a node/edge.
                   Nodes below this value are removed from the graph.
        top_k: If set, retain only the top-k edges by absolute score,
               regardless of threshold. Overrides threshold when set.
        normalize: If True, normalize scores to [0, 1] before thresholding.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        top_k: Optional[int] = None,
        normalize: bool = True,
    ):
        self.threshold = threshold
        self.top_k = top_k
        self.normalize = normalize

    def build_graph(
        self,
        relevance_scores: Dict[str, torch.Tensor],
        token_labels: Optional[List[str]] = None,
    ) -> Union["nx.DiGraph", Dict]:
        """
        Build a sparse causal topology graph from LRP relevance scores.

        Args:
            relevance_scores: Dict mapping hook point names to relevance
                              score tensors of shape (batch, seq, d_model).
            token_labels: Optional list of token strings for node labeling.

        Returns:
            A NetworkX DiGraph if networkx is available, otherwise a JSON
            serializable dictionary with 'nodes' and 'edges' keys.
        """
        # Aggregate scores: mean over batch dimension, take L2 norm over d_model
        aggregated: Dict[str, torch.Tensor] = {}
        for name, tensor in relevance_scores.items():
            if tensor.dim() == 3:
                # (batch, seq, d_model) -> (seq,)
                score = tensor.abs().mean(dim=0).norm(dim=-1)
            elif tensor.dim() == 2:
                # (seq, d_model) -> (seq,)
                score = tensor.abs().norm(dim=-1)
            else:
                score = tensor.abs().flatten()
            aggregated[name] = score.detach().cpu()

        # Normalize if requested
        if self.normalize and aggregated:
            all_scores = torch.cat(list(aggregated.values()))
            max_score = all_scores.max().item()
            if max_score > 0:
                aggregated = {k: v / max_score for k, v in aggregated.items()}

        # Build edge list: layer-wise consecutive connections
        edges: List[Dict] = []
        nodes: Dict[str, Dict] = {}
        sorted_names = sorted(
            aggregated.keys(),
            key=lambda n: (self._extract_layer_idx(n), self._extract_component(n), n),
        )
        names_by_layer: Dict[int, List[str]] = {}
        for name in sorted_names:
            names_by_layer.setdefault(self._extract_layer_idx(name), []).append(name)

        for name in sorted_names:
            scores = aggregated[name]
            layer_idx = self._extract_layer_idx(name)
            component = self._extract_component(name)

            for pos_idx, score_val in enumerate(scores.tolist()):
                node_id = f"{name}__pos{pos_idx}"
                token_label = (
                    token_labels[pos_idx]
                    if token_labels and pos_idx < len(token_labels)
                    else f"pos_{pos_idx}"
                )
                nodes[node_id] = {
                    "id": node_id,
                    "layer": layer_idx,
                    "component": component,
                    "position": pos_idx,
                    "token": token_label,
                    "score": round(score_val, 6),
                }

                # Connect from all components in previous layer at same position.
                if layer_idx < 0:
                    continue
                prev_layer_names = names_by_layer.get(layer_idx - 1, [])
                for prev_name in prev_layer_names:
                    src_id = f"{prev_name}__pos{pos_idx}"
                    prev_scores = aggregated.get(prev_name)
                    prev_score_val = (
                        float(prev_scores[pos_idx].item())
                        if prev_scores is not None and pos_idx < len(prev_scores)
                        else 0.0
                    )
                    edge_score = (prev_score_val + score_val) / 2.0
                    if edge_score >= self.threshold:
                        edges.append(
                            {
                                "source": src_id,
                                "target": node_id,
                                "weight": round(edge_score, 6),
                            }
                        )

        # Apply top-k filtering if specified
        if self.top_k is not None:
            edges = sorted(edges, key=lambda e: e["weight"], reverse=True)
            edges = edges[:self.top_k]

        # Filter nodes to only those referenced by retained edges
        referenced_ids = set()
        for edge in edges:
            referenced_ids.add(edge["source"])
            referenced_ids.add(edge["target"])
        # Always keep input and output layer nodes
        for node_id, node_data in nodes.items():
            if node_data["score"] >= self.threshold:
                referenced_ids.add(node_id)

        pruned_nodes = {nid: nd for nid, nd in nodes.items() if nid in referenced_ids}

        if HAS_NETWORKX:
            return self._to_networkx(pruned_nodes, edges)
        else:
            return {"nodes": list(pruned_nodes.values()), "edges": edges}

    def _to_networkx(
        self,
        nodes: Dict[str, Dict],
        edges: List[Dict],
    ) -> "nx.DiGraph":
        """Convert nodes and edges to a NetworkX DiGraph."""
        G = nx.DiGraph()
        for node_id, node_data in nodes.items():
            G.add_node(node_id, **node_data)
        for edge in edges:
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge["weight"],
            )
        return G

    def to_json(
        self,
        graph: Union["nx.DiGraph", Dict],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Serialize the topology graph to a JSON string.

        Args:
            graph: NetworkX DiGraph or dict topology.
            output_path: If provided, write JSON to this file path.

        Returns:
            JSON string representation of the graph.
        """
        if HAS_NETWORKX and isinstance(graph, nx.DiGraph):
            data = {
                "nodes": [
                    {"id": n, **d} for n, d in graph.nodes(data=True)
                ],
                "edges": [
                    {"source": u, "target": v, **d}
                    for u, v, d in graph.edges(data=True)
                ],
            }
        else:
            data = graph

        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @staticmethod
    def _extract_layer_idx(hook_name: str) -> int:
        """Extract layer index from a TransformerLens hook point name."""
        parts = hook_name.split(".")
        for i, part in enumerate(parts):
            if part == "blocks" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return -1  # Input embedding or final layer

    @staticmethod
    def _extract_component(hook_name: str) -> str:
        """Extract component type from a TransformerLens hook point name."""
        if "mlp" in hook_name:
            return "mlp"
        elif "attn" in hook_name:
            return "attn"
        elif "ln" in hook_name:
            return "ln"
        elif "resid" in hook_name:
            return "resid"
        elif "embed" in hook_name:
            return "embed"
        return "other"
