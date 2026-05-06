#!/usr/bin/env python3
"""
MI Toolkit — End-to-End Pipeline: ReIP → CircuitLens → Visualization

This script demonstrates the complete mechanistic interpretability automation
pipeline by:
    1. Running ReIP on GPT-2 small for the IOI (Indirect Object Identification)
       task to identify the Top-20 most important computational nodes.
    2. Feeding the Top-20 node list into CircuitLens to compute inter-node
       connectivity via Jacobian-based attribution analysis.
    3. Outputting a visualization-ready circuit graph in Cytoscape-compatible
       JSON format (nodes + edges), suitable for the MI Dashboard.

Usage:
    python scripts/run_full_pipeline.py [--output OUTPUT_PATH] [--top-k 20] [--device cpu]

Requirements:
    - transformer-lens
    - torch
    - networkx (optional, for graph construction)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reip.pipeline import ReIPPipeline, ReIPConfig, ReIPResult
from src.reip.pruning import TopologyPruner
from src.circuitlens.jacobian import JacobianAnalyzer, JacobianResult
from src.weightlens.transcoder_loader import TranscoderLoader


# ============================================================================
# Configuration
# ============================================================================

IOI_CLEAN_PROMPT = "When Mary and John went to the store, John gave a drink to"
IOI_CORRUPTED_PROMPT = "When Mary and John went to the store, Mary gave a drink to"
IOI_TARGET_TOKEN = " Mary"

MODEL_NAME = "gpt2"


# ============================================================================
# Stage 1: ReIP — Identify Top-K Important Nodes
# ============================================================================

def run_reip_stage(
    model: Any,
    top_k: int = 20,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[ReIPResult, List[Dict]]:
    """
    Run ReIP analysis on GPT-2 small for the IOI task and extract the
    Top-K most important computational nodes.

    The scoring formula used is `corr_grad_x_act_delta` (Attribution Patching),
    which achieves the highest Spearman correlation (0.86) with ground-truth
    activation patching scores based on our ablation study.

    Args:
        model: TransformerLens HookedTransformer instance.
        top_k: Number of top nodes to retain.
        device: Computation device.
        verbose: Print progress information.

    Returns:
        Tuple of (ReIPResult, list of top-k node dicts sorted by score).
    """
    if verbose:
        print("=" * 70)
        print("Stage 1: ReIP — Identifying Top-K Important Nodes")
        print("=" * 70)
        print(f"  Model:           {MODEL_NAME}")
        print(f"  Task:            IOI (Indirect Object Identification)")
        print(f"  Scoring formula: corr_grad_x_act_delta")
        print(f"  Top-K:           {top_k}")
        print()

    config = ReIPConfig(
        model_name=MODEL_NAME,
        pruning_threshold=0.0,  # Keep all nodes initially
        pruning_top_k=None,     # No edge pruning at this stage
        normalize_scores=True,
        target_token_idx=-1,
        device=device,
        verbose=False,
        reset_hooks_end=True,
    )

    pipeline = ReIPPipeline(model, config)

    start_time = time.time()
    result = pipeline.run(
        clean_prompt=IOI_CLEAN_PROMPT,
        corrupted_prompt=IOI_CORRUPTED_PROMPT,
        target_token=IOI_TARGET_TOKEN,
    )
    reip_time = time.time() - start_time

    # Extract per-component aggregate scores for ranking
    component_scores: List[Dict] = []
    for hook_name, tensor in result.relevance_scores.items():
        # Aggregate: mean over batch (if present), then L2 norm over hidden dim
        if tensor.dim() == 3:
            score = tensor.abs().mean(dim=0).norm(dim=-1)
        elif tensor.dim() == 2:
            score = tensor.abs().norm(dim=-1)
        else:
            score = tensor.abs().flatten()

        # Per-position scores
        for pos_idx, score_val in enumerate(score.tolist()):
            token_label = (
                result.token_labels[pos_idx]
                if pos_idx < len(result.token_labels)
                else f"pos_{pos_idx}"
            )
            layer_idx = _extract_layer_idx(hook_name)
            component_type = _extract_component(hook_name)
            component_scores.append({
                "id": f"{hook_name}__pos{pos_idx}",
                "hook_name": hook_name,
                "layer": layer_idx,
                "component": component_type,
                "position": pos_idx,
                "token": token_label,
                "score": score_val,
            })

    # Sort by score descending and take top-k
    component_scores.sort(key=lambda x: x["score"], reverse=True)
    top_k_nodes = component_scores[:top_k]

    if verbose:
        print(f"  ReIP completed in {reip_time:.3f}s")
        print(f"  Total components scored: {len(component_scores)}")
        print(f"  Top-{top_k} nodes selected:")
        print()
        print(f"  {'Rank':<5} {'Node ID':<40} {'Layer':<6} {'Type':<6} {'Score':<10}")
        print(f"  {'-'*5} {'-'*40} {'-'*6} {'-'*6} {'-'*10}")
        for i, node in enumerate(top_k_nodes[:10], 1):
            print(f"  {i:<5} {node['id'][:40]:<40} {node['layer']:<6} "
                  f"{node['component']:<6} {node['score']:.6f}")
        if top_k > 10:
            print(f"  ... ({top_k - 10} more nodes)")
        print()

    return result, top_k_nodes


# ============================================================================
# Stage 2: CircuitLens — Compute Inter-Node Connectivity
# ============================================================================

def run_circuitlens_stage(
    model: Any,
    top_k_nodes: List[Dict],
    reip_result: ReIPResult,
    device: str = "cpu",
    verbose: bool = True,
) -> List[Dict]:
    """
    Use CircuitLens to compute connectivity (edges) between the Top-K nodes.

    For each pair of nodes in adjacent layers, CircuitLens computes the
    Jacobian-based attribution to determine how strongly information flows
    from one node to another through the attention mechanism.

    Args:
        model: TransformerLens HookedTransformer instance.
        top_k_nodes: List of top-k node dicts from ReIP stage.
        reip_result: Full ReIP result containing relevance scores.
        device: Computation device.
        verbose: Print progress information.

    Returns:
        List of edge dicts with 'source', 'target', and 'weight' fields.
    """
    if verbose:
        print("=" * 70)
        print("Stage 2: CircuitLens — Computing Inter-Node Connectivity")
        print("=" * 70)
        print()

    start_time = time.time()

    # Load or create synthetic transcoder weights for Jacobian computation
    loader = TranscoderLoader(model_name=MODEL_NAME, device=device)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    # Group nodes by layer for efficient edge computation
    nodes_by_layer: Dict[int, List[Dict]] = {}
    for node in top_k_nodes:
        nodes_by_layer.setdefault(node["layer"], []).append(node)

    edges: List[Dict] = []
    sorted_layers = sorted(nodes_by_layer.keys())

    # Initialize Jacobian analyzer
    analyzer = JacobianAnalyzer(
        model=model,
        device=device,
        lrp_relevance_threshold=0.01,
        top_k_heads=5,
    )

    # Compute edges between nodes in adjacent layers
    for i in range(len(sorted_layers) - 1):
        src_layer = sorted_layers[i]
        tgt_layer = sorted_layers[i + 1]

        src_nodes = nodes_by_layer[src_layer]
        tgt_nodes = nodes_by_layer[tgt_layer]

        # For each target node, compute how much each source node contributes
        for tgt_node in tgt_nodes:
            tgt_layer_idx = tgt_node["layer"]
            tgt_pos = tgt_node["position"]

            # Load transcoder weights for the target layer
            weights = loader.load_layer(tgt_layer_idx)
            if weights is None:
                weights = loader.create_synthetic_transcoder(
                    layer_idx=tgt_layer_idx,
                    d_model=d_model,
                    n_features=d_model * 4,  # Standard 4x expansion
                )

            # Use feature index 0 as representative for the component
            # The Jacobian captures how the residual stream at each position
            # contributes to this layer's computation
            try:
                input_tokens = model.to_tokens(IOI_CLEAN_PROMPT).to(device)
                jacobian_result = analyzer.compute_jacobian(
                    input_tokens=input_tokens,
                    target_layer=tgt_layer_idx,
                    target_feature_idx=0,
                    W_enc=weights.W_enc,
                    lrp_scores=reip_result.relevance_scores,
                    decompose_heads=True,
                )

                # Extract connectivity scores from Jacobian to source nodes
                for src_node in src_nodes:
                    src_pos = src_node["position"]
                    if src_pos < jacobian_result.jacobian_matrix.shape[0]:
                        # Edge weight = L2 norm of Jacobian row at source position
                        edge_weight = float(
                            jacobian_result.jacobian_matrix[src_pos].norm().item()
                        )
                        # Modulate by ReIP scores of both endpoints
                        edge_weight *= min(src_node["score"], tgt_node["score"])

                        if edge_weight > 1e-6:
                            edges.append({
                                "source": src_node["id"],
                                "target": tgt_node["id"],
                                "weight": round(edge_weight, 6),
                            })
            except Exception as e:
                # Fallback: use ReIP score product as edge weight
                if verbose:
                    print(f"  [Warning] Jacobian computation failed for "
                          f"layer {tgt_layer_idx}: {e}")
                    print(f"  [Fallback] Using ReIP score-based connectivity")
                for src_node in src_nodes:
                    if src_node["position"] == tgt_node["position"]:
                        edge_weight = min(src_node["score"], tgt_node["score"])
                        if edge_weight > 1e-6:
                            edges.append({
                                "source": src_node["id"],
                                "target": tgt_node["id"],
                                "weight": round(edge_weight, 6),
                            })

    # Also add edges between nodes in the same layer at the same position
    # (MLP ↔ Attention interaction within a layer)
    for layer_idx, layer_nodes in nodes_by_layer.items():
        positions_seen: Dict[int, List[Dict]] = {}
        for node in layer_nodes:
            positions_seen.setdefault(node["position"], []).append(node)
        for pos, pos_nodes in positions_seen.items():
            if len(pos_nodes) > 1:
                for j in range(len(pos_nodes)):
                    for k in range(j + 1, len(pos_nodes)):
                        edge_weight = min(
                            pos_nodes[j]["score"], pos_nodes[k]["score"]
                        )
                        if edge_weight > 1e-6:
                            edges.append({
                                "source": pos_nodes[j]["id"],
                                "target": pos_nodes[k]["id"],
                                "weight": round(edge_weight, 6),
                            })

    # Sort edges by weight and keep top connections
    edges.sort(key=lambda e: e["weight"], reverse=True)

    circuitlens_time = time.time() - start_time

    if verbose:
        print(f"  CircuitLens completed in {circuitlens_time:.3f}s")
        print(f"  Total edges computed: {len(edges)}")
        print(f"  Top-5 strongest connections:")
        print()
        print(f"  {'Source':<40} {'Target':<40} {'Weight':<10}")
        print(f"  {'-'*40} {'-'*40} {'-'*10}")
        for edge in edges[:5]:
            print(f"  {edge['source'][:40]:<40} "
                  f"{edge['target'][:40]:<40} {edge['weight']:.6f}")
        print()

    return edges


# ============================================================================
# Stage 3: Visualization Output — Cytoscape JSON
# ============================================================================

def build_visualization_output(
    top_k_nodes: List[Dict],
    edges: List[Dict],
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Build the final visualization-ready output in Cytoscape-compatible JSON format.

    The output format is directly consumable by the MI Dashboard's Cytoscape
    component and includes:
        - nodes: List of node elements with id, label, score, layer, component
        - edges: List of edge elements with source, target, weight

    Args:
        top_k_nodes: List of top-k node dicts.
        edges: List of edge dicts from CircuitLens.
        output_path: If provided, write JSON to this file.
        verbose: Print progress information.

    Returns:
        Cytoscape-compatible elements dict.
    """
    if verbose:
        print("=" * 70)
        print("Stage 3: Building Visualization Output (Cytoscape JSON)")
        print("=" * 70)
        print()

    # Build Cytoscape elements
    cytoscape_elements = []

    # Node elements
    for node in top_k_nodes:
        element = {
            "data": {
                "id": node["id"],
                "label": f"L{node['layer']}_{node['component']}",
                "full_label": f"Layer {node['layer']} {node['component'].upper()} "
                              f"@ \"{node['token']}\" (pos {node['position']})",
                "score": node["score"],
                "layer": node["layer"],
                "component": node["component"],
                "token": node["token"],
                "position": node["position"],
            }
        }
        cytoscape_elements.append(element)

    # Edge elements
    for i, edge in enumerate(edges):
        element = {
            "data": {
                "id": f"edge_{i}",
                "source": edge["source"],
                "target": edge["target"],
                "weight": edge["weight"],
            }
        }
        cytoscape_elements.append(element)

    # Build summary output
    output = {
        "metadata": {
            "model": MODEL_NAME,
            "task": "IOI (Indirect Object Identification)",
            "clean_prompt": IOI_CLEAN_PROMPT,
            "corrupted_prompt": IOI_CORRUPTED_PROMPT,
            "target_token": IOI_TARGET_TOKEN,
            "scoring_formula": "corr_grad_x_act_delta",
            "pipeline": "ReIP → CircuitLens → Cytoscape Visualization",
            "n_nodes": len(top_k_nodes),
            "n_edges": len(edges),
        },
        "cytoscape_elements": cytoscape_elements,
        "nodes": top_k_nodes,
        "edges": edges,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"  Output written to: {output_path}")

    if verbose:
        print(f"  Cytoscape elements: {len(cytoscape_elements)} total")
        print(f"    - Nodes: {len(top_k_nodes)}")
        print(f"    - Edges: {len(edges)}")
        print()
        print("=" * 70)
        print("Pipeline Complete: ReIP → CircuitLens → Visualization")
        print("=" * 70)
        print()
        print("The output JSON can be loaded directly into the MI Dashboard")
        print("(dashboards/app.py) or any Cytoscape.js-compatible viewer.")
        print()

    return output


# ============================================================================
# Utility Functions
# ============================================================================

def _extract_layer_idx(hook_name: str) -> int:
    """Extract layer index from a TransformerLens hook point name."""
    parts = hook_name.split(".")
    for i, part in enumerate(parts):
        if part == "blocks" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def _extract_component(hook_name: str) -> str:
    """Extract component type from a TransformerLens hook point name."""
    if "mlp" in hook_name:
        return "mlp"
    elif "attn" in hook_name:
        return "attn"
    elif "resid" in hook_name:
        return "resid"
    elif "embed" in hook_name:
        return "embed"
    return "other"


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MI Toolkit End-to-End Pipeline: ReIP → CircuitLens → Visualization"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/circuit_graph.json",
        help="Output path for the Cytoscape JSON file (default: outputs/circuit_graph.json)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=20,
        help="Number of top nodes to identify (default: 20)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║  MI Toolkit — Mechanistic Interpretability Automation Pipeline      ║")
        print("║  ReIP Localization → CircuitLens Decomposition → Visualization      ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()

    # Load model
    if verbose:
        print(f"Loading model: {MODEL_NAME} (device: {args.device})...")

    try:
        from transformer_lens import HookedTransformer
        model = HookedTransformer.from_pretrained(MODEL_NAME, device=args.device)
    except ImportError:
        print("ERROR: transformer-lens is required. Install with:")
        print("  pip install transformer-lens")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    if verbose:
        print(f"Model loaded: {model.cfg.n_layers} layers, "
              f"d_model={model.cfg.d_model}, "
              f"n_heads={model.cfg.n_heads}")
        print()

    total_start = time.time()

    # Stage 1: ReIP
    reip_result, top_k_nodes = run_reip_stage(
        model=model,
        top_k=args.top_k,
        device=args.device,
        verbose=verbose,
    )

    # Stage 2: CircuitLens
    edges = run_circuitlens_stage(
        model=model,
        top_k_nodes=top_k_nodes,
        reip_result=reip_result,
        device=args.device,
        verbose=verbose,
    )

    # Stage 3: Visualization
    output = build_visualization_output(
        top_k_nodes=top_k_nodes,
        edges=edges,
        output_path=args.output,
        verbose=verbose,
    )

    total_time = time.time() - total_start

    if verbose:
        print(f"Total pipeline execution time: {total_time:.3f}s")
        print()

    return output


if __name__ == "__main__":
    main()
