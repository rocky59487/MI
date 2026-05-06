"""
Dash Callbacks for MI Circuit Explorer — Directional Causal Flow.

This module implements the interactive logic for the redesigned dashboard:
    1. Analyze button: Generate demo topology and start sequential reveal animation
    2. Animation interval: Reveal nodes one-by-one (typewriter effect)
    3. Node click: Expand node details and show child sub-nodes
    4. Layout/Top-N controls: Update graph configuration
    5. Graph statistics: Show causal path summary
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import random

try:
    from dash import Input, Output, State, callback, html, no_update, ctx
    import dash
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

from .stylesheet import build_stylesheet, get_layout_config
from .layout import topology_to_cytoscape_elements


# ============================================================================
# Demo Data Generator
# ============================================================================

def _generate_demo_topology(prompt: str, top_n: int = 12) -> Dict:
    """
    Generate a realistic demo topology for visualization.

    This creates a directional causal graph that mimics how GPT-2 processes
    the IOI (Indirect Object Identification) task, with nodes arranged by
    layer and edges showing causal flow.

    Args:
        prompt: The input prompt text.
        top_n: Number of top nodes to generate.

    Returns:
        Dict with 'nodes' and 'edges' keys.
    """
    tokens = prompt.split() if prompt else ["The", "model", "thinks"]

    # Define realistic node templates across layers
    node_templates = [
        # (layer, component, semantic_hint, base_score)
        (-1, "embed", "input token embedding", 0.45),
        (0, "attn", "positional attention", 0.62),
        (0, "mlp", "token identity", 0.55),
        (1, "attn", "induction head", 0.78),
        (2, "attn", "subject tracking", 0.85),
        (2, "mlp", "entity binding", 0.71),
        (3, "attn", "indirect object", 0.92),
        (3, "mlp", "role assignment", 0.68),
        (4, "attn", "name mover head", 0.95),
        (4, "mlp", "output projection", 0.73),
        (5, "attn", "backup name mover", 0.64),
        (5, "mlp", "logit attribution", 0.81),
        (6, "resid", "final residual", 0.58),
        (7, "attn", "S-inhibition head", 0.88),
        (8, "mlp", "prediction head", 0.76),
    ]

    # Select top-N nodes based on score
    node_templates.sort(key=lambda x: x[3], reverse=True)
    selected = node_templates[:top_n]
    # Re-sort by layer for flow ordering
    selected.sort(key=lambda x: (x[0], x[3]))

    nodes = []
    for i, (layer, comp, semantic, score) in enumerate(selected):
        # Add slight randomness to scores
        jittered_score = min(1.0, max(0.1, score + random.uniform(-0.05, 0.05)))
        token_idx = i % len(tokens)
        node_id = f"blocks.{max(0, layer)}.hook_{comp}_out__pos{token_idx}"
        if layer == -1:
            node_id = f"embed__pos{token_idx}"

        nodes.append({
            "id": node_id,
            "layer": layer,
            "component": comp,
            "token": tokens[token_idx] if token_idx < len(tokens) else f"pos_{token_idx}",
            "score": round(jittered_score, 4),
            "position": token_idx,
            "semantic": semantic,
        })

    # Generate edges following causal flow (earlier layer → later layer)
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            src = nodes[i]
            tgt = nodes[j]
            # Only connect adjacent or near-adjacent layers
            layer_diff = tgt["layer"] - src["layer"]
            if 0 < layer_diff <= 2:
                # Edge weight based on both node scores and proximity
                weight = (src["score"] * tgt["score"]) * (1.0 / layer_diff)
                weight = min(1.0, weight + random.uniform(-0.1, 0.1))
                if weight > 0.2:
                    edges.append({
                        "source": src["id"],
                        "target": tgt["id"],
                        "weight": round(max(0.1, weight), 4),
                    })

    # Keep only the strongest edges to avoid clutter
    edges.sort(key=lambda e: e["weight"], reverse=True)
    max_edges = top_n * 2  # Roughly 2 edges per node
    edges = edges[:max_edges]

    return {"nodes": nodes, "edges": edges}


# ============================================================================
# Callback Registration
# ============================================================================

def register_callbacks(app: Any) -> None:
    """
    Register all Dash callbacks for the directional flow visualization.

    Args:
        app: A Dash application instance.
    """
    if not HAS_DASH:
        raise ImportError("Dash is required. Install with: pip install dash")

    # ------------------------------------------------------------------
    # Callback 1: Run Analysis — generate topology and start animation
    # ------------------------------------------------------------------
    @app.callback(
        Output("topology-store", "data"),
        Output("all-elements-store", "data"),
        Output("animation-step", "data"),
        Output("animation-interval", "disabled"),
        Output("animation-interval", "n_intervals"),
        Output("circuit-graph", "elements", allow_duplicate=True),
        Output("status-text", "children"),
        Input("run-analysis-btn", "n_clicks"),
        State("clean-prompt-input", "value"),
        State("topn-selector", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(n_clicks, prompt, top_n):
        """Trigger analysis: generate topology and start sequential reveal."""
        if not n_clicks or not prompt:
            return (no_update,) * 7

        top_n = top_n or 12

        # Generate demo topology
        topology = _generate_demo_topology(prompt, top_n=top_n)

        # Convert to cytoscape elements
        elements = topology_to_cytoscape_elements(
            topology,
            semantic_labels={
                n["id"]: n.get("semantic", "")
                for n in topology["nodes"]
            },
            top_n=top_n,
        )

        # Start with empty graph, animation will reveal nodes
        return (
            topology,           # topology-store
            elements,           # all-elements-store
            0,                  # animation-step reset
            False,              # animation-interval enabled
            0,                  # reset n_intervals
            [],                 # clear graph
            "Tracing causal path...",
        )

    # ------------------------------------------------------------------
    # Callback 2: Animation interval — reveal nodes sequentially
    # ------------------------------------------------------------------
    @app.callback(
        Output("circuit-graph", "elements"),
        Output("animation-interval", "disabled", allow_duplicate=True),
        Output("status-text", "children", allow_duplicate=True),
        Input("animation-interval", "n_intervals"),
        State("all-elements-store", "data"),
        prevent_initial_call=True,
    )
    def animate_reveal(n_intervals, all_elements):
        """Reveal nodes and edges one step at a time."""
        if not all_elements:
            return no_update, True, no_update

        # Separate nodes and edges
        nodes = [e for e in all_elements if "source" not in e.get("data", {})]
        edges = [e for e in all_elements if "source" in e.get("data", {})]

        total_steps = len(nodes) + len(edges)

        if n_intervals >= total_steps:
            # Animation complete
            # Mark all as visible
            revealed = []
            for e in all_elements:
                e_copy = dict(e)
                e_copy["classes"] = (e_copy.get("classes", "") + " visible").strip()
                revealed.append(e_copy)
            n_nodes = len(nodes)
            n_edges = len(edges)
            status = f"Analysis complete — {n_nodes} key nodes, {n_edges} causal connections"
            return revealed, True, status

        # Reveal up to current step
        revealed = []

        # First reveal nodes, then edges
        nodes_to_show = min(n_intervals, len(nodes))
        edges_to_show = max(0, n_intervals - len(nodes))

        for i in range(nodes_to_show):
            e_copy = dict(nodes[i])
            e_copy["classes"] = (e_copy.get("classes", "") + " visible").strip()
            revealed.append(e_copy)

        for i in range(edges_to_show):
            e_copy = dict(edges[i])
            e_copy["classes"] = (e_copy.get("classes", "") + " visible").strip()
            revealed.append(e_copy)

        status = f"Revealing node {nodes_to_show}/{len(nodes)}..."
        if edges_to_show > 0:
            status = f"Connecting edges {edges_to_show}/{len(edges)}..."

        return revealed, False, status

    # ------------------------------------------------------------------
    # Callback 3: Node click — show detailed info and expand sub-nodes
    # ------------------------------------------------------------------
    @app.callback(
        Output("node-info", "children"),
        Output("expanded-node-store", "data"),
        Input("circuit-graph", "tapNodeData"),
        State("topology-store", "data"),
        prevent_initial_call=True,
    )
    def display_node_details(tap_data, topology_data):
        """Display detailed information for a clicked node."""
        if not tap_data:
            return html.P(
                "Click a node to inspect its role in the causal chain.",
                style={"color": "#606060", "fontSize": "11px"},
            ), None

        layer = tap_data.get("layer", "?")
        component = tap_data.get("component", "?")
        token = tap_data.get("token", "")
        score = tap_data.get("score", 0.0)
        full_label = tap_data.get("full_label", "")
        rank = tap_data.get("rank", "?")
        node_id = tap_data.get("id", "")

        # Score bar visualization
        score_pct = min(100, int(float(score) * 100))
        score_color = "#E74C3C" if score > 0.8 else "#F39C12" if score > 0.5 else "#4A90D9"

        children = [
            # Node identity card
            html.Div(
                style={
                    "backgroundColor": "#0D1B2A",
                    "padding": "14px",
                    "borderRadius": "8px",
                    "marginBottom": "12px",
                    "border": "1px solid #2C3E50",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                        children=[
                            html.Span(
                                f"Layer {layer}",
                                style={"color": "#00FFFF", "fontSize": "14px", "fontWeight": "700"},
                            ),
                            html.Span(
                                f"#{rank}",
                                style={
                                    "color": "#808080", "fontSize": "11px",
                                    "backgroundColor": "#2C3E50",
                                    "padding": "2px 8px", "borderRadius": "10px",
                                },
                            ),
                        ],
                    ),
                    html.P(
                        component.upper(),
                        style={
                            "color": _component_color(component),
                            "fontSize": "12px", "fontWeight": "600",
                            "margin": "6px 0 2px 0",
                        },
                    ),
                    html.P(
                        f'Token: "{token}"' if token else "Token: N/A",
                        style={"color": "#B0B0B0", "fontSize": "11px", "margin": "2px 0"},
                    ),
                ],
            ),

            # ReIP Score bar
            html.Div(
                style={"marginBottom": "12px"},
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"},
                        children=[
                            html.Span("ReIP Score", style={"color": "#808080", "fontSize": "10px"}),
                            html.Span(f"{score:.4f}", style={"color": score_color, "fontSize": "11px", "fontWeight": "600"}),
                        ],
                    ),
                    html.Div(
                        style={
                            "width": "100%", "height": "6px",
                            "backgroundColor": "#2C3E50", "borderRadius": "3px",
                            "overflow": "hidden",
                        },
                        children=[
                            html.Div(
                                style={
                                    "width": f"{score_pct}%", "height": "100%",
                                    "backgroundColor": score_color,
                                    "borderRadius": "3px",
                                    "transition": "width 0.5s ease",
                                },
                            ),
                        ],
                    ),
                ],
            ),
        ]

        # Semantic label
        if full_label:
            children.append(
                html.Div(
                    style={"marginBottom": "12px"},
                    children=[
                        html.P(
                            "Semantic Role",
                            style={"color": "#808080", "fontSize": "10px", "margin": "0 0 4px 0"},
                        ),
                        html.Div(
                            full_label,
                            style={
                                "color": "#E0E0E0", "fontSize": "11px",
                                "backgroundColor": "#0D1B2A",
                                "padding": "8px 10px", "borderRadius": "6px",
                                "border": "1px solid #2C3E50",
                                "lineHeight": "1.4",
                            },
                        ),
                    ],
                )
            )

        # Causal interpretation
        interpretation = _get_causal_interpretation(layer, component, token)
        if interpretation:
            children.append(
                html.Div(
                    style={"marginBottom": "12px"},
                    children=[
                        html.P(
                            "Causal Interpretation",
                            style={"color": "#808080", "fontSize": "10px", "margin": "0 0 4px 0"},
                        ),
                        html.Div(
                            interpretation,
                            style={
                                "color": "#B0B0B0", "fontSize": "11px",
                                "backgroundColor": "#0D1B2A",
                                "padding": "8px 10px", "borderRadius": "6px",
                                "border": "1px solid #2C3E50",
                                "fontStyle": "italic",
                                "lineHeight": "1.4",
                            },
                        ),
                    ],
                )
            )

        # Connected nodes info
        if topology_data:
            connections = _get_connections(node_id, topology_data)
            if connections:
                children.append(
                    html.Div(
                        style={"marginTop": "8px"},
                        children=[
                            html.P(
                                f"Connections ({len(connections)})",
                                style={"color": "#808080", "fontSize": "10px", "margin": "0 0 6px 0"},
                            ),
                            html.Div(
                                children=[
                                    html.Div(
                                        style={
                                            "display": "flex", "justifyContent": "space-between",
                                            "padding": "4px 8px", "marginBottom": "3px",
                                            "backgroundColor": "#0D1B2A", "borderRadius": "4px",
                                            "fontSize": "10px",
                                        },
                                        children=[
                                            html.Span(
                                                conn["direction"] + " " + conn["node"][:25],
                                                style={"color": "#B0B0B0"},
                                            ),
                                            html.Span(
                                                f"{conn['weight']:.3f}",
                                                style={"color": "#F39C12"},
                                            ),
                                        ],
                                    )
                                    for conn in connections[:6]
                                ],
                            ),
                        ],
                    )
                )

        return children, node_id

    # ------------------------------------------------------------------
    # Callback 4: Layout selector
    # ------------------------------------------------------------------
    @app.callback(
        Output("circuit-graph", "layout"),
        Input("layout-selector", "value"),
        prevent_initial_call=True,
    )
    def update_layout(layout_name):
        """Update the graph layout direction."""
        return get_layout_config(layout_name or "dagre-lr")

    # ------------------------------------------------------------------
    # Callback 5: Top-N selector — re-filter and re-animate
    # ------------------------------------------------------------------
    @app.callback(
        Output("circuit-graph", "elements", allow_duplicate=True),
        Output("all-elements-store", "data", allow_duplicate=True),
        Input("topn-selector", "value"),
        State("topology-store", "data"),
        prevent_initial_call=True,
    )
    def update_topn(top_n, topology_data):
        """Re-filter topology with new top-N value."""
        if not topology_data:
            return no_update, no_update

        top_n = top_n or 12

        elements = topology_to_cytoscape_elements(
            topology_data,
            semantic_labels={
                n["id"]: n.get("semantic", n.get("token", ""))
                for n in topology_data.get("nodes", [])
            },
            top_n=top_n,
        )

        # Show all immediately (no re-animation)
        visible_elements = []
        for e in elements:
            e_copy = dict(e)
            e_copy["classes"] = (e_copy.get("classes", "") + " visible").strip()
            visible_elements.append(e_copy)

        return visible_elements, elements

    # ------------------------------------------------------------------
    # Callback 6: Graph statistics
    # ------------------------------------------------------------------
    @app.callback(
        Output("graph-stats", "children"),
        Input("circuit-graph", "elements"),
        State("topology-store", "data"),
        prevent_initial_call=True,
    )
    def update_graph_stats(elements, topology_data):
        """Display causal path summary statistics."""
        if not elements:
            return html.P(
                "No analysis loaded.",
                style={"color": "#606060", "fontSize": "11px"},
            )

        n_nodes = sum(1 for e in elements if "source" not in e.get("data", {}))
        n_edges = sum(1 for e in elements if "source" in e.get("data", {}))

        # Find layer range
        layers = [
            e["data"].get("layer", 0)
            for e in elements
            if "source" not in e.get("data", {})
        ]
        min_layer = min(layers) if layers else 0
        max_layer = max(layers) if layers else 0

        return html.Div(
            children=[
                _stat_row("Active Nodes", str(n_nodes), "#4A90D9"),
                _stat_row("Causal Edges", str(n_edges), "#E67E22"),
                _stat_row("Layer Span", f"L{min_layer} → L{max_layer}", "#27AE60"),
                _stat_row("Path Depth", f"{max_layer - min_layer + 1} layers", "#8E44AD"),
            ],
        )


# ============================================================================
# Helper Functions
# ============================================================================

def _component_color(component: str) -> str:
    """Return the theme color for a component type."""
    colors = {
        "embed": "#8E44AD",
        "attn": "#E67E22",
        "mlp": "#4A90D9",
        "resid": "#27AE60",
        "output": "#E74C3C",
    }
    return colors.get(component, "#808080")


def _get_causal_interpretation(layer: int, component: str, token: str) -> str:
    """Generate a human-readable causal interpretation for a node."""
    if layer == -1 or component == "embed":
        return f"This embedding captures the initial representation of \"{token}\" before any transformer processing."
    elif component == "attn" and layer <= 2:
        return f"Early attention head at layer {layer} — likely performing positional or syntactic pattern matching."
    elif component == "attn" and layer <= 5:
        return f"Mid-layer attention at layer {layer} — tracking entity relationships and binding subjects to objects."
    elif component == "attn":
        return f"Late attention at layer {layer} — performing the final name-moving or copying operation for output."
    elif component == "mlp" and layer <= 2:
        return f"Early MLP at layer {layer} — enriching token representations with learned features."
    elif component == "mlp":
        return f"MLP at layer {layer} — computing non-linear transformations for output logit attribution."
    elif component == "resid":
        return f"Residual stream at layer {layer} — accumulating information from all previous computations."
    return ""


def _get_connections(node_id: str, topology_data: Dict) -> List[Dict]:
    """Get all connections (incoming and outgoing) for a node."""
    connections = []
    for edge in topology_data.get("edges", []):
        if edge["source"] == node_id:
            connections.append({
                "direction": "→",
                "node": edge["target"].split("__")[0].replace("blocks.", "L").replace(".hook_", " "),
                "weight": edge["weight"],
            })
        elif edge["target"] == node_id:
            connections.append({
                "direction": "←",
                "node": edge["source"].split("__")[0].replace("blocks.", "L").replace(".hook_", " "),
                "weight": edge["weight"],
            })
    connections.sort(key=lambda x: x["weight"], reverse=True)
    return connections


def _stat_row(label: str, value: str, color: str):
    """Create a statistics row for the summary panel."""
    return html.Div(
        style={
            "display": "flex", "justifyContent": "space-between",
            "padding": "6px 0", "borderBottom": "1px solid #2C3E50",
        },
        children=[
            html.Span(label, style={"color": "#808080", "fontSize": "11px"}),
            html.Span(value, style={"color": color, "fontSize": "11px", "fontWeight": "600"}),
        ],
    )
