"""
Dashboard Layout for MI Circuit Explorer — Directional Causal Flow.

This module defines the Dash application layout for the redesigned Circuit
Explorer. The visualization shows a directional flow graph (left-to-right)
that reveals how the model "thinks" — tracing the causal path from input
through attention heads, MLPs, and residual streams to the output prediction.

Key design principles:
    - Directional flow: Input → Early layers → Mid layers → Late layers → Output
    - Top-N filtering: Only show the most important nodes (Top-10/15 by ReIP score)
    - Sequential reveal: Nodes appear one-by-one like a typewriter effect
    - Click-to-expand: Click any node to see sub-details and child connections
    - Deep space + neural pulse aesthetic
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

try:
    import dash
    from dash import dcc, html
    import dash_cytoscape as cyto
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

from .stylesheet import build_stylesheet, get_layout_config


# ============================================================================
# Data Conversion
# ============================================================================

def topology_to_cytoscape_elements(
    topology: Union[Dict, Any],
    semantic_labels: Optional[Dict[str, str]] = None,
    cluster_labels: Optional[Dict[str, str]] = None,
    top_n: int = 12,
) -> List[Dict]:
    """
    Convert a ReIP topology graph to Cytoscape elements format.

    Only the Top-N nodes (by ReIP score) are included by default. Nodes are
    sorted by layer to enable left-to-right directional layout.

    Args:
        topology: Either a NetworkX DiGraph or a dict with 'nodes' and 'edges'.
        semantic_labels: Optional dict mapping node_id to semantic label.
        cluster_labels: Optional dict mapping node_id to cluster label.
        top_n: Maximum number of nodes to display (sorted by score).

    Returns:
        List of Cytoscape element dicts.
    """
    elements = []

    # Normalize topology to dict format
    if hasattr(topology, "nodes") and hasattr(topology, "edges"):
        nodes = [{"id": n, **d} for n, d in topology.nodes(data=True)]
        edges = [
            {"source": u, "target": v, **d}
            for u, v, d in topology.edges(data=True)
        ]
    elif isinstance(topology, dict):
        nodes = topology.get("nodes", [])
        edges = topology.get("edges", [])
    else:
        return []

    # Sort nodes by score and take top-N
    nodes_sorted = sorted(nodes, key=lambda x: x.get("score", 0), reverse=True)
    top_nodes = nodes_sorted[:top_n]
    top_node_ids = {n.get("id", "") for n in top_nodes}

    # Sort top nodes by layer for directional ordering
    top_nodes.sort(key=lambda x: (x.get("layer", 0), x.get("position", 0)))

    # Build node elements
    for idx, node in enumerate(top_nodes):
        node_id = node.get("id", "")
        score = node.get("score", 0.0)
        layer = node.get("layer", -1)
        component = node.get("component", "other")
        token = node.get("token", "")
        position = node.get("position", 0)

        # Resolve semantic label
        semantic = ""
        if semantic_labels and node_id in semantic_labels:
            semantic = semantic_labels[node_id]
        elif token:
            semantic = token

        # Resolve cluster label
        cluster_info = ""
        if cluster_labels and node_id in cluster_labels:
            cluster_info = cluster_labels[node_id]

        # Build concise display label with score
        comp_short = component[:4].upper()
        if semantic:
            display_label = f"L{layer} {comp_short}\n\"{semantic[:15]}\""
        else:
            display_label = f"L{layer} {comp_short}"

        elements.append({
            "data": {
                "id": node_id,
                "label": display_label,
                "full_label": semantic,
                "cluster_info": cluster_info,
                "score": round(float(score), 4),
                "layer": layer,
                "component": component,
                "token": token,
                "position": position,
                "rank": idx + 1,
                "score_display": f"{score:.3f}",
            },
            "classes": component,
        })

    # Build edge elements — only include edges between visible top-N nodes
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        weight = edge.get("weight", 0.0)

        if not source or not target:
            continue
        if source not in top_node_ids or target not in top_node_ids:
            continue

        elements.append({
            "data": {
                "source": source,
                "target": target,
                "weight": round(float(weight), 4),
                "id": f"{source}__{target}",
            },
        })

    return elements


# ============================================================================
# Layout Builder
# ============================================================================

def build_layout(app_title: str = "MI Circuit Explorer") -> Any:
    """
    Build the complete Dash application layout for the directional flow design.

    The layout features:
        - A clean prompt input with Analyze button
        - A full-width directional graph (left-to-right causal flow)
        - A slide-out detail panel on node click
        - Sequential reveal animation controls

    Returns:
        Dash HTML component tree.
    """
    if not HAS_DASH:
        raise ImportError(
            "Dash and dash-cytoscape are required. "
            "Install with: pip install dash dash-cytoscape"
        )

    layout = html.Div(
        id="app-container",
        style={
            "fontFamily": "'JetBrains Mono', 'Fira Code', monospace",
            "backgroundColor": "#0D1B2A",
            "minHeight": "100vh",
            "color": "#E0E0E0",
            "overflow": "hidden",
        },
        children=[
            # ---------------------------------------------------------------
            # Header with prompt input
            # ---------------------------------------------------------------
            html.Div(
                id="header",
                style={
                    "backgroundColor": "#1B2838",
                    "padding": "16px 24px",
                    "borderBottom": "1px solid #2C3E50",
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "20px",
                },
                children=[
                    # Title
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "12px"},
                        children=[
                            html.Div(
                                style={
                                    "width": "8px", "height": "8px",
                                    "borderRadius": "50%",
                                    "backgroundColor": "#00FFFF",
                                    "boxShadow": "0 0 8px #00FFFF",
                                },
                            ),
                            html.H1(
                                app_title,
                                style={
                                    "color": "#E0E0E0", "fontSize": "16px",
                                    "margin": 0, "fontWeight": "600",
                                    "letterSpacing": "0.5px",
                                },
                            ),
                        ],
                    ),

                    # Prompt input area
                    html.Div(
                        style={
                            "display": "flex", "flex": "1",
                            "gap": "12px", "alignItems": "center",
                        },
                        children=[
                            dcc.Input(
                                id="clean-prompt-input",
                                type="text",
                                placeholder="Enter prompt to analyze (e.g., 'When Mary and John went to the store, John gave a drink to')",
                                style={
                                    "flex": "1",
                                    "backgroundColor": "#0D1B2A",
                                    "color": "#E0E0E0",
                                    "border": "1px solid #2C3E50",
                                    "borderRadius": "6px",
                                    "padding": "10px 14px",
                                    "fontSize": "13px",
                                    "outline": "none",
                                },
                                debounce=True,
                            ),
                            html.Button(
                                "Analyze",
                                id="run-analysis-btn",
                                style={
                                    "backgroundColor": "#00FFFF",
                                    "color": "#0D1B2A",
                                    "border": "none",
                                    "borderRadius": "6px",
                                    "padding": "10px 24px",
                                    "fontSize": "13px",
                                    "fontWeight": "700",
                                    "cursor": "pointer",
                                    "letterSpacing": "0.5px",
                                    "transition": "all 0.2s ease",
                                },
                            ),
                        ],
                    ),

                    # Layout & top-N controls
                    html.Div(
                        style={"display": "flex", "gap": "12px", "alignItems": "center"},
                        children=[
                            html.Label(
                                "Top-N:",
                                style={"color": "#808080", "fontSize": "11px"},
                            ),
                            dcc.Dropdown(
                                id="topn-selector",
                                options=[
                                    {"label": "Top 8", "value": 8},
                                    {"label": "Top 10", "value": 10},
                                    {"label": "Top 12", "value": 12},
                                    {"label": "Top 15", "value": 15},
                                ],
                                value=12,
                                clearable=False,
                                style={
                                    "width": "100px", "fontSize": "12px",
                                    "backgroundColor": "#0D1B2A",
                                },
                            ),
                            html.Label(
                                "Direction:",
                                style={"color": "#808080", "fontSize": "11px"},
                            ),
                            dcc.Dropdown(
                                id="layout-selector",
                                options=[
                                    {"label": "Left → Right", "value": "dagre-lr"},
                                    {"label": "Top → Bottom", "value": "dagre-tb"},
                                ],
                                value="dagre-lr",
                                clearable=False,
                                style={
                                    "width": "130px", "fontSize": "12px",
                                    "backgroundColor": "#0D1B2A",
                                },
                            ),
                        ],
                    ),
                ],
            ),

            # ---------------------------------------------------------------
            # Main content: Graph + Detail Panel
            # ---------------------------------------------------------------
            html.Div(
                id="main-content",
                style={
                    "display": "flex",
                    "height": "calc(100vh - 72px)",
                    "position": "relative",
                },
                children=[
                    # Graph panel (full width)
                    html.Div(
                        id="graph-panel",
                        style={
                            "flex": "1",
                            "position": "relative",
                            "background": "radial-gradient(ellipse at center, #1B2838 0%, #0D1B2A 70%)",
                        },
                        children=[
                            # Status indicator
                            html.Div(
                                id="status-bar",
                                style={
                                    "position": "absolute",
                                    "top": "12px",
                                    "left": "16px",
                                    "zIndex": "100",
                                    "display": "flex",
                                    "gap": "8px",
                                    "alignItems": "center",
                                },
                                children=[
                                    html.Div(
                                        id="status-text",
                                        style={
                                            "color": "#808080",
                                            "fontSize": "11px",
                                            "backgroundColor": "rgba(13, 27, 42, 0.8)",
                                            "padding": "4px 10px",
                                            "borderRadius": "12px",
                                            "border": "1px solid #2C3E50",
                                        },
                                        children="Enter a prompt and click Analyze to trace the model's reasoning path",
                                    ),
                                ],
                            ),

                            # Legend
                            html.Div(
                                id="legend",
                                style={
                                    "position": "absolute",
                                    "bottom": "16px",
                                    "left": "16px",
                                    "zIndex": "100",
                                    "backgroundColor": "rgba(13, 27, 42, 0.9)",
                                    "padding": "10px 14px",
                                    "borderRadius": "8px",
                                    "border": "1px solid #2C3E50",
                                    "fontSize": "10px",
                                },
                                children=[
                                    html.Div(
                                        style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                                        children=[
                                            _legend_item("#8E44AD", "Embed/Input"),
                                            _legend_item("#E67E22", "Attention"),
                                            _legend_item("#4A90D9", "MLP"),
                                            _legend_item("#27AE60", "Residual"),
                                            _legend_item("#E74C3C", "Output"),
                                        ],
                                    ),
                                ],
                            ),

                            # Cytoscape graph
                            cyto.Cytoscape(
                                id="circuit-graph",
                                elements=[],
                                stylesheet=build_stylesheet(),
                                layout=get_layout_config("dagre-lr"),
                                style={"width": "100%", "height": "100%"},
                                minZoom=0.3,
                                maxZoom=3.0,
                                responsive=True,
                                boxSelectionEnabled=False,
                            ),
                        ],
                    ),

                    # Detail panel (slide-out on node click)
                    html.Div(
                        id="detail-panel",
                        style={
                            "width": "320px",
                            "backgroundColor": "#1B2838",
                            "borderLeft": "1px solid #2C3E50",
                            "padding": "20px",
                            "overflowY": "auto",
                            "transition": "transform 0.3s ease",
                        },
                        children=[
                            html.Div(
                                style={"marginBottom": "16px"},
                                children=[
                                    html.H3(
                                        "Node Details",
                                        style={
                                            "color": "#00FFFF", "fontSize": "13px",
                                            "margin": "0 0 4px 0", "fontWeight": "600",
                                        },
                                    ),
                                    html.P(
                                        "Click a node to inspect its causal role",
                                        style={"color": "#606060", "fontSize": "11px", "margin": 0},
                                    ),
                                ],
                            ),
                            html.Div(id="node-info"),
                            html.Hr(style={"borderColor": "#2C3E50", "margin": "16px 0"}),
                            html.Div(
                                style={"marginBottom": "12px"},
                                children=[
                                    html.H3(
                                        "Causal Path Summary",
                                        style={
                                            "color": "#00FFFF", "fontSize": "13px",
                                            "margin": "0 0 8px 0", "fontWeight": "600",
                                        },
                                    ),
                                ],
                            ),
                            html.Div(id="graph-stats"),
                        ],
                    ),
                ],
            ),

            # ---------------------------------------------------------------
            # Hidden stores and intervals
            # ---------------------------------------------------------------
            dcc.Store(id="topology-store"),
            dcc.Store(id="semantics-store"),
            dcc.Store(id="all-elements-store"),
            dcc.Store(id="animation-step", data=0),
            dcc.Store(id="expanded-node-store", data=None),
            dcc.Interval(
                id="animation-interval",
                interval=250,  # ms between each node reveal
                n_intervals=0,
                disabled=True,
            ),
        ],
    )
    return layout


def _legend_item(color: str, label: str):
    """Create a legend item with colored dot and label."""
    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "4px"},
        children=[
            html.Div(
                style={
                    "width": "8px", "height": "8px",
                    "borderRadius": "50%",
                    "backgroundColor": color,
                },
            ),
            html.Span(label, style={"color": "#B0B0B0"}),
        ],
    )
