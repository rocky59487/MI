"""
Dashboard Layout and Data Conversion for MI Toolkit.

This module converts ReIP topology graphs and WeightLens/CircuitLens semantic
labels into Cytoscape-compatible elements lists, and defines the Dash
application layout.

Data flow:
    ReIP topology (NetworkX DiGraph / dict)
    + WeightLens FeatureSemantics (JSON)
    + CircuitLens ClusterResult (dict)
    ↓
    Cytoscape elements: [{'data': {...}}, ...]
    ↓
    Dash layout with Cytoscape graph + info panel
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


def topology_to_cytoscape_elements(
    topology: Union[Dict, Any],
    semantic_labels: Optional[Dict[str, str]] = None,
    cluster_labels: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Convert a ReIP topology graph to Cytoscape elements format.

    Args:
        topology: Either a NetworkX DiGraph or a dict with 'nodes' and 'edges' keys.
        semantic_labels: Optional dict mapping node_id to WeightLens semantic label.
        cluster_labels: Optional dict mapping node_id to CircuitLens cluster label.

    Returns:
        List of Cytoscape element dicts with 'data' and optional 'position' keys.
        Format:
            [
                {'data': {'id': 'layer_0_mlp__pos0', 'label': 'run, walk',
                          'score': 0.85, 'component': 'mlp', 'layer': 0, ...}},
                {'data': {'source': 'node_a', 'target': 'node_b', 'weight': 0.72}},
                ...
            ]
    """
    elements = []

    # Normalize topology to dict format
    if hasattr(topology, "nodes") and hasattr(topology, "edges"):
        # NetworkX DiGraph
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

    # Build node elements
    for node in nodes:
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

        # Truncate label for display
        display_label = semantic[:20] + "…" if len(semantic) > 20 else semantic
        if not display_label:
            display_label = f"L{layer}_{component[:3]}"

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
            },
            "classes": component,
        })

    # Build edge elements
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        weight = edge.get("weight", 0.0)

        if not source or not target:
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


def build_layout(app_title: str = "MI Toolkit — Circuit Visualization") -> Any:
    """
    Build the complete Dash application layout.

    Returns the top-level Dash layout component tree including:
        - Header bar with title and controls
        - Cytoscape graph panel (main visualization)
        - Info panel (node details on hover/click)
        - Control panel (layout selector, threshold slider)

    Args:
        app_title: Title string displayed in the header.

    Returns:
        Dash HTML component tree.
    """
    if not HAS_DASH:
        raise ImportError(
            "Dash and dash-cytoscape are required for the dashboard. "
            "Install with: pip install dash dash-cytoscape"
        )

    layout = html.Div(
        id="app-container",
        style={"fontFamily": "monospace", "backgroundColor": "#1A1A2E", "minHeight": "100vh"},
        children=[
            # ---------------------------------------------------------------
            # Header
            # ---------------------------------------------------------------
            html.Div(
                id="header",
                style={
                    "backgroundColor": "#16213E",
                    "padding": "12px 24px",
                    "borderBottom": "1px solid #0F3460",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                },
                children=[
                    html.H1(
                        app_title,
                        style={"color": "#E0E0E0", "fontSize": "18px", "margin": 0},
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "16px", "alignItems": "center"},
                        children=[
                            html.Label("Layout:", style={"color": "#B0B0B0", "fontSize": "13px"}),
                            dcc.Dropdown(
                                id="layout-selector",
                                options=[
                                    {"label": "Dagre (Hierarchical)", "value": "dagre"},
                                    {"label": "Breadth-First", "value": "breadthfirst"},
                                    {"label": "CoSE (Force-Directed)", "value": "cose"},
                                    {"label": "Grid", "value": "grid"},
                                ],
                                value="dagre",
                                clearable=False,
                                style={"width": "200px", "fontSize": "13px"},
                            ),
                            html.Label("Threshold:", style={"color": "#B0B0B0", "fontSize": "13px"}),
                            dcc.Slider(
                                id="threshold-slider",
                                min=0.0,
                                max=1.0,
                                step=0.05,
                                value=0.1,
                                marks={0: "0", 0.5: "0.5", 1: "1"},
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="threshold-slider",
                            ),
                        ],
                    ),
                ],
            ),

            # ---------------------------------------------------------------
            # Main content area
            # ---------------------------------------------------------------
            html.Div(
                id="main-content",
                style={"display": "flex", "height": "calc(100vh - 60px)"},
                children=[
                    # Cytoscape graph panel
                    html.Div(
                        id="graph-panel",
                        style={"flex": "1", "position": "relative"},
                        children=[
                            cyto.Cytoscape(
                                id="circuit-graph",
                                elements=[],
                                stylesheet=build_stylesheet(),
                                layout=get_layout_config("dagre"),
                                style={"width": "100%", "height": "100%"},
                                minZoom=0.1,
                                maxZoom=3.0,
                                responsive=True,
                            ),
                            # Loading overlay
                            dcc.Loading(
                                id="graph-loading",
                                type="circle",
                                color="#4A90D9",
                                children=html.Div(id="graph-loading-output"),
                            ),
                        ],
                    ),

                    # Info panel
                    html.Div(
                        id="info-panel",
                        style={
                            "width": "320px",
                            "backgroundColor": "#16213E",
                            "borderLeft": "1px solid #0F3460",
                            "padding": "16px",
                            "overflowY": "auto",
                            "color": "#E0E0E0",
                        },
                        children=[
                            html.H3(
                                "Feature Details",
                                style={"color": "#4A90D9", "fontSize": "14px", "marginTop": 0},
                            ),
                            html.Div(
                                id="node-info",
                                children=[
                                    html.P(
                                        "Click or hover on a node to view its semantic details.",
                                        style={"color": "#808080", "fontSize": "12px"},
                                    )
                                ],
                            ),
                            html.Hr(style={"borderColor": "#0F3460"}),
                            html.H3(
                                "Graph Statistics",
                                style={"color": "#4A90D9", "fontSize": "14px"},
                            ),
                            html.Div(id="graph-stats"),
                            html.Hr(style={"borderColor": "#0F3460"}),
                            html.H3(
                                "Analysis Controls",
                                style={"color": "#4A90D9", "fontSize": "14px"},
                            ),
                            html.Div(
                                children=[
                                    html.Label(
                                        "Clean Prompt:",
                                        style={"color": "#B0B0B0", "fontSize": "12px"},
                                    ),
                                    dcc.Textarea(
                                        id="clean-prompt-input",
                                        placeholder="Enter clean prompt...",
                                        style={
                                            "width": "100%",
                                            "height": "60px",
                                            "backgroundColor": "#0F3460",
                                            "color": "#E0E0E0",
                                            "border": "1px solid #4A90D9",
                                            "fontSize": "11px",
                                        },
                                    ),
                                    html.Label(
                                        "Corrupted Prompt:",
                                        style={"color": "#B0B0B0", "fontSize": "12px", "marginTop": "8px"},
                                    ),
                                    dcc.Textarea(
                                        id="corrupted-prompt-input",
                                        placeholder="Enter corrupted prompt...",
                                        style={
                                            "width": "100%",
                                            "height": "60px",
                                            "backgroundColor": "#0F3460",
                                            "color": "#E0E0E0",
                                            "border": "1px solid #4A90D9",
                                            "fontSize": "11px",
                                        },
                                    ),
                                    html.Button(
                                        "Run ReIP Analysis",
                                        id="run-analysis-btn",
                                        style={
                                            "marginTop": "8px",
                                            "width": "100%",
                                            "backgroundColor": "#4A90D9",
                                            "color": "white",
                                            "border": "none",
                                            "padding": "8px",
                                            "cursor": "pointer",
                                            "fontSize": "12px",
                                        },
                                    ),
                                ]
                            ),
                        ],
                    ),
                ],
            ),

            # Hidden data store for topology
            dcc.Store(id="topology-store"),
            dcc.Store(id="semantics-store"),
        ],
    )
    return layout
