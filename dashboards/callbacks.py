"""
Dash Callback Functions for MI Toolkit Dashboard.

This module registers all Dash callbacks that handle:
    1. Node hover/click: Display semantic description, Z-scores, driving vocabulary,
       and cluster results in the info panel.
    2. Layout selector: Update Cytoscape layout algorithm.
    3. Threshold slider: Filter edges below the selected weight threshold.
    4. Run Analysis button: Trigger ReIP pipeline and update the graph.
    5. Graph statistics: Update node/edge count display.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from dash import Input, Output, State, callback, html, no_update
    import dash
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

from .stylesheet import build_stylesheet, get_layout_config


def register_callbacks(app: Any) -> None:
    """
    Register all Dash callbacks on the given Dash application instance.

    Args:
        app: A Dash application instance.
    """
    if not HAS_DASH:
        raise ImportError("Dash is required. Install with: pip install dash")

    # ------------------------------------------------------------------
    # Callback 1: Node hover — display semantic details in info panel
    # ------------------------------------------------------------------
    @app.callback(
        Output("node-info", "children"),
        Input("circuit-graph", "mouseoverNodeData"),
        Input("circuit-graph", "tapNodeData"),
        prevent_initial_call=True,
    )
    def display_node_info(hover_data: Optional[Dict], tap_data: Optional[Dict]):
        """
        Display detailed semantic information for a hovered or clicked node.

        Priority: tap (click) > hover.
        """
        node_data = tap_data or hover_data
        if not node_data:
            return html.P(
                "Hover or click a node to view details.",
                style={"color": "#808080", "fontSize": "12px"},
            )

        layer = node_data.get("layer", "?")
        component = node_data.get("component", "?")
        token = node_data.get("token", "")
        score = node_data.get("score", 0.0)
        full_label = node_data.get("full_label", "")
        cluster_info = node_data.get("cluster_info", "")
        node_id = node_data.get("id", "")

        children = [
            html.Div(
                style={
                    "backgroundColor": "#0F3460",
                    "padding": "10px",
                    "borderRadius": "4px",
                    "marginBottom": "8px",
                },
                children=[
                    html.P(
                        f"Layer {layer} — {component.upper()}",
                        style={"color": "#4A90D9", "fontSize": "13px",
                               "fontWeight": "bold", "margin": "0 0 4px 0"},
                    ),
                    html.P(
                        f"Token: \"{token}\"" if token else "Token: N/A",
                        style={"color": "#E0E0E0", "fontSize": "12px", "margin": "2px 0"},
                    ),
                    html.P(
                        f"Relevance Score: {score:.4f}",
                        style={"color": "#F39C12", "fontSize": "12px",
                               "fontWeight": "bold", "margin": "2px 0"},
                    ),
                ],
            ),
        ]

        if full_label:
            children.append(
                html.Div(
                    style={"marginBottom": "8px"},
                    children=[
                        html.P(
                            "Semantic Label:",
                            style={"color": "#B0B0B0", "fontSize": "11px",
                                   "margin": "0 0 2px 0"},
                        ),
                        html.P(
                            full_label,
                            style={"color": "#E0E0E0", "fontSize": "11px",
                                   "backgroundColor": "#0A2040",
                                   "padding": "6px", "borderRadius": "3px",
                                   "wordBreak": "break-word"},
                        ),
                    ],
                )
            )

        if cluster_info:
            children.append(
                html.Div(
                    style={"marginBottom": "8px"},
                    children=[
                        html.P(
                            "Circuit Cluster:",
                            style={"color": "#B0B0B0", "fontSize": "11px",
                                   "margin": "0 0 2px 0"},
                        ),
                        html.P(
                            cluster_info,
                            style={"color": "#27AE60", "fontSize": "11px",
                                   "backgroundColor": "#0A2040",
                                   "padding": "6px", "borderRadius": "3px",
                                   "wordBreak": "break-word"},
                        ),
                    ],
                )
            )

        children.append(
            html.P(
                f"Node ID: {node_id}",
                style={"color": "#606060", "fontSize": "10px",
                       "wordBreak": "break-all"},
            )
        )

        return children

    # ------------------------------------------------------------------
    # Callback 2: Layout selector — update Cytoscape layout
    # ------------------------------------------------------------------
    @app.callback(
        Output("circuit-graph", "layout"),
        Input("layout-selector", "value"),
        prevent_initial_call=True,
    )
    def update_layout(layout_name: str) -> Dict:
        """Update the Cytoscape graph layout algorithm."""
        return get_layout_config(layout_name or "dagre")

    # ------------------------------------------------------------------
    # Callback 3: Threshold slider — filter low-weight edges
    # ------------------------------------------------------------------
    @app.callback(
        Output("circuit-graph", "elements"),
        Input("threshold-slider", "value"),
        State("topology-store", "data"),
        State("semantics-store", "data"),
        prevent_initial_call=True,
    )
    def filter_edges_by_threshold(
        threshold: float,
        topology_data: Optional[Dict],
        semantics_data: Optional[Dict],
    ) -> List[Dict]:
        """
        Filter graph elements to only show edges above the weight threshold.
        """
        if not topology_data:
            return []

        from .layout import topology_to_cytoscape_elements

        # Rebuild elements from stored topology
        all_elements = topology_to_cytoscape_elements(
            topology_data,
            semantic_labels=semantics_data or {},
        )

        # Filter edges below threshold; keep all nodes
        filtered = []
        for element in all_elements:
            data = element.get("data", {})
            if "source" in data and "target" in data:
                # It's an edge
                if data.get("weight", 0.0) >= threshold:
                    filtered.append(element)
            else:
                # It's a node
                filtered.append(element)

        return filtered

    # ------------------------------------------------------------------
    # Callback 4: Run Analysis button — trigger ReIP and update graph
    # ------------------------------------------------------------------
    @app.callback(
        Output("topology-store", "data"),
        Output("graph-loading-output", "children"),
        Input("run-analysis-btn", "n_clicks"),
        State("clean-prompt-input", "value"),
        State("corrupted-prompt-input", "value"),
        prevent_initial_call=True,
    )
    def run_analysis(
        n_clicks: Optional[int],
        clean_prompt: Optional[str],
        corrupted_prompt: Optional[str],
    ):
        """
        Trigger the ReIP analysis pipeline and store topology data.

        In production, this callback invokes the ReIPPipeline. In the
        dashboard demo mode, it returns a placeholder topology.
        """
        if not n_clicks or not clean_prompt or not corrupted_prompt:
            return no_update, no_update

        # Placeholder topology for dashboard demo
        # In production: invoke ReIPPipeline.run(clean_prompt, corrupted_prompt)
        demo_topology = {
            "nodes": [
                {"id": "embed__pos0", "layer": -1, "component": "embed",
                 "token": clean_prompt.split()[0] if clean_prompt else "?",
                 "score": 0.5, "position": 0},
                {"id": "blocks.0.hook_mlp_out__pos0", "layer": 0, "component": "mlp",
                 "token": "mlp_0", "score": 0.7, "position": 0},
                {"id": "blocks.1.hook_attn_out__pos0", "layer": 1, "component": "attn",
                 "token": "attn_1", "score": 0.9, "position": 0},
                {"id": "blocks.2.hook_mlp_out__pos0", "layer": 2, "component": "mlp",
                 "token": "mlp_2", "score": 0.6, "position": 0},
            ],
            "edges": [
                {"source": "embed__pos0",
                 "target": "blocks.0.hook_mlp_out__pos0", "weight": 0.5},
                {"source": "blocks.0.hook_mlp_out__pos0",
                 "target": "blocks.1.hook_attn_out__pos0", "weight": 0.8},
                {"source": "blocks.1.hook_attn_out__pos0",
                 "target": "blocks.2.hook_mlp_out__pos0", "weight": 0.6},
            ],
        }

        return demo_topology, ""

    # ------------------------------------------------------------------
    # Callback 5: Graph statistics display
    # ------------------------------------------------------------------
    @app.callback(
        Output("graph-stats", "children"),
        Input("circuit-graph", "elements"),
        prevent_initial_call=True,
    )
    def update_graph_stats(elements: Optional[List[Dict]]):
        """Display node and edge count statistics."""
        if not elements:
            return html.P("No graph loaded.", style={"color": "#808080", "fontSize": "12px"})

        n_nodes = sum(
            1 for e in elements
            if "source" not in e.get("data", {})
        )
        n_edges = sum(
            1 for e in elements
            if "source" in e.get("data", {})
        )

        return html.Div(
            children=[
                html.P(
                    f"Nodes: {n_nodes}",
                    style={"color": "#4A90D9", "fontSize": "12px", "margin": "2px 0"},
                ),
                html.P(
                    f"Edges: {n_edges}",
                    style={"color": "#E67E22", "fontSize": "12px", "margin": "2px 0"},
                ),
            ]
        )
