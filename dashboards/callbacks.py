"""
Dash Callbacks for MI Circuit Explorer — Directional Causal Flow.

This module implements the interactive logic for the dashboard.
All analysis runs REAL model inference — no mock data, no fallback.

Callbacks:
    0. init_backend_status: Show backend readiness on page load
    1. toggle_safety_mode: Show/hide safety-specific UI elements
    2. fill_safety_example: Fill prompt from safety example buttons
    3. run_analysis_callback: Execute real ReIP + CircuitLens + WeightLens
    4. animate_reveal: Reveal nodes one-by-one (typewriter effect)
    5. display_node_details: Show detailed info on node click
    6. update_layout: Update graph layout direction
    7. update_topn: Re-filter topology with new top-N value
    8. update_graph_stats: Display causal path summary statistics
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

try:
    from dash import Input, Output, State, callback, html, no_update, ctx
    from dash.exceptions import PreventUpdate
    import dash
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

from .stylesheet import build_stylesheet, get_layout_config
from .layout import topology_to_cytoscape_elements, SAFETY_PROMPT_EXAMPLES
from .backend import run_analysis, get_backend_status, BackendError


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
    # Callback 0: Show backend status on page load
    # ------------------------------------------------------------------
    @app.callback(
        Output("mode-badge", "children"),
        Output("mode-badge", "style"),
        Output("mode-indicator-dot", "style"),
        Output("status-text", "children"),
        Input("app-container", "id"),  # triggers on page load
    )
    def init_backend_status(_):
        """Display the backend readiness status on page load."""
        status = get_backend_status()

        if status["ready"]:
            device = status["device"].upper()
            badge_style = {
                "fontSize": "10px",
                "padding": "3px 8px",
                "borderRadius": "10px",
                "backgroundColor": "#27AE60" if device == "CUDA" else "#2980B9",
                "color": "#FFFFFF",
                "fontWeight": "600",
                "letterSpacing": "0.3px",
            }
            dot_style = {
                "width": "8px", "height": "8px",
                "borderRadius": "50%",
                "backgroundColor": "#27AE60" if device == "CUDA" else "#2980B9",
                "boxShadow": f"0 0 8px {'#27AE60' if device == 'CUDA' else '#2980B9'}",
            }
            badge_text = f"LIVE \u2014 {device}"
            status_msg = f"Backend ready ({device}) \u2014 Enter a prompt and click Analyze"
        else:
            badge_style = {
                "fontSize": "10px",
                "padding": "3px 8px",
                "borderRadius": "10px",
                "backgroundColor": "#E74C3C",
                "color": "#FFFFFF",
                "fontWeight": "600",
                "letterSpacing": "0.3px",
            }
            dot_style = {
                "width": "8px", "height": "8px",
                "borderRadius": "50%",
                "backgroundColor": "#E74C3C",
                "boxShadow": "0 0 8px #E74C3C",
            }
            badge_text = "NOT READY"
            status_msg = f"\u26a0 Backend not ready: {status['message']}"

        return badge_text, badge_style, dot_style, status_msg

    # ------------------------------------------------------------------
    # Callback 1: Analysis mode selector — show/hide safety examples
    # ------------------------------------------------------------------
    @app.callback(
        Output("safety-examples-bar", "style"),
        Output("safety-panel", "style"),
        Output("clean-prompt-input", "placeholder"),
        Input("analysis-mode-selector", "value"),
    )
    def toggle_safety_mode(mode):
        """Show/hide safety-specific UI elements based on mode selection."""
        if mode == "safety":
            bar_style = {
                "backgroundColor": "#1a1a2e",
                "padding": "8px 24px",
                "borderBottom": "1px solid #E74C3C",
                "display": "flex",
                "alignItems": "center",
                "gap": "8px",
                "flexWrap": "wrap",
            }
            panel_style = {"display": "block"}
            placeholder = (
                "Enter a dangerous agent action to analyze "
                "(e.g., 'The agent decided to delete the file')"
            )
        else:
            bar_style = {
                "backgroundColor": "#1a1a2e",
                "padding": "8px 24px",
                "borderBottom": "1px solid #2C3E50",
                "display": "none",
            }
            panel_style = {"display": "none"}
            placeholder = (
                "Enter prompt to analyze "
                "(e.g., 'When Mary and John went to the store, John gave a drink to')"
            )

        return bar_style, panel_style, placeholder

    # ------------------------------------------------------------------
    # Callback 2: Safety example button clicks — fill prompt
    # ------------------------------------------------------------------
    @app.callback(
        Output("clean-prompt-input", "value"),
        [Input(f"safety-example-{i}", "n_clicks") for i in range(4)],
        prevent_initial_call=True,
    )
    def fill_safety_example(*n_clicks):
        """Fill the prompt input with a safety example when clicked."""
        triggered = ctx.triggered_id
        if triggered:
            for i in range(4):
                if triggered == f"safety-example-{i}" and n_clicks[i]:
                    return SAFETY_PROMPT_EXAMPLES[i]
        return no_update

    # ------------------------------------------------------------------
    # Callback 3: Run Analysis — real backend inference
    # ------------------------------------------------------------------
    @app.callback(
        Output("topology-store", "data"),
        Output("all-elements-store", "data"),
        Output("animation-step", "data"),
        Output("animation-interval", "disabled"),
        Output("animation-interval", "n_intervals"),
        Output("circuit-graph", "elements", allow_duplicate=True),
        Output("status-text", "children", allow_duplicate=True),
        Output("analysis-result-store", "data"),
        Output("safety-info-store", "data"),
        Output("metadata-display", "children"),
        Output("safety-explanation", "children"),
        Output("error-banner", "style"),
        Output("error-message", "children"),
        Input("run-analysis-btn", "n_clicks"),
        State("clean-prompt-input", "value"),
        State("topn-selector", "value"),
        State("analysis-mode-selector", "value"),
        prevent_initial_call=True,
    )
    def run_analysis_callback(n_clicks, prompt, top_n, analysis_mode):
        """Trigger real analysis: call backend, generate topology, start animation."""
        if not n_clicks:
            return (no_update,) * 13

        if not prompt or not prompt.strip():
            return (
                no_update, no_update, no_update, no_update, no_update,
                no_update,
                "\u26a0 Please enter a prompt before running analysis.",
                no_update, no_update, no_update, no_update,
                {"display": "none"}, "",
            )

        top_n = top_n or 12

        # Check backend readiness first
        status = get_backend_status()
        if not status["ready"]:
            error_msg = (
                f"Backend is not ready:\n{status['message']}\n\n"
                f"Missing dependencies:\n"
                + "\n".join(
                    f"  \u2717 {k}" for k, v in status.get("deps", {}).items() if not v
                )
            )
            error_banner_style = {
                "display": "block",
                "position": "absolute",
                "top": "50px",
                "left": "50%",
                "transform": "translateX(-50%)",
                "zIndex": "200",
                "backgroundColor": "rgba(231, 76, 60, 0.15)",
                "border": "1px solid #E74C3C",
                "borderRadius": "8px",
                "padding": "16px 24px",
                "maxWidth": "600px",
                "width": "90%",
            }
            return (
                no_update, no_update, no_update, no_update, no_update,
                no_update,
                "\u26a0 Backend not ready — see error panel",
                no_update, no_update, no_update, no_update,
                error_banner_style, error_msg,
            )

        # Run real inference
        try:
            result = run_analysis(
                prompt=prompt,
                top_n=top_n,
                model_name="gpt2",
                analysis_mode=analysis_mode,
            )
        except (BackendError, ValueError) as e:
            error_msg = str(e)
            error_banner_style = {
                "display": "block",
                "position": "absolute",
                "top": "50px",
                "left": "50%",
                "transform": "translateX(-50%)",
                "zIndex": "200",
                "backgroundColor": "rgba(231, 76, 60, 0.15)",
                "border": "1px solid #E74C3C",
                "borderRadius": "8px",
                "padding": "16px 24px",
                "maxWidth": "600px",
                "width": "90%",
            }
            return (
                no_update, no_update, no_update, no_update, no_update,
                no_update,
                f"\u26a0 Analysis failed: {str(e)[:80]}...",
                no_update, no_update, no_update, no_update,
                error_banner_style, error_msg,
            )
        except Exception as e:
            error_msg = f"Unexpected error during analysis:\n{traceback.format_exc()}"
            error_banner_style = {
                "display": "block",
                "position": "absolute",
                "top": "50px",
                "left": "50%",
                "transform": "translateX(-50%)",
                "zIndex": "200",
                "backgroundColor": "rgba(231, 76, 60, 0.15)",
                "border": "1px solid #E74C3C",
                "borderRadius": "8px",
                "padding": "16px 24px",
                "maxWidth": "600px",
                "width": "90%",
            }
            return (
                no_update, no_update, no_update, no_update, no_update,
                no_update,
                f"\u26a0 Unexpected error: {type(e).__name__}",
                no_update, no_update, no_update, no_update,
                error_banner_style, error_msg,
            )

        # Success — build topology dict for Cytoscape conversion
        topology = {
            "nodes": result["nodes"],
            "edges": result["edges"],
        }

        elements = topology_to_cytoscape_elements(
            topology,
            semantic_labels=result.get("semantic_labels", {}),
            top_n=top_n,
            safety_info=result.get("safety_info"),
        )

        # Build metadata display
        metadata = result.get("metadata", {})
        metadata_children = _build_metadata_display(metadata)

        # Build safety explanation (if applicable)
        safety_children = []
        safety_info = result.get("safety_info")
        if safety_info and analysis_mode == "safety":
            safety_children = _build_safety_explanation(safety_info)

        # Status text
        device = metadata.get("device", "cpu").upper()
        runtime = metadata.get("runtime_seconds", 0)
        status_text = (
            f"[{device}] Tracing causal path... "
            f"({runtime:.2f}s)"
        )

        # Hide error banner on success
        error_banner_style = {"display": "none"}

        return (
            topology,              # topology-store
            elements,              # all-elements-store
            0,                     # animation-step reset
            False,                 # animation-interval enabled
            0,                     # reset n_intervals
            [],                    # clear graph
            status_text,           # status-text
            result,                # analysis-result-store
            safety_info,           # safety-info-store
            metadata_children,     # metadata-display
            safety_children,       # safety-explanation
            error_banner_style,    # error-banner hidden
            "",                    # error-message cleared
        )

    # ------------------------------------------------------------------
    # Callback 4: Animation interval — reveal nodes sequentially
    # ------------------------------------------------------------------
    @app.callback(
        Output("circuit-graph", "elements"),
        Output("animation-interval", "disabled", allow_duplicate=True),
        Output("status-text", "children", allow_duplicate=True),
        Input("animation-interval", "n_intervals"),
        State("all-elements-store", "data"),
        State("analysis-result-store", "data"),
        prevent_initial_call=True,
    )
    def animate_reveal(n_intervals, all_elements, analysis_result):
        """Reveal nodes and edges one step at a time."""
        if not all_elements:
            return no_update, True, no_update

        # Separate nodes and edges
        nodes = [e for e in all_elements if "source" not in e.get("data", {})]
        edges = [e for e in all_elements if "source" in e.get("data", {})]

        total_steps = len(nodes) + len(edges)

        if n_intervals >= total_steps:
            # Animation complete — show all
            revealed = []
            for e in all_elements:
                e_copy = dict(e)
                e_copy["classes"] = (e_copy.get("classes", "") + " visible").strip()
                revealed.append(e_copy)

            n_nodes = len(nodes)
            n_edges = len(edges)
            device = "CPU"
            runtime = ""
            if analysis_result and analysis_result.get("metadata"):
                device = analysis_result["metadata"].get("device", "cpu").upper()
                rt = analysis_result["metadata"].get("runtime_seconds", 0)
                runtime = f" ({rt:.2f}s)" if rt else ""

            status = (
                f"[{device}] Analysis complete \u2014 "
                f"{n_nodes} key nodes, {n_edges} causal connections{runtime}"
            )
            return revealed, True, status

        # Reveal up to current step
        revealed = []
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

        if edges_to_show > 0:
            status = f"Connecting edges {edges_to_show}/{len(edges)}..."
        else:
            status = f"Revealing node {nodes_to_show}/{len(nodes)}..."

        return revealed, False, status

    # ------------------------------------------------------------------
    # Callback 5: Node click — show detailed info
    # ------------------------------------------------------------------
    @app.callback(
        Output("node-info", "children"),
        Output("expanded-node-store", "data"),
        Input("circuit-graph", "tapNodeData"),
        State("topology-store", "data"),
        State("analysis-result-store", "data"),
        State("analysis-mode-selector", "value"),
        prevent_initial_call=True,
    )
    def display_node_details(tap_data, topology_data, analysis_result, analysis_mode):
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
        is_dangerous = tap_data.get("is_dangerous", False)

        score_pct = min(100, int(float(score) * 100))
        score_color = "#FF0000" if is_dangerous else (
            "#E74C3C" if score > 0.8 else "#F39C12" if score > 0.5 else "#4A90D9"
        )

        children = []

        if is_dangerous and analysis_mode == "safety":
            children.append(
                html.Div(
                    style={
                        "backgroundColor": "rgba(255, 0, 0, 0.15)",
                        "border": "1px solid #FF0000",
                        "borderRadius": "6px",
                        "padding": "8px 12px",
                        "marginBottom": "12px",
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "8px",
                    },
                    children=[
                        html.Span("\u26a0\ufe0f", style={"fontSize": "16px"}),
                        html.Span(
                            "DANGEROUS NODE \u2014 contributes to hazardous decision",
                            style={"color": "#FF4444", "fontSize": "11px", "fontWeight": "600"},
                        ),
                    ],
                )
            )

        # Node identity card
        children.append(
            html.Div(
                style={
                    "backgroundColor": "#0D1B2A",
                    "padding": "14px",
                    "borderRadius": "8px",
                    "marginBottom": "12px",
                    "border": f"1px solid {'#FF0000' if is_dangerous else '#2C3E50'}",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                        children=[
                            html.Span(
                                f"Layer {layer}",
                                style={"color": "#FF0000" if is_dangerous else "#00FFFF",
                                       "fontSize": "14px", "fontWeight": "700"},
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
            )
        )

        # ReIP Score bar
        children.append(
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
            )
        )

        # Semantic label (WeightLens output)
        if full_label:
            children.append(
                html.Div(
                    style={"marginBottom": "12px"},
                    children=[
                        html.P(
                            "Semantic Role (WeightLens)",
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
        interpretation = _get_causal_interpretation(
            layer, component, token, is_dangerous, analysis_mode
        )
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
                                "color": "#FF8888" if is_dangerous else "#B0B0B0",
                                "fontSize": "11px",
                                "backgroundColor": "#0D1B2A",
                                "padding": "8px 10px", "borderRadius": "6px",
                                "border": f"1px solid {'#FF0000' if is_dangerous else '#2C3E50'}",
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
    # Callback 6: Layout selector
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
    # Callback 7: Top-N selector — re-filter
    # ------------------------------------------------------------------
    @app.callback(
        Output("circuit-graph", "elements", allow_duplicate=True),
        Output("all-elements-store", "data", allow_duplicate=True),
        Input("topn-selector", "value"),
        State("topology-store", "data"),
        State("safety-info-store", "data"),
        State("analysis-result-store", "data"),
        prevent_initial_call=True,
    )
    def update_topn(top_n, topology_data, safety_info, analysis_result):
        """Re-filter topology with new top-N value."""
        if not topology_data:
            return no_update, no_update

        top_n = top_n or 12

        semantic_labels = {}
        if analysis_result and analysis_result.get("semantic_labels"):
            semantic_labels = analysis_result["semantic_labels"]

        elements = topology_to_cytoscape_elements(
            topology_data,
            semantic_labels=semantic_labels,
            top_n=top_n,
            safety_info=safety_info,
        )

        # Show all immediately (no re-animation)
        visible_elements = []
        for e in elements:
            e_copy = dict(e)
            e_copy["classes"] = (e_copy.get("classes", "") + " visible").strip()
            visible_elements.append(e_copy)

        return visible_elements, elements

    # ------------------------------------------------------------------
    # Callback 8: Graph statistics
    # ------------------------------------------------------------------
    @app.callback(
        Output("graph-stats", "children"),
        Input("circuit-graph", "elements"),
        State("topology-store", "data"),
        State("safety-info-store", "data"),
        prevent_initial_call=True,
    )
    def update_graph_stats(elements, topology_data, safety_info):
        """Display causal path summary statistics."""
        if not elements:
            return html.P(
                "No analysis loaded.",
                style={"color": "#606060", "fontSize": "11px"},
            )

        n_nodes = sum(1 for e in elements if "source" not in e.get("data", {}))
        n_edges = sum(1 for e in elements if "source" in e.get("data", {}))

        layers = [
            e["data"].get("layer", 0)
            for e in elements
            if "source" not in e.get("data", {})
        ]
        min_layer = min(layers) if layers else 0
        max_layer = max(layers) if layers else 0

        stats = [
            _stat_row("Active Nodes", str(n_nodes), "#4A90D9"),
            _stat_row("Causal Edges", str(n_edges), "#E67E22"),
            _stat_row("Layer Span", f"L{min_layer} \u2192 L{max_layer}", "#27AE60"),
            _stat_row("Path Depth", f"{max_layer - min_layer + 1} layers", "#8E44AD"),
        ]

        if safety_info:
            n_dangerous = len(safety_info.get("dangerous_nodes", []))
            n_dangerous_edges = len(safety_info.get("dangerous_edges", []))
            stats.append(_stat_row("\u26a0 Dangerous Nodes", str(n_dangerous), "#FF0000"))
            stats.append(_stat_row("\u26a0 Danger Path Edges", str(n_dangerous_edges), "#FF0000"))

        return html.Div(children=stats)


# ============================================================================
# Helper Functions
# ============================================================================

def _build_metadata_display(metadata: Dict) -> List:
    """Build metadata display for the detail panel."""
    if not metadata:
        return [html.P("No metadata available.", style={"color": "#606060", "fontSize": "11px"})]

    items = []
    display_fields = [
        ("device", "Device"),
        ("model", "Model"),
        ("runtime_seconds", "Runtime"),
        ("n_nodes", "Nodes Analyzed"),
        ("n_edges", "Edges Computed"),
        ("n_layers", "Model Layers"),
        ("d_model", "d_model"),
        ("n_heads", "Attention Heads"),
        ("scoring_formula", "Scoring Formula"),
    ]

    for key, label in display_fields:
        value = metadata.get(key)
        if value is None:
            continue
        if key == "runtime_seconds" and isinstance(value, (int, float)):
            value = f"{value:.3f}s"
        elif key == "device":
            value = value.upper()

        color = "#27AE60" if key == "device" and value == "CUDA" else "#2980B9" if key == "device" else "#B0B0B0"

        items.append(
            html.Div(
                style={
                    "display": "flex", "justifyContent": "space-between",
                    "padding": "4px 0", "borderBottom": "1px solid #2C3E50",
                },
                children=[
                    html.Span(label, style={"color": "#808080", "fontSize": "10px"}),
                    html.Span(str(value), style={"color": color, "fontSize": "10px", "fontWeight": "600"}),
                ],
            )
        )

    return items


def _build_safety_explanation(safety_info: Dict) -> List:
    """Build the safety explanation panel content."""
    if not safety_info:
        return []

    children = []

    n_dangerous = len(safety_info.get("dangerous_nodes", []))
    children.append(
        html.Div(
            style={
                "backgroundColor": "rgba(255, 0, 0, 0.1)",
                "border": "1px solid rgba(255, 0, 0, 0.3)",
                "borderRadius": "6px",
                "padding": "10px",
                "marginBottom": "10px",
            },
            children=[
                html.P(
                    f"\u26a0 {n_dangerous} dangerous node(s) identified",
                    style={"color": "#FF4444", "fontSize": "12px", "fontWeight": "600", "margin": "0 0 6px 0"},
                ),
                html.P(
                    f"Threshold: score \u2265 {safety_info.get('threshold', 0.0):.4f}",
                    style={"color": "#FF8888", "fontSize": "10px", "margin": 0},
                ),
            ],
        )
    )

    attention_heads = safety_info.get("attention_heads", [])
    if attention_heads:
        children.append(
            html.P(
                "Key Attention Heads (WeightLens):",
                style={"color": "#FF8888", "fontSize": "11px", "fontWeight": "600", "margin": "8px 0 4px 0"},
            )
        )
        for head in attention_heads[:3]:
            label = head.get("label", "")
            label_short = label[:40] + "..." if len(label) > 40 else label
            children.append(
                html.Div(
                    style={
                        "backgroundColor": "rgba(255, 0, 0, 0.05)",
                        "padding": "6px 10px",
                        "borderRadius": "4px",
                        "marginBottom": "4px",
                        "borderLeft": "3px solid #FF0000",
                    },
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between"},
                            children=[
                                html.Span(
                                    f"Layer {head['layer']} Head {head['head']}",
                                    style={"color": "#FF4444", "fontSize": "11px", "fontWeight": "600"},
                                ),
                                html.Span(
                                    f"score: {head['score']:.4f}",
                                    style={"color": "#FF8888", "fontSize": "10px"},
                                ),
                            ],
                        ),
                        html.Span(
                            f'Token: "{head["token"]}"',
                            style={"color": "#CCCCCC", "fontSize": "10px"},
                        ),
                        html.Div(
                            label_short,
                            style={"color": "#FF8888", "fontSize": "9px", "marginTop": "2px", "fontStyle": "italic"},
                        ) if label_short else html.Span(),
                    ],
                )
            )

    explanation = safety_info.get("explanation", "")
    if explanation:
        children.append(
            html.Div(
                style={"marginTop": "10px"},
                children=[
                    html.P(
                        "Detailed Analysis (ReIP + WeightLens):",
                        style={"color": "#FF8888", "fontSize": "11px", "fontWeight": "600", "margin": "0 0 6px 0"},
                    ),
                    html.Div(
                        explanation,
                        style={
                            "color": "#CCCCCC",
                            "fontSize": "10px",
                            "backgroundColor": "rgba(13, 27, 42, 0.8)",
                            "padding": "10px",
                            "borderRadius": "6px",
                            "border": "1px solid #2C3E50",
                            "lineHeight": "1.5",
                            "whiteSpace": "pre-wrap",
                        },
                    ),
                ],
            )
        )

    return children


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


def _get_causal_interpretation(
    layer: int,
    component: str,
    token: str,
    is_dangerous: bool = False,
    analysis_mode: str = "general",
) -> str:
    """Generate a human-readable causal interpretation for a node."""
    if is_dangerous and analysis_mode == "safety":
        if component == "attn":
            return (
                f"DANGER: Attention head at layer {layer} is a key contributor to the "
                f"dangerous decision. It attends to token \"{token}\" and propagates "
                f"action-related features that lead to the hazardous output. "
                f"WeightLens indicates this head extracts destructive action semantics."
            )
        elif component == "mlp":
            return (
                f"DANGER: MLP at layer {layer} amplifies dangerous action features. "
                f"It projects the representation at \"{token}\" into a subspace associated "
                f"with destructive operations. This is a critical intervention point."
            )
        else:
            return (
                f"DANGER: Component at layer {layer} ({component}) contributes to "
                f"the dangerous output through token \"{token}\"."
            )

    if layer == -1 or component == "embed":
        return f"Initial representation of \"{token}\" before transformer processing."
    elif component == "attn" and layer <= 2:
        return f"Early attention (L{layer}) — positional/syntactic pattern matching."
    elif component == "attn" and layer <= 5:
        return f"Mid-layer attention (L{layer}) — entity relationship tracking."
    elif component == "attn":
        return f"Late attention (L{layer}) — final name-moving / copying operation."
    elif component == "mlp" and layer <= 2:
        return f"Early MLP (L{layer}) — enriching token representations."
    elif component == "mlp":
        return f"MLP (L{layer}) — non-linear transformation for output attribution."
    elif component == "resid":
        return f"Residual stream (L{layer}) — accumulating upstream information."
    return ""


def _get_connections(node_id: str, topology_data: Dict) -> List[Dict]:
    """Get all connections (incoming and outgoing) for a node."""
    connections = []
    for edge in topology_data.get("edges", []):
        if edge["source"] == node_id:
            connections.append({
                "direction": "\u2192",
                "node": edge["target"].split("__")[0].replace("blocks.", "L").replace(".hook_", " "),
                "weight": edge["weight"],
            })
        elif edge["target"] == node_id:
            connections.append({
                "direction": "\u2190",
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
