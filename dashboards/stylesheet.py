"""
Cytoscape Stylesheet for MI Toolkit Dashboard.

This module defines the visual style mappings for the interactive DAG
visualization. Causal attribution scores from ReIP are mapped to:
    - Edge width (weight -> line thickness)
    - Edge color intensity (weight -> color gradient)
    - Node size (feature importance -> node radius)
    - Node color (component type -> color category)

Color scheme:
    - MLP nodes:      #4A90D9 (blue)
    - Attention nodes: #E67E22 (orange)
    - Residual nodes:  #27AE60 (green)
    - Embedding nodes: #8E44AD (purple)
    - Selected nodes:  #E74C3C (red highlight)
"""

from __future__ import annotations

from typing import List, Dict


def build_stylesheet(
    min_edge_width: float = 1.0,
    max_edge_width: float = 12.0,
    min_node_size: float = 20.0,
    max_node_size: float = 60.0,
) -> List[Dict]:
    """
    Build the Cytoscape stylesheet for the circuit topology visualization.

    The stylesheet uses CSS-like selectors to apply visual properties to
    nodes and edges based on their data attributes.

    Args:
        min_edge_width: Minimum edge line width in pixels.
        max_edge_width: Maximum edge line width in pixels.
        min_node_size: Minimum node diameter in pixels.
        max_node_size: Maximum node diameter in pixels.

    Returns:
        List of Cytoscape stylesheet dictionaries.
    """
    stylesheet = [
        # ---------------------------------------------------------------
        # Default node style
        # ---------------------------------------------------------------
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "9px",
                "font-family": "monospace",
                "color": "#FFFFFF",
                "text-outline-color": "#2C3E50",
                "text-outline-width": "1px",
                "text-wrap": "wrap",
                "text-max-width": "80px",
                "width": "mapData(score, 0, 1, 20, 55)",
                "height": "mapData(score, 0, 1, 20, 55)",
                "background-color": "#4A90D9",
                "border-width": "1.5px",
                "border-color": "#2C3E50",
                "opacity": 0.9,
            },
        },
        # ---------------------------------------------------------------
        # Component-specific node colors
        # ---------------------------------------------------------------
        {
            "selector": "node[component = 'mlp']",
            "style": {
                "background-color": "#4A90D9",  # Blue
                "border-color": "#1A5276",
            },
        },
        {
            "selector": "node[component = 'attn']",
            "style": {
                "background-color": "#E67E22",  # Orange
                "border-color": "#784212",
            },
        },
        {
            "selector": "node[component = 'resid']",
            "style": {
                "background-color": "#27AE60",  # Green
                "border-color": "#145A32",
            },
        },
        {
            "selector": "node[component = 'embed']",
            "style": {
                "background-color": "#8E44AD",  # Purple
                "border-color": "#4A235A",
            },
        },
        {
            "selector": "node[component = 'ln']",
            "style": {
                "background-color": "#95A5A6",  # Gray
                "border-color": "#566573",
            },
        },
        # ---------------------------------------------------------------
        # High-relevance node highlight
        # ---------------------------------------------------------------
        {
            "selector": "node[score > 0.7]",
            "style": {
                "border-width": "3px",
                "border-color": "#F39C12",
                "opacity": 1.0,
            },
        },
        # ---------------------------------------------------------------
        # Selected node
        # ---------------------------------------------------------------
        {
            "selector": "node:selected",
            "style": {
                "background-color": "#E74C3C",
                "border-color": "#922B21",
                "border-width": "3px",
                "z-index": 9999,
            },
        },
        # ---------------------------------------------------------------
        # Hovered node
        # ---------------------------------------------------------------
        {
            "selector": "node:active",
            "style": {
                "overlay-color": "#F39C12",
                "overlay-padding": "5px",
                "overlay-opacity": 0.3,
            },
        },
        # ---------------------------------------------------------------
        # Default edge style
        # ---------------------------------------------------------------
        {
            "selector": "edge",
            "style": {
                "width": f"mapData(weight, 0, 1, {min_edge_width}, {max_edge_width})",
                "line-color": "mapData(weight, 0, 1, #BDC3C7, #E74C3C)",
                "target-arrow-color": "mapData(weight, 0, 1, #BDC3C7, #E74C3C)",
                "target-arrow-shape": "triangle",
                "arrow-scale": 0.8,
                "curve-style": "bezier",
                "opacity": "mapData(weight, 0, 1, 0.3, 0.9)",
            },
        },
        # ---------------------------------------------------------------
        # High-weight edge highlight
        # ---------------------------------------------------------------
        {
            "selector": "edge[weight > 0.7]",
            "style": {
                "line-color": "#C0392B",
                "target-arrow-color": "#C0392B",
                "opacity": 1.0,
                "z-index": 100,
            },
        },
        # ---------------------------------------------------------------
        # Selected edge
        # ---------------------------------------------------------------
        {
            "selector": "edge:selected",
            "style": {
                "line-color": "#F39C12",
                "target-arrow-color": "#F39C12",
                "width": max_edge_width,
                "opacity": 1.0,
            },
        },
    ]
    return stylesheet


def get_layout_config(layout_name: str = "dagre") -> Dict:
    """
    Return Cytoscape layout configuration for hierarchical DAG display.

    Args:
        layout_name: Layout algorithm name.
                     Options: "dagre", "breadthfirst", "cose", "grid"

    Returns:
        Layout configuration dictionary for Cytoscape.
    """
    layouts = {
        "dagre": {
            "name": "dagre",
            "rankDir": "TB",       # Top-to-bottom: input -> output layers
            "nodeSep": 50,
            "rankSep": 80,
            "edgeSep": 10,
            "animate": True,
            "animationDuration": 500,
        },
        "breadthfirst": {
            "name": "breadthfirst",
            "directed": True,
            "spacingFactor": 1.5,
            "animate": True,
            "animationDuration": 500,
        },
        "cose": {
            "name": "cose",
            "idealEdgeLength": 100,
            "nodeOverlap": 20,
            "refresh": 20,
            "fit": True,
            "padding": 30,
            "randomize": False,
            "componentSpacing": 100,
            "nodeRepulsion": 400000,
            "edgeElasticity": 100,
            "nestingFactor": 5,
            "gravity": 80,
            "numIter": 1000,
            "animate": True,
        },
        "grid": {
            "name": "grid",
            "fit": True,
            "padding": 30,
            "avoidOverlap": True,
        },
    }
    return layouts.get(layout_name, layouts["dagre"])
