"""
Cytoscape Stylesheet for MI Circuit Explorer — Neural Pulse Flow Design.

This module defines the visual style mappings for the directional causal flow
visualization. The design follows a "neural pulse flow" aesthetic with a deep
space theme, but arranged as a left-to-right directional graph showing how
information flows through the model's layers.

Color scheme:
    - Input/Embed nodes:  #8E44AD (purple)
    - Attention nodes:    #E67E22 (orange)
    - MLP nodes:          #4A90D9 (blue)
    - Residual nodes:     #27AE60 (green)
    - Output nodes:       #E74C3C (red)
    - Dangerous nodes:    #FF0000 (bright red with pulse animation)
    - Edges:              Gradient from #4A90D9 to #E74C3C based on weight
    - Dangerous edges:    #FF0000 with pulse effect
"""

from __future__ import annotations

from typing import List, Dict


def build_stylesheet() -> List[Dict]:
    """
    Build the Cytoscape stylesheet for the directional causal flow visualization.

    Returns:
        List of Cytoscape stylesheet dictionaries.
    """
    stylesheet = [
        # ---------------------------------------------------------------
        # Default node style — rounded rectangle for flow nodes
        # ---------------------------------------------------------------
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "text-valign": "center",
                "text-halign": "right",
                "text-margin-x": "8px",
                "font-size": "11px",
                "font-family": "'JetBrains Mono', 'Fira Code', monospace",
                "color": "#E0E0E0",
                "text-outline-color": "#0D1B2A",
                "text-outline-width": "1.5px",
                "text-wrap": "wrap",
                "text-max-width": "120px",
                "width": "mapData(score, 0, 1, 28, 56)",
                "height": "mapData(score, 0, 1, 28, 56)",
                "background-color": "#4A90D9",
                "border-width": "2px",
                "border-color": "#1B2838",
                "shape": "ellipse",
                "opacity": 0,
                "transition-property": "opacity, background-color, width, height",
                "transition-duration": "0.6s",
                "transition-timing-function": "ease-out",
                "ghost": "yes",
                "ghost-offset-x": "2px",
                "ghost-offset-y": "2px",
                "ghost-opacity": 0.15,
            },
        },
        # ---------------------------------------------------------------
        # Visible node (after animation reveals it)
        # ---------------------------------------------------------------
        {
            "selector": "node.visible",
            "style": {
                "opacity": 1.0,
            },
        },
        # ---------------------------------------------------------------
        # Component-specific node colors
        # ---------------------------------------------------------------
        {
            "selector": "node[component = 'embed']",
            "style": {
                "background-color": "#8E44AD",
                "border-color": "#6C3483",
                "shape": "diamond",
            },
        },
        {
            "selector": "node[component = 'attn']",
            "style": {
                "background-color": "#E67E22",
                "border-color": "#A04000",
            },
        },
        {
            "selector": "node[component = 'mlp']",
            "style": {
                "background-color": "#4A90D9",
                "border-color": "#1A5276",
            },
        },
        {
            "selector": "node[component = 'resid']",
            "style": {
                "background-color": "#27AE60",
                "border-color": "#145A32",
            },
        },
        {
            "selector": "node[component = 'output']",
            "style": {
                "background-color": "#E74C3C",
                "border-color": "#922B21",
                "shape": "star",
            },
        },
        # ---------------------------------------------------------------
        # High-relevance node glow effect
        # ---------------------------------------------------------------
        {
            "selector": "node[score > 0.7]",
            "style": {
                "border-width": "3px",
                "border-color": "#F39C12",
                "shadow-blur": "12px",
                "shadow-color": "#F39C12",
                "shadow-opacity": 0.6,
            },
        },
        # ---------------------------------------------------------------
        # DANGEROUS NODE — Safety Mode highlighting
        # ---------------------------------------------------------------
        {
            "selector": "node.dangerous",
            "style": {
                "background-color": "#FF0000",
                "border-color": "#FF4444",
                "border-width": "4px",
                "shadow-blur": "25px",
                "shadow-color": "#FF0000",
                "shadow-opacity": 0.9,
                "shape": "ellipse",
                "width": "mapData(score, 0, 1, 36, 64)",
                "height": "mapData(score, 0, 1, 36, 64)",
                "text-outline-color": "#330000",
                "color": "#FFFFFF",
            },
        },
        {
            "selector": "node.dangerous.visible",
            "style": {
                "opacity": 1.0,
            },
        },
        # ---------------------------------------------------------------
        # Selected / clicked node
        # ---------------------------------------------------------------
        {
            "selector": "node:selected",
            "style": {
                "border-color": "#00FFFF",
                "border-width": "3px",
                "shadow-blur": "20px",
                "shadow-color": "#00FFFF",
                "shadow-opacity": 0.8,
                "z-index": 9999,
            },
        },
        # ---------------------------------------------------------------
        # Expanded node (clicked for details)
        # ---------------------------------------------------------------
        {
            "selector": "node.expanded",
            "style": {
                "border-color": "#00FFFF",
                "border-width": "3px",
                "background-opacity": 1.0,
            },
        },
        # ---------------------------------------------------------------
        # Child/detail nodes
        # ---------------------------------------------------------------
        {
            "selector": "node.child-node",
            "style": {
                "width": 20,
                "height": 20,
                "font-size": "9px",
                "border-width": "1px",
                "border-style": "dashed",
                "opacity": 0.85,
            },
        },
        # ---------------------------------------------------------------
        # Default edge style — directional arrows
        # ---------------------------------------------------------------
        {
            "selector": "edge",
            "style": {
                "width": "mapData(weight, 0, 1, 1.5, 6)",
                "line-color": "mapData(weight, 0, 1, #2C3E50, #E74C3C)",
                "target-arrow-color": "mapData(weight, 0, 1, #2C3E50, #E74C3C)",
                "target-arrow-shape": "triangle",
                "arrow-scale": 1.0,
                "curve-style": "bezier",
                "opacity": 0,
                "transition-property": "opacity, line-color, width",
                "transition-duration": "0.5s",
                "transition-timing-function": "ease-out",
            },
        },
        # ---------------------------------------------------------------
        # Visible edge (after animation)
        # ---------------------------------------------------------------
        {
            "selector": "edge.visible",
            "style": {
                "opacity": "mapData(weight, 0, 1, 0.4, 0.95)",
            },
        },
        # ---------------------------------------------------------------
        # High-weight edge — pulse glow
        # ---------------------------------------------------------------
        {
            "selector": "edge[weight > 0.7]",
            "style": {
                "line-color": "#E74C3C",
                "target-arrow-color": "#E74C3C",
                "z-index": 100,
            },
        },
        # ---------------------------------------------------------------
        # DANGEROUS EDGE — Safety Mode red pulse path
        # ---------------------------------------------------------------
        {
            "selector": "edge.dangerous-edge",
            "style": {
                "line-color": "#FF0000",
                "target-arrow-color": "#FF0000",
                "width": 5,
                "opacity": 1.0,
                "line-style": "solid",
                "z-index": 200,
            },
        },
        {
            "selector": "edge.dangerous-edge.visible",
            "style": {
                "opacity": 1.0,
            },
        },
        # ---------------------------------------------------------------
        # Selected edge
        # ---------------------------------------------------------------
        {
            "selector": "edge:selected",
            "style": {
                "line-color": "#00FFFF",
                "target-arrow-color": "#00FFFF",
                "width": 8,
                "opacity": 1.0,
            },
        },
        # ---------------------------------------------------------------
        # Highlighted path edges
        # ---------------------------------------------------------------
        {
            "selector": "edge.highlighted",
            "style": {
                "line-color": "#00FFFF",
                "target-arrow-color": "#00FFFF",
                "opacity": 1.0,
                "width": 4,
            },
        },
    ]
    return stylesheet


def get_layout_config(layout_name: str = "dagre-lr") -> Dict:
    """
    Return Cytoscape layout configuration for directional flow display.

    The primary layout is left-to-right (LR) dagre which shows the causal
    flow from input layers through intermediate processing to output.

    Args:
        layout_name: Layout algorithm name.

    Returns:
        Layout configuration dictionary for Cytoscape.
    """
    layouts = {
        "dagre-lr": {
            "name": "dagre",
            "rankDir": "LR",
            "nodeSep": 60,
            "rankSep": 120,
            "edgeSep": 20,
            "animate": True,
            "animationDuration": 800,
            "fit": True,
            "padding": 40,
        },
        "dagre-tb": {
            "name": "dagre",
            "rankDir": "TB",
            "nodeSep": 50,
            "rankSep": 100,
            "edgeSep": 15,
            "animate": True,
            "animationDuration": 800,
            "fit": True,
            "padding": 40,
        },
        "breadthfirst": {
            "name": "breadthfirst",
            "directed": True,
            "spacingFactor": 1.8,
            "animate": True,
            "animationDuration": 800,
            "fit": True,
            "padding": 40,
        },
    }
    return layouts.get(layout_name, layouts["dagre-lr"])
