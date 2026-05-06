"""
MI Circuit Explorer — Directional Causal Flow Dashboard.

Launch the interactive circuit visualization dashboard with:
    python dashboards/app.py

The dashboard visualizes how a language model "thinks" by showing the
directional causal flow from input through attention heads and MLPs to
the final output prediction. Nodes appear sequentially like a typewriter
effect, revealing the model's reasoning chain step by step.

Features:
    - Directional flow graph (left-to-right or top-to-bottom)
    - Top-N filtering (only show most important nodes by ReIP score)
    - Sequential reveal animation (nodes appear one-by-one)
    - Click-to-expand node details with causal interpretation
    - Deep space + neural pulse visual theme
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on the Python path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import dash
    import dash_cytoscape as cyto
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    print(
        "[MI Circuit Explorer] Dash or dash-cytoscape not installed.\n"
        "Install with: pip install dash dash-cytoscape\n"
    )

from dashboards.layout import build_layout
from dashboards.callbacks import register_callbacks


def create_app(debug: bool = False) -> "dash.Dash":
    """
    Create and configure the MI Circuit Explorer Dash application.

    Args:
        debug: Enable Dash debug mode (hot reload, error overlay).

    Returns:
        Configured Dash application instance.
    """
    if not HAS_DASH:
        raise ImportError(
            "Dash and dash-cytoscape are required. "
            "Install with: pip install dash dash-cytoscape"
        )

    # Register Cytoscape extra layouts (dagre requires cytoscape-dagre)
    cyto.load_extra_layouts()

    app = dash.Dash(
        __name__,
        title="MI Circuit Explorer — Causal Flow Visualization",
        suppress_callback_exceptions=True,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
    )

    app.layout = build_layout()
    register_callbacks(app)

    return app


def main():
    """Launch the dashboard server."""
    debug = os.environ.get("MI_DEBUG", "0") == "1"
    port = int(os.environ.get("MI_PORT", "8050"))
    host = os.environ.get("MI_HOST", "0.0.0.0")

    print(f"[MI Circuit Explorer] Starting server at http://{host}:{port}")
    print(f"[MI Circuit Explorer] Debug mode: {debug}")
    print(f"[MI Circuit Explorer] Directional causal flow visualization")

    app = create_app(debug=debug)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
