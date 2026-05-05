"""
MI Toolkit Dashboard Application Entry Point.

Launch the interactive circuit visualization dashboard with:
    python dashboards/app.py

Or from the project root:
    python -m dashboards.app

The dashboard provides:
    - Interactive DAG visualization of ReIP causal topology graphs
    - Node hover/click for WeightLens semantic labels and Z-scores
    - Edge thickness/color encoding for attribution scores
    - Layout switching (dagre, breadthfirst, cose)
    - Real-time threshold filtering
    - In-browser ReIP analysis trigger
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
        "[MI Dashboard] Dash or dash-cytoscape not installed.\n"
        "Install with: pip install dash dash-cytoscape\n"
    )

from dashboards.layout import build_layout
from dashboards.callbacks import register_callbacks


def create_app(debug: bool = False) -> "dash.Dash":
    """
    Create and configure the MI Toolkit Dash application.

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
        title="MI Toolkit — Mechanistic Interpretability Dashboard",
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

    print(f"[MI Dashboard] Starting server at http://{host}:{port}")
    print(f"[MI Dashboard] Debug mode: {debug}")

    app = create_app(debug=debug)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
