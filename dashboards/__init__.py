"""
MI Toolkit Dashboard Package.

Provides the interactive Dash-based circuit visualization dashboard
with real backend integration and Agent Safety Mode.

This is a PRODUCTION research tool. There is no demo mode or mock data.
All analysis requires real model inference via the MI backend pipeline.
"""

from .app import create_app, main
from .backend import run_analysis, get_backend_status

__all__ = ["create_app", "main", "run_analysis", "get_backend_status"]
