"""
MI Toolkit Hardware Optimization Module.

Provides VRAM allocation management, quantization strategy selection,
and asynchronous background hardware monitoring.
"""

from .vram_manager import VRAMManager, VRAMProfile
from .monitor import HardwareMonitor, HardwareSnapshot

__all__ = [
    "VRAMManager",
    "VRAMProfile",
    "HardwareMonitor",
    "HardwareSnapshot",
]
