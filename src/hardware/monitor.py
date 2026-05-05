"""
Asynchronous Background Hardware Monitor for MI Toolkit.

This module provides a non-blocking background thread that continuously
monitors GPU VRAM usage, temperature, and utilization, exposing metrics
via a simple REST API endpoint (FastAPI) and an in-process callback system.

The monitor runs in a daemon thread and does not block the main analysis
pipeline. It is designed for long-running analysis sessions where VRAM
pressure needs to be tracked in real time.

Usage::

    from src.hardware.monitor import HardwareMonitor

    monitor = HardwareMonitor(poll_interval=2.0)
    monitor.start()

    # ... run analysis ...

    stats = monitor.get_latest_stats()
    print(stats)

    monitor.stop()
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

import torch


@dataclass
class HardwareSnapshot:
    """
    A single point-in-time hardware metrics snapshot.

    Attributes:
        timestamp: Unix timestamp of the snapshot.
        gpu_id: CUDA device index.
        vram_allocated_gb: Currently allocated VRAM in GB.
        vram_reserved_gb: Currently reserved VRAM in GB.
        vram_total_gb: Total VRAM capacity in GB.
        gpu_utilization_pct: GPU compute utilization percentage (0-100).
        gpu_temperature_c: GPU temperature in Celsius (if available).
        cpu_ram_used_gb: System RAM used in GB (if psutil available).
    """
    timestamp: float
    gpu_id: int
    vram_allocated_gb: float
    vram_reserved_gb: float
    vram_total_gb: float
    gpu_utilization_pct: float = 0.0
    gpu_temperature_c: float = 0.0
    cpu_ram_used_gb: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "gpu_id": self.gpu_id,
            "vram_allocated_gb": self.vram_allocated_gb,
            "vram_reserved_gb": self.vram_reserved_gb,
            "vram_total_gb": self.vram_total_gb,
            "vram_utilization_pct": round(
                100 * self.vram_reserved_gb / self.vram_total_gb, 1
            ) if self.vram_total_gb > 0 else 0.0,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "gpu_temperature_c": self.gpu_temperature_c,
            "cpu_ram_used_gb": self.cpu_ram_used_gb,
        }


class HardwareMonitor:
    """
    Non-blocking background hardware monitor for GPU and system resources.

    Runs a daemon thread that polls hardware metrics at a configurable
    interval and stores a rolling history of snapshots. Supports callback
    registration for threshold-based alerts.

    Args:
        gpu_id: CUDA device index to monitor. Defaults to 0.
        poll_interval: Seconds between metric polls. Defaults to 2.0.
        history_size: Number of snapshots to retain in rolling history.
        vram_alert_threshold_gb: Trigger alert callbacks when available VRAM
                                  drops below this value.
    """

    def __init__(
        self,
        gpu_id: int = 0,
        poll_interval: float = 2.0,
        history_size: int = 300,
        vram_alert_threshold_gb: float = 2.0,
    ):
        self.gpu_id = gpu_id
        self.poll_interval = poll_interval
        self.vram_alert_threshold_gb = vram_alert_threshold_gb

        self._history: Deque[HardwareSnapshot] = deque(maxlen=history_size)
        self._callbacks: List[Callable[[HardwareSnapshot], None]] = []
        self._alert_callbacks: List[Callable[[HardwareSnapshot], None]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Check for optional dependencies
        self._has_pynvml = self._check_pynvml()
        self._has_psutil = self._check_psutil()

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="MI-HardwareMonitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background monitoring thread gracefully."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_latest_stats(self) -> Optional[Dict]:
        """Return the most recent hardware snapshot as a dictionary."""
        with self._lock:
            if not self._history:
                return None
            return self._history[-1].to_dict()

    def get_history(self, n_last: int = 60) -> List[Dict]:
        """Return the last n_last snapshots as a list of dictionaries."""
        with self._lock:
            snapshots = list(self._history)[-n_last:]
        return [s.to_dict() for s in snapshots]

    def register_callback(
        self, callback: Callable[[HardwareSnapshot], None]
    ) -> None:
        """Register a callback invoked on every metric poll."""
        self._callbacks.append(callback)

    def register_alert_callback(
        self, callback: Callable[[HardwareSnapshot], None]
    ) -> None:
        """Register a callback invoked when VRAM drops below alert threshold."""
        self._alert_callbacks.append(callback)

    def take_snapshot(self) -> HardwareSnapshot:
        """Take an immediate hardware snapshot (blocking)."""
        return self._collect_metrics()

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Main monitoring loop running in the background thread."""
        while not self._stop_event.is_set():
            try:
                snapshot = self._collect_metrics()
                with self._lock:
                    self._history.append(snapshot)

                # Invoke registered callbacks
                for cb in self._callbacks:
                    try:
                        cb(snapshot)
                    except Exception:
                        pass

                # Check VRAM alert threshold
                available = snapshot.vram_total_gb - snapshot.vram_reserved_gb
                if available < self.vram_alert_threshold_gb:
                    for cb in self._alert_callbacks:
                        try:
                            cb(snapshot)
                        except Exception:
                            pass

            except Exception:
                pass

            self._stop_event.wait(timeout=self.poll_interval)

    def _collect_metrics(self) -> HardwareSnapshot:
        """Collect current hardware metrics."""
        timestamp = time.time()
        vram_allocated = 0.0
        vram_reserved = 0.0
        vram_total = 0.0
        gpu_util = 0.0
        gpu_temp = 0.0
        cpu_ram = 0.0

        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(self.gpu_id)
                vram_total = props.total_memory / (1024 ** 3)
                vram_allocated = torch.cuda.memory_allocated(self.gpu_id) / (1024 ** 3)
                vram_reserved = torch.cuda.memory_reserved(self.gpu_id) / (1024 ** 3)
            except Exception:
                pass

        # Try pynvml for GPU utilization and temperature
        if self._has_pynvml:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = float(util.gpu)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                gpu_temp = float(temp)
            except Exception:
                pass

        # Try psutil for CPU RAM
        if self._has_psutil:
            try:
                import psutil
                vm = psutil.virtual_memory()
                cpu_ram = vm.used / (1024 ** 3)
            except Exception:
                pass

        return HardwareSnapshot(
            timestamp=timestamp,
            gpu_id=self.gpu_id,
            vram_allocated_gb=round(vram_allocated, 3),
            vram_reserved_gb=round(vram_reserved, 3),
            vram_total_gb=round(vram_total, 3),
            gpu_utilization_pct=gpu_util,
            gpu_temperature_c=gpu_temp,
            cpu_ram_used_gb=round(cpu_ram, 3),
        )

    @staticmethod
    def _check_pynvml() -> bool:
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except Exception:
            return False

    @staticmethod
    def _check_psutil() -> bool:
        try:
            import psutil
            return True
        except Exception:
            return False


def create_monitor_api(monitor: HardwareMonitor, port: int = 8051):
    """
    Create a lightweight FastAPI server exposing hardware metrics via REST.

    Endpoints:
        GET /stats         — Latest hardware snapshot
        GET /history?n=60  — Last n snapshots
        GET /health        — Service health check

    Args:
        monitor: Running HardwareMonitor instance.
        port: Port to bind the API server on.

    Returns:
        FastAPI app instance (not started; call uvicorn.run() separately).
    """
    try:
        from fastapi import FastAPI
        import uvicorn
    except ImportError:
        raise ImportError(
            "FastAPI and uvicorn are required for the monitor API. "
            "Install with: pip install fastapi uvicorn"
        )

    api = FastAPI(title="MI Toolkit Hardware Monitor API")

    @api.get("/stats")
    def get_stats():
        return monitor.get_latest_stats() or {"error": "No data yet"}

    @api.get("/history")
    def get_history(n: int = 60):
        return monitor.get_history(n_last=n)

    @api.get("/health")
    def health():
        return {"status": "ok", "monitoring": monitor._thread is not None}

    return api
