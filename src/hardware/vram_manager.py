"""
VRAM Allocation Manager for MI Toolkit.

This module implements the hardware-aware VRAM allocation strategy defined
in the implementation plan, automatically selecting the optimal quantization
precision based on model size and available GPU memory.

VRAM Allocation Matrix (RTX 4090, 24GB):
    ┌─────────────────────┬──────────────┬─────────────────────────────────┐
    │ Model Size          │ Precision    │ Strategy                        │
    ├─────────────────────┼──────────────┼─────────────────────────────────┤
    │ 1.5B parameters     │ FP16         │ Full model in VRAM              │
    │ 2–3B parameters     │ FP16         │ Full model in VRAM              │
    │ 7–8B parameters     │ INT8 / FP8   │ bitsandbytes 8-bit quantization │
    │ 9B+ parameters      │ INT8 / FP8   │ 8-bit + KV-cache paging         │
    └─────────────────────┴──────────────┴─────────────────────────────────┘

L2 Cache Optimization:
    RTX 4090 L2 cache: 72MB
    Tile size for attention computation: 72MB / (d_model * 2 bytes)
    Batch size selection: maximize L2 utilization without overflow.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class VRAMProfile:
    """
    Hardware-aware VRAM allocation profile for a specific model.

    Attributes:
        model_name: Model identifier string.
        n_params_billions: Estimated parameter count in billions.
        precision: Selected precision ("fp16", "int8", "fp8", "int4").
        use_paging: Whether to enable KV-cache paging for large models.
        recommended_batch_size: Optimal batch size for L2 cache utilization.
        tile_size: Attention computation tile size (tokens per tile).
        estimated_vram_gb: Estimated VRAM usage in GB.
    """
    model_name: str
    n_params_billions: float
    precision: str
    use_paging: bool
    recommended_batch_size: int
    tile_size: int
    estimated_vram_gb: float


# L2 cache size for RTX 4090 in bytes
RTX4090_L2_CACHE_BYTES = 72 * 1024 * 1024  # 72 MB


class VRAMManager:
    """
    Manages VRAM allocation and quantization strategy selection for MI Toolkit.

    Automatically detects available GPU memory and selects the optimal
    precision and batching strategy for the target model.

    Args:
        target_gpu_id: CUDA device index to target. Defaults to 0.
        l2_cache_bytes: L2 cache size in bytes. Defaults to RTX 4090 (72MB).
        safety_margin_gb: Reserved VRAM buffer in GB to avoid OOM errors.
    """

    def __init__(
        self,
        target_gpu_id: int = 0,
        l2_cache_bytes: int = RTX4090_L2_CACHE_BYTES,
        safety_margin_gb: float = 2.0,
    ):
        self.target_gpu_id = target_gpu_id
        self.l2_cache_bytes = l2_cache_bytes
        self.safety_margin_gb = safety_margin_gb
        self._device = f"cuda:{target_gpu_id}" if torch.cuda.is_available() else "cpu"

    def get_available_vram_gb(self) -> float:
        """Return available VRAM in GB on the target GPU."""
        if not torch.cuda.is_available():
            return 0.0
        total = torch.cuda.get_device_properties(self.target_gpu_id).total_memory
        reserved = torch.cuda.memory_reserved(self.target_gpu_id)
        available = total - reserved
        return available / (1024 ** 3)

    def select_profile(
        self,
        model_name: str,
        n_params_billions: Optional[float] = None,
        d_model: Optional[int] = None,
    ) -> VRAMProfile:
        """
        Select the optimal VRAM allocation profile for a given model.

        Args:
            model_name: Model name string (used to infer parameter count if
                        n_params_billions is not provided).
            n_params_billions: Explicit parameter count in billions. If None,
                               inferred from model_name.
            d_model: Model hidden dimension (used for tile size calculation).

        Returns:
            VRAMProfile with recommended precision and batching strategy.
        """
        if n_params_billions is None:
            n_params_billions = self._infer_param_count(model_name)

        available_vram = self.get_available_vram_gb()
        usable_vram = max(0.0, available_vram - self.safety_margin_gb)

        # Select precision based on model size and available VRAM
        precision, use_paging = self._select_precision(n_params_billions, usable_vram)

        # Estimate VRAM usage
        estimated_vram = self._estimate_vram(n_params_billions, precision)

        # Compute tile size for L2 cache optimization
        tile_size = self._compute_tile_size(d_model or 2048, precision)

        # Recommend batch size
        batch_size = self._recommend_batch_size(
            n_params_billions, precision, usable_vram
        )

        return VRAMProfile(
            model_name=model_name,
            n_params_billions=n_params_billions,
            precision=precision,
            use_paging=use_paging,
            recommended_batch_size=batch_size,
            tile_size=tile_size,
            estimated_vram_gb=estimated_vram,
        )

    def load_model_with_profile(
        self,
        model_name: str,
        profile: Optional[VRAMProfile] = None,
        **kwargs,
    ) -> Any:
        """
        Load a TransformerLens model with the recommended VRAM profile.

        Args:
            model_name: Hugging Face model identifier.
            profile: VRAMProfile to apply. If None, auto-selects.
            **kwargs: Additional arguments passed to HookedTransformer.from_pretrained.

        Returns:
            Loaded HookedTransformer model.
        """
        try:
            from transformer_lens import HookedTransformer
        except ImportError:
            raise ImportError(
                "transformer_lens is required. "
                "Install with: pip install transformer_lens"
            )

        if profile is None:
            profile = self.select_profile(model_name)

        load_kwargs = dict(kwargs)

        # Apply quantization settings
        if profile.precision == "fp16":
            load_kwargs["dtype"] = torch.float16
        elif profile.precision in ("int8", "fp8"):
            load_kwargs["dtype"] = torch.float16  # Load in FP16, quantize post-load
        elif profile.precision == "int4":
            load_kwargs["dtype"] = torch.float16

        load_kwargs.setdefault("device", self._device)

        model = HookedTransformer.from_pretrained(model_name, **load_kwargs)

        # Apply post-load quantization for large models
        if profile.precision in ("int8",) and profile.n_params_billions >= 7.0:
            model = self._apply_int8_quantization(model)

        return model

    def clear_vram(self, model: Optional[Any] = None) -> None:
        """
        Clear VRAM by removing model and running garbage collection.

        Implements the reset_hooks_end / clear_contexts cleanup pattern
        from the implementation plan.

        Args:
            model: Optional model to delete before clearing VRAM.
        """
        if model is not None:
            # Remove all hooks before deletion
            if hasattr(model, "reset_hooks"):
                model.reset_hooks(including_permanent=True)
            del model

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_vram_stats(self) -> Dict[str, float]:
        """Return current VRAM usage statistics in GB."""
        if not torch.cuda.is_available():
            return {"available": 0.0, "allocated": 0.0, "reserved": 0.0, "total": 0.0}

        props = torch.cuda.get_device_properties(self.target_gpu_id)
        total = props.total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(self.target_gpu_id) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(self.target_gpu_id) / (1024 ** 3)
        available = total - reserved

        return {
            "total_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "available_gb": round(available, 2),
            "utilization_pct": round(100 * reserved / total, 1) if total > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_param_count(model_name: str) -> float:
        """Infer approximate parameter count in billions from model name."""
        name_lower = model_name.lower()
        # Check for explicit size indicators
        for indicator, size in [
            ("70b", 70.0), ("34b", 34.0), ("13b", 13.0), ("9b", 9.0),
            ("8b", 8.0), ("7b", 7.0), ("3b", 3.0), ("2b", 2.0),
            ("1.5b", 1.5), ("1b", 1.0), ("410m", 0.41), ("160m", 0.16),
        ]:
            if indicator in name_lower:
                return size
        # Fallback: assume 7B for unknown models
        return 7.0

    @staticmethod
    def _select_precision(
        n_params_billions: float, usable_vram_gb: float
    ) -> Tuple[str, bool]:
        """
        Select quantization precision and paging strategy.

        Returns:
            Tuple of (precision_string, use_paging_bool).
        """
        # FP16 memory estimate: ~2 bytes/param
        fp16_vram = n_params_billions * 2.0  # GB

        if fp16_vram <= usable_vram_gb * 0.8:
            return "fp16", False

        # INT8 memory estimate: ~1 byte/param
        int8_vram = n_params_billions * 1.0  # GB

        if int8_vram <= usable_vram_gb * 0.8:
            use_paging = n_params_billions >= 9.0
            return "int8", use_paging

        # INT4 fallback for very large models
        return "int4", True

    @staticmethod
    def _estimate_vram(n_params_billions: float, precision: str) -> float:
        """Estimate VRAM usage in GB for the given model and precision."""
        bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "fp8": 1, "int4": 0.5}
        bpp = bytes_per_param.get(precision, 2)
        # Add ~20% overhead for activations and optimizer states
        return round(n_params_billions * bpp * 1.2, 2)

    def _compute_tile_size(self, d_model: int, precision: str) -> int:
        """
        Compute optimal attention tile size to maximize L2 cache utilization.

        Tile size = L2_cache_bytes / (d_model * bytes_per_element * 2)
        The factor of 2 accounts for storing both Q and K tiles simultaneously.
        """
        bytes_per_elem = 2 if precision in ("fp16", "bf16", "int8", "fp8") else 4
        tile_size = self.l2_cache_bytes // (d_model * bytes_per_elem * 2)
        # Clamp to reasonable range
        return max(16, min(512, tile_size))

    @staticmethod
    def _recommend_batch_size(
        n_params_billions: float, precision: str, usable_vram_gb: float
    ) -> int:
        """Recommend a batch size based on available VRAM after model loading."""
        model_vram = n_params_billions * (2 if precision == "fp16" else 1)
        remaining_vram = usable_vram_gb - model_vram
        if remaining_vram <= 1.0:
            return 1
        elif remaining_vram <= 4.0:
            return 2
        elif remaining_vram <= 8.0:
            return 4
        return 8

    @staticmethod
    def _apply_int8_quantization(model: Any) -> Any:
        """
        Apply bitsandbytes INT8 quantization to linear layers.

        This is a best-effort quantization that replaces nn.Linear layers
        with 8-bit quantized equivalents when bitsandbytes is available.
        """
        try:
            import bitsandbytes as bnb
            import torch.nn as nn

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and module.weight.shape[0] >= 64:
                    # Replace with 8-bit linear
                    parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                    parent = model if not parent_name else dict(model.named_modules())[parent_name]
                    int8_layer = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False,
                    )
                    setattr(parent, child_name, int8_layer)
        except ImportError:
            pass  # bitsandbytes not available; skip quantization
        return model
