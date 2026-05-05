"""
JSON-Based Precomputed Feature Cache Library for WeightLens.

This module implements a two-level caching system for WeightLens feature
semantics, enabling millisecond-speed lookup for previously analyzed features:

    Level 1 (in-memory): Python dict for the current session.
    Level 2 (on-disk):   JSON files organized by model name and layer index.

Cache structure on disk:
    ~/.cache/mi_toolkit/weightlens/{model_name}/
        layer_00.json
        layer_01.json
        ...
        metadata.json

Each layer JSON file maps feature indices to FeatureSemantics dicts:
    {
        "0": { "feature_idx": 0, "layer_idx": 0, "input_tokens": [...], ... },
        "1": { ... },
        ...
    }
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from .projection import FeatureSemantics


class FeatureCache:
    """
    Two-level (memory + disk) cache for WeightLens feature semantics.

    On cache hit: returns stored FeatureSemantics in O(1) time.
    On cache miss: caller computes semantics, then calls cache.store() to
                   persist the result for future sessions.

    Args:
        model_name: Model identifier used as the cache directory name.
        cache_root: Root directory for all cached data.
                    Defaults to ~/.cache/mi_toolkit/weightlens/.
        auto_save: If True, automatically write to disk on every store() call.
    """

    def __init__(
        self,
        model_name: str,
        cache_root: Optional[str] = None,
        auto_save: bool = True,
    ):
        self.model_name = model_name.lower().replace("/", "_").replace("\\", "_")
        self.auto_save = auto_save

        if cache_root is None:
            self.cache_root = Path.home() / ".cache" / "mi_toolkit" / "weightlens"
        else:
            self.cache_root = Path(cache_root)

        self.cache_dir = self.cache_root / self.model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache: {layer_idx: {feature_idx: FeatureSemantics}}
        self._memory: Dict[int, Dict[int, FeatureSemantics]] = {}
        self._dirty_layers: set = set()  # Layers with unsaved changes

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(
        self, layer_idx: int, feature_idx: int
    ) -> Optional[FeatureSemantics]:
        """
        Retrieve a cached FeatureSemantics instance.

        Args:
            layer_idx: Transformer layer index.
            feature_idx: Feature index within the transcoder.

        Returns:
            FeatureSemantics if found in cache, None otherwise (cache miss).
        """
        # Level 1: in-memory cache
        if layer_idx in self._memory:
            return self._memory[layer_idx].get(feature_idx)

        # Level 2: disk cache
        layer_data = self._load_layer_from_disk(layer_idx)
        if layer_data is not None:
            self._memory[layer_idx] = layer_data
            return layer_data.get(feature_idx)

        return None

    def get_layer(self, layer_idx: int) -> Dict[int, FeatureSemantics]:
        """
        Retrieve all cached features for a given layer.

        Returns:
            Dictionary mapping feature_idx to FeatureSemantics.
            Empty dict if layer is not cached.
        """
        if layer_idx not in self._memory:
            layer_data = self._load_layer_from_disk(layer_idx)
            if layer_data is not None:
                self._memory[layer_idx] = layer_data
            else:
                self._memory[layer_idx] = {}
        return self._memory[layer_idx]

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        semantics: FeatureSemantics,
        save_immediately: bool = False,
    ) -> None:
        """
        Store a FeatureSemantics instance in the cache.

        Args:
            semantics: The FeatureSemantics to cache.
            save_immediately: If True, write to disk immediately regardless
                              of the auto_save setting.
        """
        layer_idx = semantics.layer_idx
        feature_idx = semantics.feature_idx

        if layer_idx not in self._memory:
            self._memory[layer_idx] = {}

        self._memory[layer_idx][feature_idx] = semantics
        self._dirty_layers.add(layer_idx)

        if self.auto_save or save_immediately:
            self._save_layer_to_disk(layer_idx)
            self._dirty_layers.discard(layer_idx)

    def store_layer(
        self,
        layer_idx: int,
        features: List[FeatureSemantics],
    ) -> None:
        """
        Store all features for a layer in a single batch operation.

        Args:
            layer_idx: Transformer layer index.
            features: List of FeatureSemantics instances for this layer.
        """
        if layer_idx not in self._memory:
            self._memory[layer_idx] = {}

        for sem in features:
            self._memory[layer_idx][sem.feature_idx] = sem

        self._dirty_layers.add(layer_idx)
        if self.auto_save:
            self._save_layer_to_disk(layer_idx)
            self._dirty_layers.discard(layer_idx)

    def flush(self) -> None:
        """Write all dirty (unsaved) layers to disk."""
        for layer_idx in list(self._dirty_layers):
            self._save_layer_to_disk(layer_idx)
        self._dirty_layers.clear()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def is_cached(self, layer_idx: int, feature_idx: int) -> bool:
        """Check if a specific feature is present in the cache."""
        return self.get(layer_idx, feature_idx) is not None

    def is_layer_cached(self, layer_idx: int) -> bool:
        """Check if an entire layer's features are cached on disk."""
        return self._layer_cache_path(layer_idx).exists()

    def clear_memory(self) -> None:
        """Clear the in-memory cache (disk cache is preserved)."""
        self.flush()
        self._memory.clear()

    def clear_all(self) -> None:
        """Clear both in-memory and on-disk caches."""
        self._memory.clear()
        self._dirty_layers.clear()
        for json_file in self.cache_dir.glob("layer_*.json"):
            json_file.unlink()

    def stats(self) -> Dict:
        """Return cache statistics."""
        disk_layers = list(self.cache_dir.glob("layer_*.json"))
        total_features = sum(
            len(v) for v in self._memory.values()
        )
        return {
            "model_name": self.model_name,
            "cache_dir": str(self.cache_dir),
            "layers_in_memory": len(self._memory),
            "features_in_memory": total_features,
            "layers_on_disk": len(disk_layers),
            "dirty_layers": len(self._dirty_layers),
        }

    # ------------------------------------------------------------------
    # Private I/O helpers
    # ------------------------------------------------------------------

    def _layer_cache_path(self, layer_idx: int) -> Path:
        return self.cache_dir / f"layer_{layer_idx:02d}.json"

    def _load_layer_from_disk(
        self, layer_idx: int
    ) -> Optional[Dict[int, FeatureSemantics]]:
        """Load a layer's feature cache from disk JSON file."""
        path = self._layer_cache_path(layer_idx)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            result = {}
            for feat_idx_str, feat_dict in raw_data.items():
                feat_idx = int(feat_idx_str)
                sem = FeatureSemantics(
                    feature_idx=feat_dict.get("feature_idx", feat_idx),
                    layer_idx=feat_dict.get("layer_idx", layer_idx),
                    input_tokens=feat_dict.get("input_tokens", []),
                    input_zscores=feat_dict.get("input_zscores", []),
                    output_tokens_promoted=feat_dict.get("output_tokens_promoted", []),
                    output_tokens_suppressed=feat_dict.get("output_tokens_suppressed", []),
                    output_zscores=feat_dict.get("output_zscores", []),
                    raw_label=feat_dict.get("raw_label", ""),
                    connected_features=feat_dict.get("connected_features", []),
                )
                result[feat_idx] = sem
            return result
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _save_layer_to_disk(self, layer_idx: int) -> None:
        """Write a layer's feature cache to disk as a JSON file."""
        if layer_idx not in self._memory:
            return
        path = self._layer_cache_path(layer_idx)
        data = {
            str(feat_idx): sem.to_dict()
            for feat_idx, sem in self._memory[layer_idx].items()
        }
        # Atomic write: write to temp file then rename
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)
