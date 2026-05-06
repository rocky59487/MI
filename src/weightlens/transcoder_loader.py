"""
Transcoder Weight Loader for WeightLens.

Loads sparse Transcoder weight dictionaries from Hugging Face Hub or local
cache. Transcoders approximate the full MLP layer's input-to-output mapping
sparsely, allowing clean decoupling of input-dependent and input-invariant
feature connections.

Supported transcoder sources:
    - Hugging Face Hub repositories (e.g., "jacobdunefsky/transcoder_circuits")
    - Local .pt / .safetensors files
    - Pre-downloaded JSON cache files

Architecture:
    Transcoder for layer l:
        f_enc^(l,i): encoder vector  (d_model -> d_transcoder)
        f_dec^(l,i): decoder vector  (d_transcoder -> d_model)
        b_enc^(l):   encoder bias
        b_dec^(l):   decoder bias

    Input-invariant cross-layer contribution:
        c(l, i, l', i') = f_dec^(l',i') · f_enc^(l,i)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class TranscoderWeights:
    """
    Container for a single layer's Transcoder weight tensors.

    Attributes:
        layer_idx: The transformer layer index this transcoder corresponds to.
        W_enc: Encoder weight matrix, shape (d_model, d_transcoder).
        W_dec: Decoder weight matrix, shape (d_transcoder, d_model).
        b_enc: Encoder bias vector, shape (d_transcoder,). May be None.
        b_dec: Decoder bias vector, shape (d_model,). May be None.
        n_features: Number of transcoder features (d_transcoder dimension).
    """
    layer_idx: int
    W_enc: torch.Tensor
    W_dec: torch.Tensor
    b_enc: Optional[torch.Tensor]
    b_dec: Optional[torch.Tensor]

    @property
    def n_features(self) -> int:
        return self.W_enc.shape[1]

    @property
    def d_model(self) -> int:
        return self.W_enc.shape[0]

    def to(self, device: str) -> "TranscoderWeights":
        """Move all tensors to the specified device."""
        return TranscoderWeights(
            layer_idx=self.layer_idx,
            W_enc=self.W_enc.to(device),
            W_dec=self.W_dec.to(device),
            b_enc=self.b_enc.to(device) if self.b_enc is not None else None,
            b_dec=self.b_dec.to(device) if self.b_dec is not None else None,
        )


class TranscoderLoader:
    """
    Loads and manages Transcoder weight dictionaries for WeightLens analysis.

    Supports loading from:
        1. Hugging Face Hub (requires internet or pre-cached files)
        2. Local .pt / .safetensors checkpoint files
        3. Pre-structured directory with per-layer weight files

    For Air-Gapped (offline) deployments, pre-download weights and specify
    the local_cache_dir parameter.

    Example::

        loader = TranscoderLoader(
            model_name="gpt2",
            local_cache_dir="/data/transcoders/gpt2",
        )
        weights = loader.load_layer(layer_idx=6)
        print(f"Layer 6 transcoder: {weights.n_features} features")
    """

    # Known Hugging Face transcoder repositories per model
    HF_TRANSCODER_REPOS: Dict[str, str] = {
        "gpt2":              "jacobdunefsky/transcoder_circuits",
        "gpt2-small":        "jacobdunefsky/transcoder_circuits",
        "gemma-2-2b":        "google/gemma-scope-2b-pt-res",
        "llama-3.2-1b":      "meta-llama/Llama-3.2-1B",
    }

    def __init__(
        self,
        model_name: str,
        local_cache_dir: Optional[str] = None,
        device: str = "cpu",
        verbose: bool = False,
    ):
        """
        Args:
            model_name: Target model name (used to locate HF repository).
            local_cache_dir: Local directory for cached transcoder weights.
                             If None, uses ~/.cache/mi_toolkit/transcoders/.
            device: Device to load tensors onto ("cpu" recommended for storage).
            verbose: Print loading progress.
        """
        self.model_name = model_name.lower()
        self.device = device
        self.verbose = verbose

        if local_cache_dir is None:
            home = Path.home()
            self.cache_dir = home / ".cache" / "mi_toolkit" / "transcoders" / self.model_name
        else:
            self.cache_dir = Path(local_cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_layers: Dict[int, TranscoderWeights] = {}

    def load_layer(self, layer_idx: int) -> Optional[TranscoderWeights]:
        """
        Load Transcoder weights for a specific layer.

        First checks the in-memory cache, then the local file cache,
        and finally attempts to download from Hugging Face Hub.

        Args:
            layer_idx: Transformer layer index to load.

        Returns:
            TranscoderWeights instance, or None if not available.
        """
        if layer_idx in self._loaded_layers:
            return self._loaded_layers[layer_idx]

        # Try local cache first
        local_path = self.cache_dir / f"layer_{layer_idx:02d}.pt"
        if local_path.exists():
            weights = self._load_from_pt(local_path, layer_idx)
            if weights is not None:
                self._loaded_layers[layer_idx] = weights
                return weights

        # Attempt HF Hub download
        weights = self._download_from_hf(layer_idx)
        if weights is not None:
            # Save to local cache for future use
            self._save_to_pt(weights, local_path)
            self._loaded_layers[layer_idx] = weights
            return weights

        if self.verbose:
            print(f"[TranscoderLoader] No transcoder found for layer {layer_idx}. "
                  "Falling back to identity (no transcoder).")
        return None

    def load_all_layers(self, n_layers: int) -> Dict[int, TranscoderWeights]:
        """
        Load Transcoder weights for all layers of the model.

        Args:
            n_layers: Total number of transformer layers.

        Returns:
            Dictionary mapping layer index to TranscoderWeights.
        """
        for layer_idx in range(n_layers):
            self.load_layer(layer_idx)
        return dict(self._loaded_layers)

    def _load_from_pt(
        self, path: Path, layer_idx: int
    ) -> Optional[TranscoderWeights]:
        """Load TranscoderWeights from a .pt checkpoint file."""
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
            return self._parse_state_dict(state, layer_idx)
        except Exception as e:
            if self.verbose:
                print(f"[TranscoderLoader] Failed to load {path}: {e}")
            return None

    def _save_to_pt(self, weights: TranscoderWeights, path: Path) -> None:
        """Save TranscoderWeights to a .pt checkpoint file."""
        state = {
            "W_enc": weights.W_enc,
            "W_dec": weights.W_dec,
            "b_enc": weights.b_enc,
            "b_dec": weights.b_dec,
            "layer_idx": weights.layer_idx,
        }
        torch.save(state, path)
        if self.verbose:
            print(f"[TranscoderLoader] Saved layer {weights.layer_idx} to {path}")

    def _parse_state_dict(
        self, state: Dict, layer_idx: int
    ) -> Optional[TranscoderWeights]:
        """Parse a state dictionary into a TranscoderWeights instance."""
        try:
            # Support multiple key naming conventions
            # NOTE: Cannot use `or` with Tensors (triggers RuntimeError:
            # "Boolean value of Tensor with more than one element is ambiguous")
            W_enc = state["W_enc"] if "W_enc" in state else state.get("encoder.weight")
            W_dec = state["W_dec"] if "W_dec" in state else state.get("decoder.weight")

            if W_enc is None or W_dec is None:
                return None

            b_enc = state["b_enc"] if "b_enc" in state else state.get("encoder.bias")
            b_dec = state["b_dec"] if "b_dec" in state else state.get("decoder.bias")

            return TranscoderWeights(
                layer_idx=layer_idx,
                W_enc=W_enc.float(),
                W_dec=W_dec.float(),
                b_enc=b_enc,
                b_dec=b_dec,
            )
        except Exception as e:
            if self.verbose:
                print(f"[TranscoderLoader] Failed to parse state dict: {e}")
            return None

    def _download_from_hf(self, layer_idx: int) -> Optional[TranscoderWeights]:
        """
        Attempt to download Transcoder weights from Hugging Face Hub.

        Returns None gracefully if the Hub is unavailable (Air-Gapped mode)
        or if no transcoder exists for this model/layer combination.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return None

        repo_id = None
        for key, repo in self.HF_TRANSCODER_REPOS.items():
            if key in self.model_name:
                repo_id = repo
                break

        if repo_id is None:
            return None

        filename = f"layer_{layer_idx:02d}/transcoder.pt"
        try:
            local_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir / "hf_cache"),
            )
            return self._load_from_pt(Path(local_file), layer_idx)
        except Exception:
            return None

    def create_synthetic_transcoder(
        self,
        layer_idx: int,
        d_model: int,
        n_features: int,
        seed: int = 42,
    ) -> TranscoderWeights:
        """
        Create a synthetic random Transcoder for testing purposes.

        This allows the pipeline to run end-to-end even when no real
        transcoder weights are available for a given model.

        Args:
            layer_idx: Layer index to assign.
            d_model: Model hidden dimension.
            n_features: Number of transcoder features.
            seed: Random seed for reproducibility.

        Returns:
            TranscoderWeights with randomly initialized tensors.
        """
        torch.manual_seed(seed + layer_idx)
        W_enc = torch.randn(d_model, n_features) * 0.02
        W_dec = torch.randn(n_features, d_model) * 0.02
        return TranscoderWeights(
            layer_idx=layer_idx,
            W_enc=W_enc,
            W_dec=W_dec,
            b_enc=torch.zeros(n_features),
            b_dec=torch.zeros(d_model),
        )
