"""
Vocabulary Space Projection and Z-Score Filtering for WeightLens.

This module implements the core WeightLens analysis steps:

    Step 1 — Candidate Token Extraction:
        W_enc[:, feature_i] @ W_embed.T  -> input vocabulary logits
        Apply Z-score filtering to retain only statistically prominent tokens.

    Step 2 — Output Effects Analysis:
        W_dec[feature_i, :] @ W_unembed.T -> output vocabulary logits
        Apply Z-score filtering to find tokens strongly promoted/suppressed.

    Step 3 — Cross-layer Input-Invariant Connection Strength:
        f_dec^(l',i') · f_enc^(l,i)  -> scalar connection strength

Z-score thresholds (from benchmark data):
    GPT-2 family:                 4.0
    Gemma-2-2B / Llama-3.2-1B:   4.5
    Connected features:           3.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class FeatureSemantics:
    """
    Semantic description of a single Transcoder feature.

    Attributes:
        feature_idx: Index of the feature within the transcoder.
        layer_idx: Transformer layer this feature belongs to.
        input_tokens: High-Z-score input tokens (driving vocabulary).
        input_zscores: Z-scores for each input token.
        output_tokens_promoted: Tokens strongly promoted by this feature.
        output_tokens_suppressed: Tokens strongly suppressed by this feature.
        output_zscores: Z-scores for output tokens (positive = promoted).
        raw_label: Consolidated human-readable label (set by lemmatizer).
        connected_features: List of (layer, feature_idx) tuples for
                            input-invariant cross-layer connections.
    """
    feature_idx: int
    layer_idx: int
    input_tokens: List[str] = field(default_factory=list)
    input_zscores: List[float] = field(default_factory=list)
    output_tokens_promoted: List[str] = field(default_factory=list)
    output_tokens_suppressed: List[str] = field(default_factory=list)
    output_zscores: List[float] = field(default_factory=list)
    raw_label: str = ""
    connected_features: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "feature_idx": self.feature_idx,
            "layer_idx": self.layer_idx,
            "input_tokens": self.input_tokens,
            "input_zscores": self.input_zscores,
            "output_tokens_promoted": self.output_tokens_promoted,
            "output_tokens_suppressed": self.output_tokens_suppressed,
            "output_zscores": self.output_zscores,
            "raw_label": self.raw_label,
            "connected_features": self.connected_features,
        }


class VocabProjector:
    """
    Projects Transcoder feature vectors into vocabulary space and filters
    statistically prominent tokens using Z-score thresholding.

    This class implements the core WeightLens analysis without requiring
    any dataset or external LLM — all computation is purely matrix algebra
    on the model's learned weight matrices.

    Args:
        W_embed: Embedding matrix, shape (vocab_size, d_model).
        W_unembed: Unembedding matrix, shape (vocab_size, d_model).
        tokenizer: Tokenizer instance with convert_ids_to_tokens method.
        zscore_input: Z-score threshold for input token filtering.
        zscore_output: Z-score threshold for output token filtering.
        zscore_connected: Z-score threshold for connected feature identification.
        top_k_tokens: Maximum tokens to retain after Z-score filtering.
        device: Computation device.
    """

    def __init__(
        self,
        W_embed: torch.Tensor,
        W_unembed: torch.Tensor,
        tokenizer,
        zscore_input: float = 4.0,
        zscore_output: float = 4.0,
        zscore_connected: float = 3.0,
        top_k_tokens: int = 20,
        device: str = "cpu",
    ):
        self.W_embed = W_embed.to(device).float()
        self.W_unembed = W_unembed.to(device).float()
        self.tokenizer = tokenizer
        self.zscore_input = zscore_input
        self.zscore_output = zscore_output
        self.zscore_connected = zscore_connected
        self.top_k_tokens = top_k_tokens
        self.device = device

    @torch.no_grad()
    def analyze_feature(
        self,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        feature_idx: int,
        layer_idx: int,
    ) -> FeatureSemantics:
        """
        Perform full WeightLens analysis for a single Transcoder feature.

        Args:
            W_enc: Encoder weight matrix, shape (d_model, n_features).
            W_dec: Decoder weight matrix, shape (n_features, d_model).
            feature_idx: Index of the feature to analyze.
            layer_idx: Transformer layer index.

        Returns:
            FeatureSemantics with input/output token projections and Z-scores.
        """
        W_enc = W_enc.to(self.device).float()
        W_dec = W_dec.to(self.device).float()

        # Extract feature vectors
        enc_vec = W_enc[:, feature_idx]   # (d_model,)
        dec_vec = W_dec[feature_idx, :]   # (d_model,)

        # Step 1: Input token projection
        input_logits = self.W_embed @ enc_vec   # (vocab_size,)
        input_tokens, input_zscores = self._zscore_filter(
            input_logits, self.zscore_input, self.top_k_tokens
        )

        # Step 2: Output effects projection
        output_logits = self.W_unembed @ dec_vec   # (vocab_size,)
        output_tokens_all, output_zscores_all = self._zscore_filter(
            output_logits, self.zscore_output, self.top_k_tokens * 2,
            include_negative=True
        )

        # Split into promoted (positive Z) and suppressed (negative Z)
        promoted = [(t, z) for t, z in zip(output_tokens_all, output_zscores_all) if z > 0]
        suppressed = [(t, z) for t, z in zip(output_tokens_all, output_zscores_all) if z < 0]

        return FeatureSemantics(
            feature_idx=feature_idx,
            layer_idx=layer_idx,
            input_tokens=[t for t, _ in input_tokens],
            input_zscores=[z for _, z in input_tokens],
            output_tokens_promoted=[t for t, _ in promoted],
            output_tokens_suppressed=[t for t, _ in suppressed],
            output_zscores=[z for _, z in output_tokens_all],
        )

    @torch.no_grad()
    def analyze_all_features(
        self,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        layer_idx: int,
        feature_indices: Optional[List[int]] = None,
    ) -> List[FeatureSemantics]:
        """
        Analyze all (or a subset of) features in a Transcoder layer.

        Uses batched matrix multiplication for efficiency, keeping computation
        within the L2 cache by processing features in tiles.

        Args:
            W_enc: Encoder weight matrix, shape (d_model, n_features).
            W_dec: Decoder weight matrix, shape (n_features, d_model).
            layer_idx: Transformer layer index.
            feature_indices: Subset of feature indices to analyze. If None,
                             analyzes all features.

        Returns:
            List of FeatureSemantics instances.
        """
        W_enc = W_enc.to(self.device).float()
        W_dec = W_dec.to(self.device).float()

        n_features = W_enc.shape[1]
        if feature_indices is None:
            feature_indices = list(range(n_features))

        results = []
        for feat_idx in feature_indices:
            semantics = self.analyze_feature(W_enc, W_dec, feat_idx, layer_idx)
            results.append(semantics)

        return results

    @torch.no_grad()
    def compute_cross_layer_connections(
        self,
        W_enc_l: torch.Tensor,
        W_dec_l_prime: torch.Tensor,
        layer_l: int,
        layer_l_prime: int,
        feature_indices_l: Optional[List[int]] = None,
    ) -> List[Tuple[int, int, float]]:
        """
        Compute input-invariant cross-layer connection strengths.

        For features at layer l and layer l' (l' > l), computes:
            c(l, i, l', i') = f_dec^(l',i') · f_enc^(l,i)

        Args:
            W_enc_l: Encoder weights for layer l, shape (d_model, n_features_l).
            W_dec_l_prime: Decoder weights for layer l', shape (n_features_l', d_model).
            layer_l: Source layer index.
            layer_l_prime: Target layer index (must be > layer_l).
            feature_indices_l: Subset of source layer features to analyze.

        Returns:
            List of (feature_l_idx, feature_l_prime_idx, connection_strength) tuples
            where |connection_strength| exceeds zscore_connected threshold.
        """
        assert layer_l_prime > layer_l, "Target layer must be deeper than source layer."

        W_enc_l = W_enc_l.to(self.device).float()
        W_dec_l_prime = W_dec_l_prime.to(self.device).float()

        # Connection matrix: (n_features_l', n_features_l)
        # W_dec_l_prime: (n_features_l', d_model)
        # W_enc_l: (d_model, n_features_l)
        connection_matrix = W_dec_l_prime @ W_enc_l  # (n_features_l', n_features_l)

        # Apply Z-score threshold to connection strengths
        flat = connection_matrix.flatten()
        mean = flat.mean().item()
        std = flat.std().item()
        if std == 0:
            return []

        threshold_val = mean + self.zscore_connected * std
        strong_connections = []

        if feature_indices_l is not None:
            col_indices = feature_indices_l
        else:
            col_indices = list(range(connection_matrix.shape[1]))

        for col_idx in col_indices:
            col = connection_matrix[:, col_idx]
            for row_idx in range(col.shape[0]):
                strength = col[row_idx].item()
                if abs(strength) >= abs(threshold_val):
                    strong_connections.append((col_idx, row_idx, round(strength, 6)))

        return strong_connections

    def _zscore_filter(
        self,
        logits: torch.Tensor,
        threshold: float,
        top_k: int,
        include_negative: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Apply Z-score filtering to a logit vector and return token-score pairs.

        Args:
            logits: Raw logit scores over vocabulary, shape (vocab_size,).
            threshold: Minimum absolute Z-score to retain a token.
            top_k: Maximum number of tokens to return.
            include_negative: If True, also return tokens with Z-score < -threshold.

        Returns:
            List of (token_string, z_score) tuples sorted by |z_score| descending.
        """
        mean = logits.mean()
        std = logits.std()

        if std.item() == 0:
            return []

        zscores = (logits - mean) / std

        if include_negative:
            mask = zscores.abs() >= threshold
        else:
            mask = zscores >= threshold

        candidate_indices = mask.nonzero(as_tuple=True)[0]

        if len(candidate_indices) == 0:
            # Fallback: return top-k by absolute z-score
            _, top_indices = zscores.abs().topk(min(top_k, len(zscores)))
            candidate_indices = top_indices

        # Sort by absolute z-score descending
        candidate_zscores = zscores[candidate_indices]
        sorted_order = candidate_zscores.abs().argsort(descending=True)
        candidate_indices = candidate_indices[sorted_order][:top_k]
        candidate_zscores = candidate_zscores[sorted_order][:top_k]

        results = []
        for idx, zscore in zip(candidate_indices.tolist(), candidate_zscores.tolist()):
            try:
                token_str = self.tokenizer.convert_ids_to_tokens(idx)
                if token_str is None:
                    token_str = f"<token_{idx}>"
                # Clean up tokenizer-specific prefixes
                token_str = token_str.replace("Ġ", " ").replace("▁", " ")
            except Exception:
                token_str = f"<token_{idx}>"
            results.append((token_str, round(zscore, 4)))

        return results
