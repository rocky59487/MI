"""
Local Deterministic Semantic Lemmatization for WeightLens.

This module consolidates extracted high-Z-score tokens into human-readable
semantic labels using local spaCy lemmatization. This completely replaces
the need for external LLM post-processing (e.g., GPT-4), ensuring:
    - Deterministic, reproducible results
    - Zero external API cost
    - Zero epistemic contamination from external model biases
    - Full offline / Air-Gapped operation

Lemmatization pipeline:
    1. Strip tokenizer artifacts (Ġ, ▁, ##, etc.)
    2. spaCy lemmatization: "running" -> "run", "dogs" -> "dog"
    3. Frequency-based consolidation: group inflected forms
    4. Generate structured label: "Action: run | Noun: dog"
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

# spaCy is optional; graceful fallback to simple rule-based lemmatization
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Simple rule-based fallback lemmatizer (no spaCy dependency)
# ---------------------------------------------------------------------------

_COMMON_SUFFIXES = [
    ("nesses", "ness"), ("ments", "ment"), ("tions", "tion"),
    ("ings", "ing"), ("ness", ""), ("ment", ""), ("tion", ""),
    ("ies", "y"), ("ves", "f"), ("ses", "s"), ("es", ""),
    ("ing", ""), ("ied", "y"), ("ed", ""), ("er", ""),
    ("est", ""), ("ly", ""), ("s", ""),
]


def _simple_lemmatize(word: str) -> str:
    """Rule-based lemmatization fallback when spaCy is unavailable."""
    word = word.lower().strip()
    for suffix, replacement in _COMMON_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    return word


# ---------------------------------------------------------------------------
# SemanticLemmatizer
# ---------------------------------------------------------------------------

class SemanticLemmatizer:
    """
    Converts lists of high-Z-score tokens into consolidated semantic labels.

    Uses spaCy for accurate lemmatization when available, with a rule-based
    fallback for offline environments without spaCy model downloads.

    Args:
        spacy_model: spaCy model name to load (e.g., "en_core_web_sm").
                     If None or unavailable, falls back to rule-based lemmatization.
        override_dict: Optional dictionary mapping raw tokens to custom labels.
                       Allows domain-specific overrides without modifying source code.
        max_label_tokens: Maximum number of unique lemmas to include in a label.
        pos_categories: POS tags to include in label generation.
    """

    def __init__(
        self,
        spacy_model: Optional[str] = "en_core_web_sm",
        override_dict: Optional[Dict[str, str]] = None,
        max_label_tokens: int = 5,
        pos_categories: Optional[List[str]] = None,
    ):
        self.override_dict = override_dict or {}
        self.max_label_tokens = max_label_tokens
        self.pos_categories = pos_categories or ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]
        self._nlp = None

        if _SPACY_AVAILABLE and spacy_model:
            try:
                self._nlp = spacy.load(spacy_model)
            except OSError:
                # Model not downloaded; try to download it
                try:
                    import subprocess, sys
                    subprocess.run(
                        [sys.executable, "-m", "spacy", "download", spacy_model],
                        check=True, capture_output=True,
                    )
                    self._nlp = spacy.load(spacy_model)
                except Exception:
                    self._nlp = None

    def lemmatize_token(self, token: str) -> str:
        """
        Lemmatize a single token string.

        Args:
            token: Raw token string (may contain tokenizer artifacts).

        Returns:
            Lemmatized base form of the token.
        """
        # Check override dictionary first
        if token in self.override_dict:
            return self.override_dict[token]

        # Clean tokenizer artifacts
        cleaned = self._clean_token(token)
        if not cleaned:
            return token

        # Apply override after cleaning
        if cleaned in self.override_dict:
            return self.override_dict[cleaned]

        # spaCy lemmatization
        if self._nlp is not None:
            doc = self._nlp(cleaned)
            if doc:
                return doc[0].lemma_.lower()

        # Fallback: simple rule-based
        return _simple_lemmatize(cleaned)

    def generate_label(
        self,
        input_tokens: List[str],
        output_tokens_promoted: List[str],
        output_tokens_suppressed: Optional[List[str]] = None,
        input_zscores: Optional[List[float]] = None,
    ) -> str:
        """
        Generate a consolidated human-readable semantic label for a feature.

        Args:
            input_tokens: High-Z-score input tokens (driving vocabulary).
            output_tokens_promoted: Tokens strongly promoted by this feature.
            output_tokens_suppressed: Tokens strongly suppressed by this feature.
            input_zscores: Z-scores for input tokens (used for weighting).

        Returns:
            A structured label string, e.g.:
            "Input: run, walk, move | Promotes: action, motion | Suppresses: stop"
        """
        # Lemmatize and consolidate input tokens
        input_lemmas = self._consolidate_tokens(
            input_tokens, input_zscores, self.max_label_tokens
        )
        promoted_lemmas = self._consolidate_tokens(
            output_tokens_promoted, None, self.max_label_tokens
        )
        suppressed_lemmas = self._consolidate_tokens(
            output_tokens_suppressed or [], None, max(2, self.max_label_tokens // 2)
        )

        parts = []
        if input_lemmas:
            parts.append(f"Input: {', '.join(input_lemmas)}")
        if promoted_lemmas:
            parts.append(f"Promotes: {', '.join(promoted_lemmas)}")
        if suppressed_lemmas:
            parts.append(f"Suppresses: {', '.join(suppressed_lemmas)}")

        if not parts:
            return "Unclassified feature"

        return " | ".join(parts)

    def _consolidate_tokens(
        self,
        tokens: List[str],
        zscores: Optional[List[float]],
        max_count: int,
    ) -> List[str]:
        """
        Lemmatize tokens and consolidate inflected forms into unique base forms.

        Tokens are weighted by their Z-scores if provided; otherwise by frequency.
        """
        if not tokens:
            return []

        lemma_scores: Dict[str, float] = {}
        for i, token in enumerate(tokens):
            lemma = self.lemmatize_token(token)
            if not lemma or len(lemma) < 2:
                continue
            score = zscores[i] if zscores and i < len(zscores) else 1.0
            lemma_scores[lemma] = lemma_scores.get(lemma, 0.0) + abs(score)

        # Sort by accumulated score descending
        sorted_lemmas = sorted(lemma_scores.items(), key=lambda x: x[1], reverse=True)
        return [lemma for lemma, _ in sorted_lemmas[:max_count]]

    @staticmethod
    def _clean_token(token: str) -> str:
        """Remove tokenizer-specific artifacts from a token string."""
        # Remove common tokenizer prefixes/suffixes
        token = token.strip()
        token = token.replace("Ġ", " ").replace("▁", " ")
        token = token.replace("##", "").replace("<0x", "").replace(">", "")
        token = re.sub(r"<[^>]+>", "", token)  # Remove special tokens like <s>
        token = token.strip()
        # Keep only alphabetic content for lemmatization
        alpha_only = re.sub(r"[^a-zA-Z\-']", "", token)
        return alpha_only.lower() if len(alpha_only) >= 2 else ""
