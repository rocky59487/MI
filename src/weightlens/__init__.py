"""
WeightLens: Static weight semantic analysis engine.

This module provides dataset-free, explainer-free feature semantic extraction
by directly analyzing the learned weight matrices of Transcoder features.

Key components:
    - transcoder_loader: Load Transcoder weight dictionaries from Hugging Face
    - projection: Vocabulary space projection and Z-score filtering
    - lemmatizer: Local deterministic lemmatization via spaCy
    - cache: JSON-based precomputed cache library for millisecond-speed lookup
"""

from .transcoder_loader import TranscoderLoader
from .projection import VocabProjector
from .lemmatizer import SemanticLemmatizer
from .cache import FeatureCache

__all__ = [
    "TranscoderLoader",
    "VocabProjector",
    "SemanticLemmatizer",
    "FeatureCache",
]
