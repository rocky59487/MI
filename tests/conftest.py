"""
Pytest Configuration and Shared Fixtures for MI Toolkit Tests.

This conftest.py provides:
    - Synthetic model fixtures (no real model download required)
    - Temporary directory management
    - Common test data (IOI prompts, synthetic transcoder weights)
    - Markers for slow integration tests (require GPU)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (requires GPU or model download)"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires real model)"
    )
    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring CUDA GPU"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device() -> str:
    """Return the appropriate device string for the test session."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def d_model() -> int:
    """Standard model dimension for synthetic tests."""
    return 64


@pytest.fixture(scope="session")
def vocab_size() -> int:
    """Standard vocabulary size for synthetic tests."""
    return 1000


@pytest.fixture(scope="session")
def n_features() -> int:
    """Standard number of transcoder features for synthetic tests."""
    return 128


@pytest.fixture(scope="session")
def synthetic_embeddings(vocab_size, d_model):
    """Synthetic embedding and unembedding matrices."""
    torch.manual_seed(42)
    W_embed = torch.randn(vocab_size, d_model)
    W_unembed = torch.randn(vocab_size, d_model)
    return W_embed, W_unembed


@pytest.fixture(scope="session")
def synthetic_transcoder_weights(d_model, n_features):
    """Synthetic transcoder weight matrices for a single layer."""
    torch.manual_seed(123)
    W_enc = torch.randn(d_model, n_features) * 0.02
    W_dec = torch.randn(n_features, d_model) * 0.02
    return W_enc, W_dec


@pytest.fixture
def temp_cache_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for cache tests, cleaned up after each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="session")
def ioi_prompts():
    """Standard IOI task prompt pairs for fidelity testing."""
    return [
        (
            "When Mary and John went to the store, John gave a drink to",
            "When Mary and John went to the store, Mary gave a drink to",
            " Mary",
        ),
        (
            "When Alice and Bob went to the park, Bob gave a ball to",
            "When Alice and Bob went to the park, Alice gave a ball to",
            " Alice",
        ),
    ]


@pytest.fixture(scope="session")
def mock_tokenizer(vocab_size):
    """Mock tokenizer that returns predictable token strings."""
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.convert_ids_to_tokens.side_effect = lambda idx: f"token_{idx}"
    tokenizer.vocab_size = vocab_size
    return tokenizer


@pytest.fixture
def feature_cache(temp_cache_dir):
    """Fresh FeatureCache instance backed by a temporary directory."""
    from src.weightlens.cache import FeatureCache
    return FeatureCache(
        model_name="test_model",
        cache_root=temp_cache_dir,
        auto_save=True,
    )


@pytest.fixture
def vocab_projector(synthetic_embeddings, mock_tokenizer):
    """VocabProjector instance with synthetic weights."""
    from src.weightlens.projection import VocabProjector
    W_embed, W_unembed = synthetic_embeddings
    return VocabProjector(
        W_embed=W_embed,
        W_unembed=W_unembed,
        tokenizer=mock_tokenizer,
        zscore_input=2.0,
        zscore_output=2.0,
        top_k_tokens=10,
        device="cpu",
    )
