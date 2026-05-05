"""
Unit Tests for WeightLens Static Weight Semantic Analysis Engine.

Tests cover:
    - VocabProjector: Z-score filtering, input/output projection
    - SemanticLemmatizer: Token cleaning, lemmatization, label generation
    - FeatureCache: CRUD operations, disk persistence, atomic writes
    - TranscoderLoader: Synthetic transcoder creation, state dict parsing
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import numpy as np


class TestVocabProjector(unittest.TestCase):
    """Tests for VocabProjector vocabulary space projection."""

    def setUp(self):
        """Create a minimal VocabProjector with synthetic weights."""
        from src.weightlens.projection import VocabProjector

        vocab_size = 1000
        d_model = 64
        n_features = 32

        # Synthetic embedding and unembedding matrices
        torch.manual_seed(42)
        self.W_embed = torch.randn(vocab_size, d_model)
        self.W_unembed = torch.randn(vocab_size, d_model)

        # Mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.convert_ids_to_tokens.side_effect = (
            lambda idx: f"token_{idx}"
        )

        self.projector = VocabProjector(
            W_embed=self.W_embed,
            W_unembed=self.W_unembed,
            tokenizer=self.mock_tokenizer,
            zscore_input=2.0,  # Lower threshold for testing
            zscore_output=2.0,
            top_k_tokens=10,
            device="cpu",
        )

        # Synthetic transcoder weights
        self.W_enc = torch.randn(d_model, n_features)
        self.W_dec = torch.randn(n_features, d_model)

    def test_analyze_feature_returns_semantics(self):
        """analyze_feature should return a FeatureSemantics with non-empty tokens."""
        from src.weightlens.projection import FeatureSemantics

        result = self.projector.analyze_feature(
            self.W_enc, self.W_dec, feature_idx=0, layer_idx=3
        )

        self.assertIsInstance(result, FeatureSemantics)
        self.assertEqual(result.feature_idx, 0)
        self.assertEqual(result.layer_idx, 3)
        # Should have at least some input tokens
        self.assertGreater(len(result.input_tokens), 0)

    def test_zscore_filter_returns_sorted_by_magnitude(self):
        """Z-score filter should return tokens sorted by |z-score| descending."""
        logits = torch.randn(100)
        results = self.projector._zscore_filter(logits, threshold=0.5, top_k=10)

        if len(results) >= 2:
            zscores = [abs(z) for _, z in results]
            self.assertEqual(zscores, sorted(zscores, reverse=True))

    def test_cross_layer_connections_shape(self):
        """Cross-layer connection computation should return valid tuples."""
        connections = self.projector.compute_cross_layer_connections(
            W_enc_l=self.W_enc,
            W_dec_l_prime=self.W_dec,
            layer_l=2,
            layer_l_prime=5,
            feature_indices_l=list(range(5)),
        )

        # Each connection should be a (int, int, float) tuple
        for conn in connections:
            self.assertEqual(len(conn), 3)
            self.assertIsInstance(conn[0], int)
            self.assertIsInstance(conn[1], int)
            self.assertIsInstance(conn[2], float)

    def test_feature_semantics_to_dict(self):
        """FeatureSemantics.to_dict() should produce a JSON-serializable dict."""
        result = self.projector.analyze_feature(
            self.W_enc, self.W_dec, feature_idx=5, layer_idx=0
        )
        d = result.to_dict()

        self.assertIn("feature_idx", d)
        self.assertIn("input_tokens", d)
        self.assertIn("output_tokens_promoted", d)

        # Should be JSON serializable
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)


class TestSemanticLemmatizer(unittest.TestCase):
    """Tests for SemanticLemmatizer token processing and label generation."""

    def setUp(self):
        from src.weightlens.lemmatizer import SemanticLemmatizer
        # Use None for spacy_model to avoid download requirement in tests
        self.lemmatizer = SemanticLemmatizer(spacy_model=None)

    def test_clean_token_removes_artifacts(self):
        """Token cleaning should remove tokenizer-specific artifacts."""
        test_cases = [
            ("Ġrunning", "running"),
            ("▁walking", "walking"),
            ("##ing", "ing"),
            ("<s>", ""),
            ("  word  ", "word"),
        ]
        for raw, expected in test_cases:
            cleaned = self.lemmatizer._clean_token(raw)
            self.assertEqual(
                cleaned, expected,
                f"_clean_token('{raw}') = '{cleaned}', expected '{expected}'"
            )

    def test_generate_label_with_tokens(self):
        """generate_label should produce a non-empty structured label."""
        label = self.lemmatizer.generate_label(
            input_tokens=["running", "walking", "jogging"],
            output_tokens_promoted=["motion", "movement"],
            output_tokens_suppressed=["stop", "halt"],
        )
        self.assertIsInstance(label, str)
        self.assertGreater(len(label), 0)
        self.assertNotEqual(label, "Unclassified feature")

    def test_generate_label_empty_tokens(self):
        """generate_label with empty token lists should return fallback label."""
        label = self.lemmatizer.generate_label(
            input_tokens=[],
            output_tokens_promoted=[],
        )
        self.assertEqual(label, "Unclassified feature")

    def test_consolidate_tokens_deduplicates(self):
        """Consolidation should merge inflected forms into unique base forms."""
        tokens = ["running", "runs", "ran", "run"]
        zscores = [4.5, 4.2, 3.9, 3.7]
        result = self.lemmatizer._consolidate_tokens(tokens, zscores, max_count=5)

        # Should have fewer unique lemmas than input tokens
        self.assertLessEqual(len(result), len(tokens))
        self.assertGreater(len(result), 0)


class TestFeatureCache(unittest.TestCase):
    """Tests for FeatureCache two-level caching system."""

    def setUp(self):
        from src.weightlens.projection import FeatureSemantics
        self.FeatureSemantics = FeatureSemantics
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_cache(self):
        from src.weightlens.cache import FeatureCache
        return FeatureCache(
            model_name="test_model",
            cache_root=self.temp_dir,
            auto_save=True,
        )

    def _make_semantics(self, feature_idx: int, layer_idx: int):
        return self.FeatureSemantics(
            feature_idx=feature_idx,
            layer_idx=layer_idx,
            input_tokens=["run", "walk"],
            input_zscores=[4.5, 4.2],
            output_tokens_promoted=["motion"],
            raw_label="Input: run, walk | Promotes: motion",
        )

    def test_store_and_retrieve(self):
        """Stored features should be retrievable from memory cache."""
        cache = self._make_cache()
        sem = self._make_semantics(feature_idx=0, layer_idx=3)
        cache.store(sem)

        retrieved = cache.get(layer_idx=3, feature_idx=0)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.feature_idx, 0)
        self.assertEqual(retrieved.raw_label, sem.raw_label)

    def test_disk_persistence(self):
        """Features should persist to disk and be loadable in a new cache instance."""
        cache1 = self._make_cache()
        sem = self._make_semantics(feature_idx=5, layer_idx=2)
        cache1.store(sem, save_immediately=True)

        # Create a new cache instance pointing to same directory
        cache2 = self._make_cache()
        retrieved = cache2.get(layer_idx=2, feature_idx=5)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.feature_idx, 5)
        self.assertEqual(retrieved.input_tokens, ["run", "walk"])

    def test_cache_miss_returns_none(self):
        """Cache miss should return None without raising exceptions."""
        cache = self._make_cache()
        result = cache.get(layer_idx=99, feature_idx=999)
        self.assertIsNone(result)

    def test_is_cached(self):
        """is_cached should correctly report cache hit/miss status."""
        cache = self._make_cache()
        sem = self._make_semantics(feature_idx=7, layer_idx=4)

        self.assertFalse(cache.is_cached(layer_idx=4, feature_idx=7))
        cache.store(sem)
        self.assertTrue(cache.is_cached(layer_idx=4, feature_idx=7))

    def test_clear_memory(self):
        """clear_memory should remove in-memory cache while preserving disk."""
        cache = self._make_cache()
        sem = self._make_semantics(feature_idx=1, layer_idx=0)
        cache.store(sem, save_immediately=True)

        cache.clear_memory()
        self.assertEqual(len(cache._memory), 0)

        # Should still be loadable from disk
        retrieved = cache.get(layer_idx=0, feature_idx=1)
        self.assertIsNotNone(retrieved)

    def test_stats(self):
        """stats() should return a valid statistics dictionary."""
        cache = self._make_cache()
        stats = cache.stats()
        self.assertIn("model_name", stats)
        self.assertIn("layers_in_memory", stats)
        self.assertIn("features_in_memory", stats)


class TestTranscoderLoader(unittest.TestCase):
    """Tests for TranscoderLoader weight loading and synthetic creation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_synthetic_transcoder(self):
        """Synthetic transcoder should have correct tensor shapes."""
        from src.weightlens.transcoder_loader import TranscoderLoader

        loader = TranscoderLoader(
            model_name="test_model",
            local_cache_dir=self.temp_dir,
        )
        weights = loader.create_synthetic_transcoder(
            layer_idx=3, d_model=512, n_features=2048
        )

        self.assertEqual(weights.layer_idx, 3)
        self.assertEqual(weights.W_enc.shape, (512, 2048))
        self.assertEqual(weights.W_dec.shape, (2048, 512))
        self.assertEqual(weights.n_features, 2048)
        self.assertEqual(weights.d_model, 512)

    def test_save_and_load_pt(self):
        """Saved .pt files should be loadable and produce identical weights."""
        from src.weightlens.transcoder_loader import TranscoderLoader

        loader = TranscoderLoader(
            model_name="test_model",
            local_cache_dir=self.temp_dir,
        )
        original = loader.create_synthetic_transcoder(
            layer_idx=0, d_model=64, n_features=128
        )

        # Save to disk
        save_path = Path(self.temp_dir) / "layer_00.pt"
        loader._save_to_pt(original, save_path)

        # Load back
        loaded = loader._load_from_pt(save_path, layer_idx=0)
        self.assertIsNotNone(loaded)
        self.assertTrue(torch.allclose(original.W_enc, loaded.W_enc, atol=1e-6))
        self.assertTrue(torch.allclose(original.W_dec, loaded.W_dec, atol=1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
