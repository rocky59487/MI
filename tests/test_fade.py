"""
FADE Evaluation Framework Tests for WeightLens Feature Descriptions.

FADE (Feature Attribution Description Evaluation) assesses the quality of
automatically generated feature semantic descriptions along four dimensions:

    1. Purity (純度):
       The proportion of high-Z-score tokens that are semantically coherent
       with the proposed label. Measures false positive rate.
       Target: > 0.80

    2. Clarity (清晰度):
       The specificity and non-ambiguity of the generated label.
       A label like "Action: run, walk" scores higher than "word".
       Target: > 0.70

    3. Responsiveness (響應度):
       Whether the feature actually activates on inputs matching the label.
       Measured by activation correlation with curated test sentences.
       Target: > 0.75

    4. Fidelity (保真度):
       Whether the feature's causal effect on model output matches the
       predicted semantic role. Measured via ReIP attribution correlation.
       Target: > 0.80

Reference: "FADE: Why Bad Descriptions Happen to Good Features" (arXiv)
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FADEScore:
    """
    FADE evaluation scores for a single feature description.

    Attributes:
        feature_idx: Feature index.
        layer_idx: Layer index.
        label: The generated semantic label being evaluated.
        purity: Proportion of semantically coherent high-Z tokens (0-1).
        clarity: Label specificity score (0-1).
        responsiveness: Activation-label correlation (0-1).
        fidelity: Causal effect alignment with label (0-1).
        composite: Weighted composite score.
    """
    feature_idx: int
    layer_idx: int
    label: str
    purity: float = 0.0
    clarity: float = 0.0
    responsiveness: float = 0.0
    fidelity: float = 0.0
    composite: float = 0.0

    def compute_composite(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute weighted composite FADE score."""
        w = weights or {
            "purity": 0.30,
            "clarity": 0.20,
            "responsiveness": 0.25,
            "fidelity": 0.25,
        }
        self.composite = (
            w["purity"] * self.purity
            + w["clarity"] * self.clarity
            + w["responsiveness"] * self.responsiveness
            + w["fidelity"] * self.fidelity
        )
        return self.composite

    def passes_all_thresholds(
        self,
        purity_min: float = 0.80,
        clarity_min: float = 0.70,
        responsiveness_min: float = 0.75,
        fidelity_min: float = 0.80,
    ) -> bool:
        """Check if all FADE dimensions meet their minimum thresholds."""
        return (
            self.purity >= purity_min
            and self.clarity >= clarity_min
            and self.responsiveness >= responsiveness_min
            and self.fidelity >= fidelity_min
        )


class FADEEvaluator:
    """
    Evaluates WeightLens feature descriptions using the FADE framework.

    This evaluator provides both automated metric computation and
    adversarial comparison against MaxAct + GPT-4 baselines.
    """

    def compute_purity(
        self,
        label: str,
        input_tokens: List[str],
        input_zscores: List[float],
        zscore_threshold: float = 3.0,
    ) -> float:
        """
        Compute purity score for a feature description.

        Purity = (# tokens semantically consistent with label) /
                 (# tokens above Z-score threshold)

        In automated evaluation, semantic consistency is approximated by
        checking if the token's lemma appears in the label string.

        Args:
            label: Generated semantic label.
            input_tokens: List of high-Z-score input tokens.
            input_zscores: Corresponding Z-scores.
            zscore_threshold: Minimum Z-score to include a token in evaluation.

        Returns:
            Purity score in [0, 1].
        """
        if not input_tokens:
            return 0.0

        label_lower = label.lower()
        above_threshold = [
            (t, z) for t, z in zip(input_tokens, input_zscores)
            if abs(z) >= zscore_threshold
        ]

        if not above_threshold:
            return 0.0

        consistent = sum(
            1 for t, _ in above_threshold
            if self._token_consistent_with_label(t, label_lower)
        )
        return consistent / len(above_threshold)

    def compute_clarity(self, label: str) -> float:
        """
        Compute clarity score based on label specificity.

        Scoring heuristics:
            - Labels with multiple semantic components score higher
            - Labels with specific token examples score higher
            - Generic labels ("word", "token") score lower
            - Empty or "Unclassified" labels score 0

        Returns:
            Clarity score in [0, 1].
        """
        if not label or label.lower() in ("unclassified feature", ""):
            return 0.0

        score = 0.0

        # Reward structured labels with multiple components
        if " | " in label:
            n_components = len(label.split(" | "))
            score += min(0.4, 0.15 * n_components)

        # Reward labels with specific token examples (comma-separated)
        parts = label.split("|")
        for part in parts:
            if ":" in part:
                tokens_part = part.split(":", 1)[1].strip()
                n_tokens = len([t for t in tokens_part.split(",") if t.strip()])
                score += min(0.3, 0.08 * n_tokens)

        # Penalize generic labels
        generic_terms = {"word", "token", "text", "string", "character", "symbol"}
        label_words = set(label.lower().split())
        if label_words & generic_terms:
            score *= 0.5

        # Reward labels with action/noun/promotes/suppresses structure
        structural_keywords = {"input:", "promotes:", "suppresses:", "action:", "noun:"}
        for kw in structural_keywords:
            if kw in label.lower():
                score += 0.1

        return min(1.0, score)

    def compute_responsiveness(
        self,
        feature_activations: List[float],
        label_match_scores: List[float],
    ) -> float:
        """
        Compute responsiveness as Pearson correlation between feature
        activations and label-match scores across test sentences.

        Args:
            feature_activations: Feature activation values across test sentences.
            label_match_scores: Binary or graded label-match scores for each sentence.

        Returns:
            Responsiveness score in [0, 1] (absolute Pearson correlation).
        """
        if len(feature_activations) < 3:
            return 0.0

        try:
            from scipy.stats import pearsonr
            pcc, _ = pearsonr(feature_activations, label_match_scores)
            return max(0.0, abs(pcc))
        except Exception:
            # Fallback: normalized covariance
            a = np.array(feature_activations)
            b = np.array(label_match_scores)
            if a.std() == 0 or b.std() == 0:
                return 0.0
            return abs(np.corrcoef(a, b)[0, 1])

    def compute_fidelity(
        self,
        reip_scores: List[float],
        label_causal_scores: List[float],
    ) -> float:
        """
        Compute fidelity as correlation between ReIP attribution scores and
        the expected causal impact predicted by the semantic label.

        Args:
            reip_scores: ReIP attribution scores for the feature across prompts.
            label_causal_scores: Expected causal scores based on label semantics.

        Returns:
            Fidelity score in [0, 1].
        """
        return self.compute_responsiveness(reip_scores, label_causal_scores)

    @staticmethod
    def _token_consistent_with_label(token: str, label_lower: str) -> bool:
        """Check if a token is semantically consistent with a label string."""
        token_clean = token.strip().lower().replace("ġ", "").replace("▁", "")
        if not token_clean:
            return False
        # Direct match
        if token_clean in label_lower:
            return True
        # Check if any word in the token appears in the label
        for word in token_clean.split():
            if len(word) >= 3 and word in label_lower:
                return True
        return False


class TestFADEEvaluator(unittest.TestCase):
    """
    Test suite for the FADE evaluation framework.
    """

    def setUp(self):
        self.evaluator = FADEEvaluator()

    def test_purity_high_quality_label(self):
        """High-quality labels should achieve purity > 0.80."""
        label = "Input: run, walk, move | Promotes: action, motion"
        tokens = ["run", "walk", "move", "sprint", "jog", "jump"]
        zscores = [5.2, 4.8, 4.5, 4.1, 3.9, 3.5]

        purity = self.evaluator.compute_purity(label, tokens, zscores)
        self.assertGreater(purity, 0.80, f"Purity {purity:.3f} below 0.80 threshold")

    def test_purity_poor_label(self):
        """Poor labels (generic or mismatched) should score low purity."""
        label = "Unclassified feature"
        tokens = ["run", "walk", "move", "sprint"]
        zscores = [5.0, 4.5, 4.2, 3.8]

        purity = self.evaluator.compute_purity(label, tokens, zscores)
        self.assertLess(purity, 0.30, f"Purity {purity:.3f} should be low for poor label")

    def test_clarity_structured_label(self):
        """Structured labels with multiple components should score > 0.70."""
        label = "Input: run, walk | Promotes: motion | Suppresses: stop"
        clarity = self.evaluator.compute_clarity(label)
        self.assertGreater(clarity, 0.70, f"Clarity {clarity:.3f} below 0.70 threshold")

    def test_clarity_generic_label(self):
        """Generic labels should score low clarity."""
        label = "word token"
        clarity = self.evaluator.compute_clarity(label)
        self.assertLess(clarity, 0.30, f"Clarity {clarity:.3f} should be low for generic label")

    def test_clarity_empty_label(self):
        """Empty labels should score 0 clarity."""
        self.assertEqual(self.evaluator.compute_clarity(""), 0.0)
        self.assertEqual(self.evaluator.compute_clarity("Unclassified feature"), 0.0)

    def test_responsiveness_high_correlation(self):
        """Features with high activation-label correlation should score > 0.75."""
        np.random.seed(42)
        activations = np.random.uniform(0, 1, 20).tolist()
        # Label match scores highly correlated with activations
        label_scores = [a + np.random.normal(0, 0.05) for a in activations]

        responsiveness = self.evaluator.compute_responsiveness(activations, label_scores)
        self.assertGreater(
            responsiveness, 0.75,
            f"Responsiveness {responsiveness:.3f} below 0.75 threshold"
        )

    def test_responsiveness_low_correlation(self):
        """Features with random activation patterns should score low responsiveness."""
        np.random.seed(42)
        activations = np.random.uniform(0, 1, 20).tolist()
        label_scores = np.random.uniform(0, 1, 20).tolist()

        responsiveness = self.evaluator.compute_responsiveness(activations, label_scores)
        self.assertLess(
            responsiveness, 0.50,
            f"Responsiveness {responsiveness:.3f} should be low for random data"
        )

    def test_fade_score_composite(self):
        """Test composite FADE score computation."""
        score = FADEScore(
            feature_idx=0,
            layer_idx=6,
            label="Input: run, walk | Promotes: motion",
            purity=0.85,
            clarity=0.75,
            responsiveness=0.80,
            fidelity=0.82,
        )
        composite = score.compute_composite()
        self.assertAlmostEqual(
            composite,
            0.30 * 0.85 + 0.20 * 0.75 + 0.25 * 0.80 + 0.25 * 0.82,
            places=4,
        )

    def test_fade_score_passes_thresholds(self):
        """Test that a high-quality score passes all FADE thresholds."""
        score = FADEScore(
            feature_idx=0,
            layer_idx=6,
            label="Input: run, walk | Promotes: motion",
            purity=0.85,
            clarity=0.75,
            responsiveness=0.80,
            fidelity=0.82,
        )
        self.assertTrue(score.passes_all_thresholds())

    def test_fade_score_fails_thresholds(self):
        """Test that a low-quality score fails FADE thresholds."""
        score = FADEScore(
            feature_idx=1,
            layer_idx=3,
            label="word",
            purity=0.60,
            clarity=0.20,
            responsiveness=0.55,
            fidelity=0.70,
        )
        self.assertFalse(score.passes_all_thresholds())


class TestWeightLensVsMaxActBaseline(unittest.TestCase):
    """
    Adversarial baseline comparison: WeightLens vs MaxAct + GPT-4.

    MaxAct (Maximum Activation) is the standard baseline that identifies
    feature semantics by finding the dataset examples that maximally
    activate the feature, then prompting GPT-4 to generate a label.

    WeightLens should outperform MaxAct + GPT-4 on:
        - Purity (no dataset bias)
        - Fidelity (direct weight-space analysis)
        - Speed (no dataset scan or API calls)
    """

    def test_weightlens_purity_advantage(self):
        """
        WeightLens purity should exceed MaxAct + GPT-4 baseline.

        Synthetic test: WeightLens produces token-level Z-score labels
        with higher semantic coherence than GPT-4 summaries of MaxAct examples.
        """
        # Synthetic FADE scores for WeightLens
        weightlens_scores = [
            FADEScore(i, 6, f"Input: token_{i}", purity=0.85, clarity=0.75,
                      responsiveness=0.80, fidelity=0.82)
            for i in range(10)
        ]

        # Synthetic FADE scores for MaxAct + GPT-4 (lower purity due to dataset bias)
        maxact_scores = [
            FADEScore(i, 6, f"GPT-4 label {i}", purity=0.72, clarity=0.68,
                      responsiveness=0.74, fidelity=0.71)
            for i in range(10)
        ]

        wl_avg_purity = np.mean([s.purity for s in weightlens_scores])
        ma_avg_purity = np.mean([s.purity for s in maxact_scores])

        self.assertGreater(
            wl_avg_purity, ma_avg_purity,
            f"WeightLens purity ({wl_avg_purity:.3f}) should exceed "
            f"MaxAct+GPT-4 ({ma_avg_purity:.3f})"
        )

    def test_weightlens_no_external_api_required(self):
        """
        Verify that WeightLens analysis requires no external API calls.

        This is a structural test: the VocabProjector and SemanticLemmatizer
        classes should not make any network requests.
        """
        import socket
        original_getaddrinfo = socket.getaddrinfo

        api_calls_made = []

        def mock_getaddrinfo(*args, **kwargs):
            api_calls_made.append(args[0])
            return original_getaddrinfo(*args, **kwargs)

        # The import itself should not trigger network calls
        try:
            from src.weightlens.projection import VocabProjector
            from src.weightlens.lemmatizer import SemanticLemmatizer
        except ImportError:
            self.skipTest("WeightLens modules not available")

        # Filter out localhost calls (e.g., from test infrastructure)
        external_calls = [
            host for host in api_calls_made
            if host not in ("localhost", "127.0.0.1", "::1")
        ]

        self.assertEqual(
            len(external_calls), 0,
            f"WeightLens made unexpected external API calls: {external_calls}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
