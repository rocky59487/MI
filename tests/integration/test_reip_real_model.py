"""Integration test for ReIP on a real TransformerLens model.

This test is optional and is skipped when heavy dependencies are unavailable.
"""

import pytest


def test_reip_real_model_smoke():
    pytest.importorskip("torch")
    tl = pytest.importorskip("transformer_lens")

    from src.reip.pipeline import ReIPConfig, ReIPPipeline

    model = tl.HookedTransformer.from_pretrained("gpt2", device="cpu")
    config = ReIPConfig(
        model_name="gpt2",
        device="cpu",
        target_token_idx=-1,
        strict_hooks=True,
        pruning_top_k=50,
    )
    pipeline = ReIPPipeline(model, config)

    result = pipeline.run(
        clean_prompt="When John and Mary went shopping, John gave a gift to",
        corrupted_prompt="When John and Mary went shopping, Mary gave a gift to",
        target_token=" Mary",
    )

    assert isinstance(result.relevance_scores, dict)
    assert len(result.relevance_scores) > 0
    assert result.metadata["objective"] == "negative_logprob_gap_with_activation_delta"
    assert result.clean_logits.shape == result.corrupted_logits.shape
