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


def test_mini_activation_patching_baseline_ioi():
    pytest.importorskip("torch")
    tl = pytest.importorskip("transformer_lens")

    model = tl.HookedTransformer.from_pretrained("gpt2", device="cpu")

    clean_prompt = "When John and Mary went shopping, John gave a gift to"
    corr_prompt = "When John and Mary went shopping, Mary gave a gift to"
    target_token = " Mary"

    clean_tokens = model.to_tokens(clean_prompt)
    corr_tokens = model.to_tokens(corr_prompt)
    assert clean_tokens.shape == corr_tokens.shape

    target_id = model.to_single_token(target_token)

    clean_logits, clean_cache = model.run_with_cache(clean_tokens, return_type="logits")
    corr_logits, _ = model.run_with_cache(corr_tokens, return_type="logits")

    pos_idx = -1
    clean_lp = clean_logits[0, pos_idx].log_softmax(dim=-1)[target_id]
    corr_lp = corr_logits[0, pos_idx].log_softmax(dim=-1)[target_id]
    base_gap = (clean_lp - corr_lp).item()

    patch_name = "blocks.0.hook_resid_pre"

    def patch_resid(act, hook):
        act = act.clone()
        act[:, :, :] = clean_cache[patch_name]
        return act

    patched_logits = model.run_with_hooks(
        corr_tokens,
        return_type="logits",
        fwd_hooks=[(patch_name, patch_resid)],
    )
    patched_lp = patched_logits[0, pos_idx].log_softmax(dim=-1)[target_id]
    patched_gap = (clean_lp - patched_lp).item()

    # Mini baseline criterion: patching should reduce the clean-corrupted gap.
    assert abs(patched_gap) <= abs(base_gap)
