"""
ReIP End-to-End Execution Pipeline.

This module orchestrates the complete ReIP analysis workflow:
    1. run_with_cache: Two forward passes (clean + corrupted inputs)
    2. LRP backward propagation via registered hooks
    3. Relevance score extraction from activation cache
    4. Topology pruning to produce sparse causal circuit graph

Computational complexity: O(2F + B) — two forward passes + one backward pass.
Target hardware: NVIDIA RTX 4090 (24GB VRAM).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .backward_hooks import ReIPHookManager
from .pruning import TopologyPruner


@dataclass
class ReIPConfig:
    """
    Configuration dataclass for the ReIP pipeline.

    Attributes:
        model_name: Name of the target model (used to infer default LRP rules).
        lrp_rules: List of LRP rule names to apply. If None, inferred from model_name.
        pruning_threshold: Minimum relevance score to retain a node/edge.
        pruning_top_k: If set, retain only top-k edges regardless of threshold.
        normalize_scores: Normalize relevance scores to [0, 1] before pruning.
        target_token_idx: Index of the target output token for loss computation.
                          If -1, uses the last token position.
        device: Torch device string ("cuda", "cpu", "cuda:0", etc.).
        verbose: Print progress and debug information.
        reset_hooks_end: Auto-cleanup hooks after each analysis run.
    """
    model_name: str = "gpt2"
    lrp_rules: Optional[List[str]] = None
    pruning_threshold: float = 0.01
    pruning_top_k: Optional[int] = 500
    normalize_scores: bool = True
    target_token_idx: int = -1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = False
    reset_hooks_end: bool = True


@dataclass
class ReIPResult:
    """
    Container for ReIP pipeline output.

    Attributes:
        topology_graph: Sparse causal circuit topology (NetworkX DiGraph or dict).
        relevance_scores: Raw LRP relevance scores per hook point.
        clean_logits: Model logits for the clean input.
        corrupted_logits: Model logits for the corrupted input.
        token_labels: String labels for each token position.
        runtime_seconds: Total pipeline execution time.
        metadata: Additional metadata (model name, config, etc.).
    """
    topology_graph: Any
    relevance_scores: Dict[str, torch.Tensor]
    clean_logits: torch.Tensor
    corrupted_logits: torch.Tensor
    token_labels: List[str]
    runtime_seconds: float
    metadata: Dict = field(default_factory=dict)


class ReIPPipeline:
    """
    End-to-end ReIP (Relevance Patching) analysis pipeline.

    Integrates ReIPHookManager and TopologyPruner to provide a single-call
    interface for causal circuit discovery in TransformerLens models.

    Example::

        from transformer_lens import HookedTransformer
        from src.reip import ReIPPipeline, ReIPConfig

        model = HookedTransformer.from_pretrained("gpt2")
        config = ReIPConfig(model_name="gpt2", pruning_threshold=0.02)
        pipeline = ReIPPipeline(model, config)

        result = pipeline.run(
            clean_prompt="When Mary and John went to the store, John gave a drink to",
            corrupted_prompt="When Mary and John went to the store, Mary gave a drink to",
            target_token=" Mary",
        )
        print(f"Nodes in circuit: {len(result.topology_graph.nodes)}")
    """

    def __init__(
        self,
        model: Any,  # transformer_lens.HookedTransformer
        config: Optional[ReIPConfig] = None,
    ):
        """
        Args:
            model: A TransformerLens HookedTransformer instance.
            config: ReIPConfig instance. Uses defaults if None.
        """
        self.model = model
        self.config = config or ReIPConfig()
        self.hook_manager = ReIPHookManager(
            model=model,
            rules=self.config.lrp_rules,
            model_name=self.config.model_name,
            verbose=self.config.verbose,
        )
        self.pruner = TopologyPruner(
            threshold=self.config.pruning_threshold,
            top_k=self.config.pruning_top_k,
            normalize=self.config.normalize_scores,
        )

    def run(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        target_token: Optional[str] = None,
        target_token_id: Optional[int] = None,
    ) -> ReIPResult:
        """
        Execute the full ReIP analysis pipeline.

        Args:
            clean_prompt: The clean input text string.
            corrupted_prompt: The corrupted (patched) input text string.
            target_token: Target token string (e.g., " Mary"). Used to compute loss.
            target_token_id: Target token ID. If both target_token and
                             target_token_id are provided, target_token_id takes precedence.

        Returns:
            ReIPResult containing the sparse topology graph and relevance scores.
        """
        start_time = time.time()
        device = self.config.device

        # ------------------------------------------------------------------
        # Step 1: Tokenize inputs
        # ------------------------------------------------------------------
        clean_tokens = self.model.to_tokens(clean_prompt).to(device)
        corrupted_tokens = self.model.to_tokens(corrupted_prompt).to(device)
        if clean_tokens.shape != corrupted_tokens.shape:
            raise ValueError(
                "clean_prompt and corrupted_prompt must produce identical token shapes. "
                f"Got {tuple(clean_tokens.shape)} vs {tuple(corrupted_tokens.shape)}."
            )

        # Resolve target token ID
        if target_token_id is None and target_token is not None:
            target_token_id = self.model.to_single_token(target_token)

        token_labels = self.model.to_str_tokens(clean_tokens[0])

        if self.config.verbose:
            print(f"[ReIPPipeline] Clean tokens: {token_labels}")
            print(f"[ReIPPipeline] Target token ID: {target_token_id}")

        # ------------------------------------------------------------------
        # Step 2: Forward pass — clean input (with activation caching)
        # ------------------------------------------------------------------
        with torch.no_grad():
            corrupted_logits, _ = self.model.run_with_cache(
                corrupted_tokens,
                return_type="logits",
                names_filter=lambda name: "hook_" in name,
            )

        # Enable gradients for clean forward pass (needed for backward)
        self.model.zero_grad()
        clean_logits, clean_cache = self.model.run_with_cache(
            clean_tokens,
            return_type="logits",
            names_filter=lambda name: "hook_" in name,
            prepend_bos=False,
        )

        if self.config.verbose:
            print(f"[ReIPPipeline] Forward passes complete. "
                  f"Clean logits shape: {clean_logits.shape}")

        # ------------------------------------------------------------------
        # Step 3: Compute loss and run LRP backward pass
        # ------------------------------------------------------------------
        relevance_scores: Dict[str, torch.Tensor] = {}

        with self.hook_manager.active():
            # Register gradient hooks on cached activations to capture relevance
            for hook_name, activation in clean_cache.items():
                if activation.requires_grad:
                    def make_capture_hook(name):
                        def capture_hook(grad):
                            relevance_scores[name] = grad.detach().clone()
                        return capture_hook
                    activation.register_hook(make_capture_hook(hook_name))

            # Compute scalar objective.
            # ReIP needs a causal contrastive signal, so we optimize the clean-vs-
            # corrupted gap rather than clean logits alone.
            pos_idx = self.config.target_token_idx
            if target_token_id is not None:
                clean_log_probs = torch.log_softmax(clean_logits[0, pos_idx], dim=-1)
                corrupted_log_probs = torch.log_softmax(
                    corrupted_logits[0, pos_idx], dim=-1
                )
                contrastive_score = (
                    clean_log_probs[target_token_id]
                    - corrupted_log_probs[target_token_id]
                )
                loss = -contrastive_score
            else:
                # Fallback: maximize the most probable token
                loss = -clean_logits[0, pos_idx].max()

            loss.backward()

        if self.config.verbose:
            print(f"[ReIPPipeline] Backward pass complete. "
                  f"Captured {len(relevance_scores)} relevance tensors.")

        # ------------------------------------------------------------------
        # Step 4: Build sparse topology graph
        # ------------------------------------------------------------------
        topology_graph = self.pruner.build_graph(
            relevance_scores=relevance_scores,
            token_labels=list(token_labels),
        )

        # ------------------------------------------------------------------
        # Step 5: Cleanup
        # ------------------------------------------------------------------
        if self.config.reset_hooks_end:
            self.model.reset_hooks(including_permanent=False)

        # Free intermediate tensors
        del clean_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        runtime = time.time() - start_time

        if self.config.verbose:
            print(f"[ReIPPipeline] Analysis complete in {runtime:.3f}s")

        return ReIPResult(
            topology_graph=topology_graph,
            relevance_scores=relevance_scores,
            clean_logits=clean_logits.detach(),
            corrupted_logits=corrupted_logits.detach(),
            token_labels=list(token_labels),
            runtime_seconds=runtime,
            metadata={
                "model_name": self.config.model_name,
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "target_token_id": target_token_id,
                "objective": (
                    "negative_clean_corrupted_logprob_gap"
                    if target_token_id is not None
                    else "negative_clean_max_logit"
                ),
                "lrp_rules": self.hook_manager.rules,
                "pruning_threshold": self.config.pruning_threshold,
            },
        )

    def run_batch(
        self,
        prompt_pairs: List[Tuple[str, str]],
        target_tokens: Optional[List[str]] = None,
    ) -> List[ReIPResult]:
        """
        Run ReIP analysis on multiple prompt pairs sequentially.

        Args:
            prompt_pairs: List of (clean_prompt, corrupted_prompt) tuples.
            target_tokens: Optional list of target token strings, one per pair.

        Returns:
            List of ReIPResult instances.
        """
        results = []
        for i, (clean, corrupted) in enumerate(prompt_pairs):
            target = target_tokens[i] if target_tokens else None
            result = self.run(clean, corrupted, target_token=target)
            results.append(result)
            if self.config.verbose:
                print(f"[ReIPPipeline] Batch progress: {i+1}/{len(prompt_pairs)}")
        return results
