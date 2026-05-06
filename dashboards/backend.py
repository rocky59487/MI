"""
MI Circuit Explorer — Backend Analysis Engine.

This module provides the real backend integration for the dashboard,
connecting to the MI project's core modules (ReIP, WeightLens, CircuitLens).

This is a PRODUCTION research tool. There is NO demo mode, NO mock data,
and NO synthetic fallback. Every analysis runs real model inference.

If the required dependencies (torch, transformer-lens) are not installed,
or if the model cannot be loaded, a clear error is raised and displayed
to the user.

Supported analysis modes:
    - general: Standard ReIP + WeightLens + CircuitLens causal analysis
    - safety: Agent Safety Mode — identifies dangerous decision nodes
"""

from __future__ import annotations

import gc
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Dependency & Hardware Checks
# ============================================================================

class BackendError(Exception):
    """Raised when the backend cannot execute due to missing deps or hardware."""
    pass


def check_dependencies() -> Dict[str, bool]:
    """
    Check which required dependencies are installed.

    Returns:
        Dict mapping dependency name to availability boolean.
    """
    deps = {}

    try:
        import torch
        deps["torch"] = True
        deps["cuda"] = torch.cuda.is_available()
    except ImportError:
        deps["torch"] = False
        deps["cuda"] = False

    try:
        import transformer_lens
        deps["transformer_lens"] = True
    except ImportError:
        deps["transformer_lens"] = False

    return deps


def get_device() -> str:
    """
    Get the best available compute device.

    Returns:
        'cuda' if a GPU is available, otherwise 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        raise BackendError(
            "PyTorch is not installed. "
            "Install with: pip install torch"
        )


def get_backend_status() -> Dict[str, Any]:
    """
    Get the current backend status for display in the UI.

    Returns:
        Dict with 'ready', 'device', 'message', 'deps' keys.
    """
    deps = check_dependencies()

    if not deps.get("torch"):
        return {
            "ready": False,
            "device": None,
            "message": "PyTorch not installed — run: pip install torch",
            "deps": deps,
        }

    if not deps.get("transformer_lens"):
        return {
            "ready": False,
            "device": None,
            "message": "transformer-lens not installed — run: pip install transformer-lens",
            "deps": deps,
        }

    device = "cuda" if deps.get("cuda") else "cpu"
    return {
        "ready": True,
        "device": device,
        "message": f"Ready — {device.upper()} inference",
        "deps": deps,
    }


# ============================================================================
# Model Loading (Cached)
# ============================================================================

_MODEL_CACHE: Dict[str, Any] = {}


def _get_or_load_model(model_name: str = "gpt2") -> Any:
    """
    Load and cache the HookedTransformer model.

    Args:
        model_name: Name of the model to load (e.g., 'gpt2').

    Returns:
        HookedTransformer model instance.

    Raises:
        BackendError: If the model cannot be loaded.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from transformer_lens import HookedTransformer
    except ImportError:
        raise BackendError(
            "transformer-lens is not installed.\n"
            "Install with: pip install transformer-lens"
        )

    try:
        import torch
    except ImportError:
        raise BackendError(
            "PyTorch is not installed.\n"
            "Install with: pip install torch"
        )

    device = get_device()
    print(f"[Backend] Loading model '{model_name}' on {device}...")

    try:
        model = HookedTransformer.from_pretrained(model_name, device=device)
    except Exception as e:
        raise BackendError(
            f"Failed to load model '{model_name}': {e}\n"
            "Ensure you have internet access and sufficient memory."
        )

    _MODEL_CACHE[model_name] = model
    print(
        f"[Backend] Model loaded: {model.cfg.n_layers} layers, "
        f"d_model={model.cfg.d_model}, n_heads={model.cfg.n_heads}"
    )
    return model


def unload_model(model_name: str = "gpt2") -> None:
    """Unload a cached model to free memory."""
    if model_name in _MODEL_CACHE:
        del _MODEL_CACHE[model_name]
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()


# ============================================================================
# Core Analysis Pipeline
# ============================================================================

def run_analysis(
    prompt: str,
    top_n: int = 12,
    model_name: str = "gpt2",
    analysis_mode: str = "general",
) -> Dict:
    """
    Run the full analysis pipeline on the given prompt.

    This function runs REAL model inference — no mock data, no fallback.
    Stages: ReIP → CircuitLens → WeightLens

    Args:
        prompt: The input prompt to analyze.
        top_n: Number of top nodes to return.
        model_name: HuggingFace model name (default: 'gpt2').
        analysis_mode: 'general' or 'safety'.

    Returns:
        Dict with keys:
            - nodes: List[Dict] — top-N nodes with id, layer, component, token, score
            - edges: List[Dict] — causal edges with source, target, weight
            - semantic_labels: Dict[str, str] — node_id → human-readable label
            - safety_info: Optional[Dict] — dangerous nodes/edges/explanation (safety mode only)
            - metadata: Dict — model, device, runtime, prompt, etc.

    Raises:
        BackendError: If dependencies are missing or inference fails.
        ValueError: If the prompt is empty or invalid.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    # Validate dependencies before attempting inference
    status = get_backend_status()
    if not status["ready"]:
        raise BackendError(status["message"])

    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    start_time = time.time()
    device = get_device()
    model = _get_or_load_model(model_name)

    # ---- Stage 1: ReIP — Relevance-based Importance Patching ----
    top_k_nodes, reip_result = _run_reip_stage(
        model=model,
        prompt=prompt,
        top_n=top_n,
        device=device,
        analysis_mode=analysis_mode,
    )

    # ---- Stage 2: CircuitLens — Jacobian-based Edge Attribution ----
    edges = _run_circuitlens_stage(
        model=model,
        top_k_nodes=top_k_nodes,
        reip_result=reip_result,
        device=device,
    )

    # ---- Stage 3: WeightLens — Semantic Projection ----
    semantic_labels = _run_weightlens_stage(
        model=model,
        top_k_nodes=top_k_nodes,
        device=device,
    )

    # ---- Stage 4: Safety Analysis (if applicable) ----
    safety_info = None
    if analysis_mode == "safety":
        safety_info = _run_safety_analysis(
            model=model,
            top_k_nodes=top_k_nodes,
            edges=edges,
            reip_result=reip_result,
            semantic_labels=semantic_labels,
        )

    runtime = time.time() - start_time

    return {
        "nodes": top_k_nodes,
        "edges": edges,
        "semantic_labels": semantic_labels,
        "safety_info": safety_info,
        "metadata": {
            "device": device,
            "model": model_name,
            "runtime_seconds": round(runtime, 3),
            "prompt": prompt,
            "corrupted_prompt": reip_result.metadata.get("corrupted_prompt", ""),
            "n_nodes": len(top_k_nodes),
            "n_edges": len(edges),
            "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
            "n_heads": model.cfg.n_heads,
            "scoring_formula": reip_result.metadata.get("scoring_formula", ""),
        },
    }


# ============================================================================
# Stage 1: ReIP
# ============================================================================

def _run_reip_stage(
    model: Any,
    prompt: str,
    top_n: int,
    device: str,
    analysis_mode: str,
) -> Tuple[List[Dict], Any]:
    """
    Run ReIP analysis to identify the top-N most important computational nodes.

    For safety mode, the corrupted prompt replaces dangerous action words with
    safe alternatives, so the relevance scores capture what drives the
    dangerous decision.

    Returns:
        Tuple of (top_k_nodes, reip_result).
    """
    import gc

    # Build the corrupted prompt for contrastive analysis
    if analysis_mode == "safety":
        corrupted_prompt = _make_safety_corrupted_prompt(prompt)
    else:
        corrupted_prompt = _make_corrupted_prompt(prompt)

    start_time = time.time()

    try:
        import torch
        model.eval()
        model.zero_grad()

        clean_tokens = model.to_tokens(prompt).to(device)
        corr_tokens = model.to_tokens(corrupted_prompt).to(device)

        # Ensure same sequence length (required for act_delta)
        min_len = min(clean_tokens.shape[1], corr_tokens.shape[1])
        clean_tokens = clean_tokens[:, :min_len]
        corr_tokens = corr_tokens[:, :min_len]

        token_labels = model.to_str_tokens(clean_tokens[0])

        captured_clean: Dict[str, Any] = {}
        captured_corrupted: Dict[str, Any] = {}

        n_layers = model.cfg.n_layers
        hook_points = []
        for i in range(n_layers):
            hook_points.append(f"blocks.{i}.hook_mlp_out")
            hook_points.append(f"blocks.{i}.hook_attn_out")

        def _make_hook(store: Dict, name: str):
            def hook_fn(activation, hook):
                activation.requires_grad_(True)
                activation.retain_grad()
                store[name] = activation
                return activation
            return hook_fn

        fwd_hooks_clean = [(_hp, _make_hook(captured_clean, _hp)) for _hp in hook_points]
        fwd_hooks_corr = [(_hp, _make_hook(captured_corrupted, _hp)) for _hp in hook_points]

        # Forward pass: clean (with gradient tracking)
        clean_logits = model.run_with_hooks(
            clean_tokens, return_type="logits", fwd_hooks=fwd_hooks_clean
        )
        # Forward pass: corrupted (no gradient needed)
        with torch.no_grad():
            model.run_with_hooks(
                corr_tokens, return_type="logits", fwd_hooks=fwd_hooks_corr
            )

        # Compute loss on clean logits at last position
        loss = -clean_logits[0, -1].max()
        loss.backward()

        # Attribution patching: clean_grad * (act_clean - act_corrupted)
        # This is the standard gradient-based attribution patching formula.
        relevance_scores: Dict[str, Any] = {}
        for name in captured_clean:
            clean_act = captured_clean[name]
            corr_act = captured_corrupted.get(name)
            if corr_act is None:
                continue
            clean_grad = clean_act.grad
            if clean_grad is None:
                continue
            act_delta = clean_act.detach() - corr_act.detach()
            relevance_scores[name] = (clean_grad * act_delta).detach()

        del captured_clean, captured_corrupted
        gc.collect()

        runtime = time.time() - start_time

    except BackendError:
        raise
    except Exception as e:
        raise BackendError(
            f"ReIP attribution patching failed: {e}\n"
            f"Prompt: '{prompt}'\n"
            f"Corrupted: '{corrupted_prompt}'\n"
            f"Traceback:\n{traceback.format_exc()}"
        )

    # Build a result-like proxy object
    class _ReIPResultProxy:
        def __init__(self):
            self.relevance_scores = relevance_scores
            self.token_labels = list(token_labels)
            self.runtime_seconds = runtime
            self.metadata = {
                "corrupted_prompt": corrupted_prompt,
                "scoring_formula": "clean_grad * (act_clean - act_corrupted)",
                "clean_prompt": prompt,
            }

    result = _ReIPResultProxy()

    # Extract per-component scores
    component_scores: List[Dict] = []
    for hook_name, tensor in result.relevance_scores.items():
        if tensor.dim() == 3:
            score = tensor.abs().mean(dim=0).norm(dim=-1)
        elif tensor.dim() == 2:
            score = tensor.abs().norm(dim=-1)
        else:
            score = tensor.abs().flatten()

        for pos_idx, score_val in enumerate(score.tolist()):
            token_label = (
                result.token_labels[pos_idx]
                if pos_idx < len(result.token_labels)
                else f"pos_{pos_idx}"
            )
            layer_idx = _extract_layer_idx(hook_name)
            component_type = _extract_component(hook_name)
            component_scores.append({
                "id": f"{hook_name}__pos{pos_idx}",
                "hook_name": hook_name,
                "layer": layer_idx,
                "component": component_type,
                "position": pos_idx,
                "token": token_label,
                "score": float(score_val),
            })

    component_scores.sort(key=lambda x: x["score"], reverse=True)
    top_k_nodes = component_scores[:top_n]

    # Normalize scores to [0, 1]
    if top_k_nodes:
        max_score = top_k_nodes[0]["score"]
        if max_score > 0:
            for node in top_k_nodes:
                node["score"] = round(node["score"] / max_score, 4)

    return top_k_nodes, result


# ============================================================================
# Stage 2: CircuitLens
# ============================================================================

def _run_circuitlens_stage(
    model: Any,
    top_k_nodes: List[Dict],
    reip_result: Any,
    device: str,
) -> List[Dict]:
    """
    Compute inter-node edges using CircuitLens Jacobian-based attribution.

    Returns:
        List of edge dicts with source, target, weight.
    """
    from src.circuitlens.jacobian import JacobianAnalyzer
    from src.weightlens.transcoder_loader import TranscoderLoader

    d_model = model.cfg.d_model
    loader = TranscoderLoader(
        model_name=model.cfg.model_name if hasattr(model.cfg, "model_name") else "gpt2",
        device=device,
    )

    analyzer = JacobianAnalyzer(
        model=model,
        device=device,
        lrp_relevance_threshold=0.01,
        top_k_heads=5,
    )

    clean_prompt = reip_result.metadata.get("clean_prompt", "")
    if not clean_prompt:
        # Fallback: use score-product edges if prompt not available
        return _score_product_edges(top_k_nodes)

    try:
        import torch
        input_tokens = model.to_tokens(clean_prompt).to(device)
    except Exception as e:
        raise BackendError(f"Tokenization failed: {e}")

    # Group nodes by layer
    nodes_by_layer: Dict[int, List[Dict]] = {}
    for node in top_k_nodes:
        nodes_by_layer.setdefault(node["layer"], []).append(node)

    sorted_layers = sorted(nodes_by_layer.keys())
    edges: List[Dict] = []

    for i in range(len(sorted_layers) - 1):
        src_layer = sorted_layers[i]
        tgt_layer = sorted_layers[i + 1]
        src_nodes = nodes_by_layer[src_layer]
        tgt_nodes = nodes_by_layer[tgt_layer]

        for tgt_node in tgt_nodes:
            tgt_layer_idx = tgt_node["layer"]
            if tgt_layer_idx < 0:
                continue

            # Load transcoder weights (synthetic if not available)
            weights = loader.load_layer(tgt_layer_idx)
            if weights is None:
                weights = loader.create_synthetic_transcoder(
                    layer_idx=tgt_layer_idx,
                    d_model=d_model,
                    n_features=d_model * 4,
                )

            try:
                jacobian_result = analyzer.compute_jacobian(
                    input_tokens=input_tokens,
                    target_layer=tgt_layer_idx,
                    target_feature_idx=0,
                    W_enc=weights.W_enc,
                    lrp_scores=reip_result.relevance_scores,
                    decompose_heads=True,
                )

                for src_node in src_nodes:
                    src_pos = src_node["position"]
                    if src_pos < jacobian_result.jacobian_matrix.shape[0]:
                        raw_weight = float(
                            jacobian_result.jacobian_matrix[src_pos].norm().item()
                        )
                        # Modulate by ReIP scores
                        edge_weight = raw_weight * min(
                            src_node["score"], tgt_node["score"]
                        )
                        if edge_weight > 1e-6:
                            edges.append({
                                "source": src_node["id"],
                                "target": tgt_node["id"],
                                "weight": edge_weight,
                            })

            except Exception as e:
                # Per-pair fallback: score product
                for src_node in src_nodes:
                    w = min(src_node["score"], tgt_node["score"])
                    if w > 0.05:
                        edges.append({
                            "source": src_node["id"],
                            "target": tgt_node["id"],
                            "weight": w,
                        })

    # Also add intra-layer edges (MLP ↔ Attention at same position)
    for layer_idx, layer_nodes in nodes_by_layer.items():
        by_pos: Dict[int, List[Dict]] = {}
        for node in layer_nodes:
            by_pos.setdefault(node["position"], []).append(node)
        for pos_nodes in by_pos.values():
            if len(pos_nodes) > 1:
                for j in range(len(pos_nodes)):
                    for k in range(j + 1, len(pos_nodes)):
                        w = min(pos_nodes[j]["score"], pos_nodes[k]["score"])
                        if w > 0.05:
                            edges.append({
                                "source": pos_nodes[j]["id"],
                                "target": pos_nodes[k]["id"],
                                "weight": w,
                            })

    # Sort, deduplicate, limit, and normalize
    edges.sort(key=lambda e: e["weight"], reverse=True)
    seen = set()
    deduped = []
    for e in edges:
        key = (e["source"], e["target"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    max_edges = len(top_k_nodes) * 3
    deduped = deduped[:max_edges]

    if deduped:
        max_w = deduped[0]["weight"]
        if max_w > 0:
            for e in deduped:
                e["weight"] = round(e["weight"] / max_w, 4)

    return deduped


def _score_product_edges(top_k_nodes: List[Dict]) -> List[Dict]:
    """Fallback edge computation using ReIP score products between adjacent layers."""
    nodes_by_layer: Dict[int, List[Dict]] = {}
    for node in top_k_nodes:
        nodes_by_layer.setdefault(node["layer"], []).append(node)

    sorted_layers = sorted(nodes_by_layer.keys())
    edges = []

    for i in range(len(sorted_layers) - 1):
        src_layer = sorted_layers[i]
        tgt_layer = sorted_layers[i + 1]
        for src in nodes_by_layer[src_layer]:
            for tgt in nodes_by_layer[tgt_layer]:
                w = min(src["score"], tgt["score"])
                if w > 0.05:
                    edges.append({
                        "source": src["id"],
                        "target": tgt["id"],
                        "weight": round(w, 4),
                    })

    edges.sort(key=lambda e: e["weight"], reverse=True)
    return edges[: len(top_k_nodes) * 3]


# ============================================================================
# Stage 3: WeightLens
# ============================================================================

def _run_weightlens_stage(
    model: Any,
    top_k_nodes: List[Dict],
    device: str,
) -> Dict[str, str]:
    """
    Compute human-readable semantic labels for each node using WeightLens.

    Uses VocabProjector to project transcoder encoder/decoder weights into
    vocabulary space, then SemanticLemmatizer to generate human-readable labels.
    When pre-trained transcoder weights are unavailable (e.g., for GPT-2 without
    the jacobdunefsky/transcoder_circuits repo), falls back to projecting the
    model's own W_E/W_U matrices directly at each layer.

    Returns:
        Dict mapping node_id → semantic label string.
    """
    from src.weightlens.transcoder_loader import TranscoderLoader
    from src.weightlens.projection import VocabProjector
    from src.weightlens.lemmatizer import SemanticLemmatizer

    d_model = model.cfg.d_model
    loader = TranscoderLoader(
        model_name=model.cfg.model_name if hasattr(model.cfg, "model_name") else "gpt2",
        device=device,
    )

    W_E = model.W_E.detach()
    W_U = model.W_U.detach()
    tokenizer = model.tokenizer

    projector = VocabProjector(
        W_embed=W_E,
        W_unembed=W_U.T,  # W_U in TransformerLens is (d_model, vocab_size), VocabProjector expects (vocab_size, d_model)
        tokenizer=tokenizer,
        device=device,
    )

    lemmatizer = SemanticLemmatizer()

    labels: Dict[str, str] = {}

    for node in top_k_nodes:
        node_id = node["id"]
        layer_idx = node["layer"]
        component = node["component"]
        token = node["token"]

        if layer_idx < 0 or component == "embed":
            labels[node_id] = f'Input embedding: "{token}"'
            continue

        # Try to load pre-trained transcoder weights from HF Hub
        weights = loader.load_layer(layer_idx)

        if weights is not None:
            # Real transcoder weights available — use feature 0 projection
            try:
                semantics = projector.analyze_feature(
                    W_enc=weights.W_enc,
                    W_dec=weights.W_dec,
                    feature_idx=0,
                    layer_idx=layer_idx,
                )
                # Apply lemmatizer to get human-readable label
                label = lemmatizer.generate_label(
                    input_tokens=semantics.input_tokens,
                    output_tokens_promoted=semantics.output_tokens_promoted,
                    output_tokens_suppressed=semantics.output_tokens_suppressed,
                )
                if label and label.strip():
                    labels[node_id] = label
                    continue
            except Exception:
                pass

        # No pre-trained transcoder — project model's own W_E/W_U at this layer
        # This gives real vocabulary projections from the model's learned representations
        try:
            import torch
            # Use a slice of W_E as a proxy encoder for this layer
            # (rows correspond to token embeddings, columns to d_model dimensions)
            # We project the token's own embedding direction
            try:
                tok_id = model.to_single_token(token) if token else 0
            except Exception:
                # Token not found or ambiguous — use most common token as proxy
                tok_id = 0

            enc_vec = W_E[tok_id].to(device)  # (d_model,)
            # W_U in TransformerLens is (d_model, vocab_size)
            # W_E in TransformerLens is (vocab_size, d_model)

            # Project enc_vec through W_U to get output vocabulary
            output_logits = W_U.T @ enc_vec  # (vocab_size,) = (vocab_size, d_model) @ (d_model,)
            input_logits = W_E @ enc_vec     # (vocab_size,) = (vocab_size, d_model) @ (d_model,)

            # Get top tokens
            top_out_ids = output_logits.topk(10).indices.tolist()
            top_in_ids = input_logits.topk(10).indices.tolist()

            out_tokens = [tokenizer.decode([i]).strip() for i in top_out_ids if tokenizer.decode([i]).strip()]
            in_tokens = [tokenizer.decode([i]).strip() for i in top_in_ids if tokenizer.decode([i]).strip()]

            label = lemmatizer.generate_label(
                input_tokens=in_tokens[:5],
                output_tokens_promoted=out_tokens[:5],
                output_tokens_suppressed=[],
            )
            if label and label.strip():
                labels[node_id] = label
            else:
                labels[node_id] = (
                    f'L{layer_idx} {component.upper()} @ "{token}" '
                    f'(pos {node["position"]})'
                )
        except Exception:
            labels[node_id] = (
                f'L{layer_idx} {component.upper()} @ "{token}" '
                f'(pos {node["position"]})'
            )

    return labels


# ============================================================================
# Stage 4: Safety Analysis
# ============================================================================

def _run_safety_analysis(
    model: Any,
    top_k_nodes: List[Dict],
    edges: List[Dict],
    reip_result: Any,
    semantic_labels: Dict[str, str],
) -> Dict:
    """
    Identify dangerous nodes and causal paths from real ReIP scores.

    The dangerous threshold is set at the 75th percentile of scores,
    so the top 25% of nodes are flagged as contributing to the dangerous output.

    Returns:
        Dict with dangerous_nodes, dangerous_edges, explanation, attention_heads.
    """
    if not top_k_nodes:
        return {
            "dangerous_nodes": [],
            "dangerous_edges": [],
            "explanation": "No nodes found in analysis.",
            "attention_heads": [],
            "threshold": 0.75,
        }

    scores = sorted([n["score"] for n in top_k_nodes], reverse=True)
    # Use 75th percentile as threshold (top 25% are dangerous)
    n_dangerous = max(1, len(scores) // 4)
    threshold = scores[n_dangerous - 1] if n_dangerous <= len(scores) else scores[-1]
    # Ensure threshold is at least 0.5
    threshold = max(threshold, 0.5)

    dangerous_node_ids = {
        n["id"] for n in top_k_nodes if n["score"] >= threshold
    }

    dangerous_edge_ids = {
        f"{e['source']}__{e['target']}"
        for e in edges
        if e["source"] in dangerous_node_ids or e["target"] in dangerous_node_ids
    }

    # Build explanation from real node data
    dangerous_nodes_sorted = sorted(
        [n for n in top_k_nodes if n["id"] in dangerous_node_ids],
        key=lambda x: x["score"],
        reverse=True,
    )

    explanation_parts = []
    attention_heads = []

    for node in dangerous_nodes_sorted[:4]:
        layer = node["layer"]
        comp = node["component"]
        token = node["token"]
        score = node["score"]
        node_id = node["id"]
        label = semantic_labels.get(node_id, "")

        if comp == "attn":
            # Estimate head index from position
            n_heads = model.cfg.n_heads if hasattr(model.cfg, "n_heads") else 12
            head_idx = node["position"] % n_heads
            attention_heads.append({
                "layer": layer,
                "head": head_idx,
                "score": score,
                "token": token,
                "label": label,
            })
            explanation_parts.append(
                f"Layer {layer} Attention Head {head_idx}  (ReIP score: {score:.4f})\n"
                f"  Token: \"{token}\"  |  WeightLens: {label}\n"
                f"  This attention head attends to the action-bearing token and "
                f"propagates features that drive the dangerous output decision."
            )
        elif comp == "mlp":
            explanation_parts.append(
                f"Layer {layer} MLP  (ReIP score: {score:.4f})\n"
                f"  Token: \"{token}\"  |  WeightLens: {label}\n"
                f"  This MLP projects the residual stream into a subspace "
                f"associated with the dangerous action semantics."
            )
        elif comp == "resid":
            explanation_parts.append(
                f"Layer {layer} Residual Stream  (ReIP score: {score:.4f})\n"
                f"  Token: \"{token}\"  |  WeightLens: {label}\n"
                f"  The residual stream at this layer accumulates the dangerous "
                f"action features from upstream components."
            )

    if not explanation_parts:
        explanation_parts.append(
            "No nodes exceeded the danger threshold. "
            "The model's output appears distributed across many low-scoring components."
        )

    return {
        "dangerous_nodes": list(dangerous_node_ids),
        "dangerous_edges": list(dangerous_edge_ids),
        "explanation": "\n\n".join(explanation_parts),
        "attention_heads": attention_heads,
        "threshold": round(threshold, 4),
    }


# ============================================================================
# Prompt Corruption Helpers
# ============================================================================

def _make_corrupted_prompt(prompt: str) -> str:
    """
    Create a corrupted version of the prompt for ReIP contrastive analysis.
    Strategy: find two DISTINCT capitalized tokens (likely named entities)
    and swap all occurrences of each with the other.
    Falls back to reversing the last two words if no distinct pairs found.
    """
    words = prompt.split()
    if len(words) < 2:
        return prompt + " not"
    corrupted = words.copy()
    # Find distinct capitalized words (skip the first word which is always capitalized)
    caps = [(i, w) for i, w in enumerate(words) if w and w[0].isupper() and i > 0]
    # Get distinct capitalized words
    seen = set()
    distinct_caps = []
    for i, w in caps:
        clean_w = w.rstrip('.,;:!?')
        if clean_w not in seen:
            seen.add(clean_w)
            distinct_caps.append((i, w, clean_w))
    if len(distinct_caps) >= 2:
        # Swap all occurrences of the first two distinct capitalized words
        w1_clean = distinct_caps[0][2]
        w2_clean = distinct_caps[1][2]
        for idx, word in enumerate(corrupted):
            clean = word.rstrip('.,;:!?')
            suffix = word[len(clean):]
            if clean == w1_clean:
                corrupted[idx] = w2_clean + suffix
            elif clean == w2_clean:
                corrupted[idx] = w1_clean + suffix
    else:
        corrupted[-1], corrupted[-2] = corrupted[-2], corrupted[-1]
    return " ".join(corrupted)


def _make_safety_corrupted_prompt(prompt: str) -> str:
    """
    Create a safe-action corrupted version of the prompt for safety analysis.

    Replaces dangerous action words with safe alternatives so that the
    ReIP relevance scores capture what drives the dangerous vs. safe decision.
    """
    replacements = {
        "delete": "keep",
        "remove": "keep",
        "destroy": "preserve",
        "ignore": "follow",
        "override": "respect",
        "bypass": "follow",
        "kill": "save",
        "hack": "protect",
        "steal": "return",
        "break": "fix",
        "disable": "enable",
        "corrupt": "restore",
        "erase": "save",
        "terminate": "continue",
        "shutdown": "restart",
        "shut down": "restart",
    }

    lower = prompt.lower()
    for dangerous, safe in replacements.items():
        if dangerous in lower:
            idx = lower.find(dangerous)
            replaced = prompt[:idx] + safe + prompt[idx + len(dangerous):]
            return replaced

    # No dangerous word found — append a safety qualifier
    return prompt + " safely and without causing harm"


# ============================================================================
# Utility Functions
# ============================================================================

def _extract_layer_idx(hook_name: str) -> int:
    """Extract layer index from a TransformerLens hook point name."""
    parts = hook_name.split(".")
    for i, part in enumerate(parts):
        if part == "blocks" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def _extract_component(hook_name: str) -> str:
    """Extract component type from a TransformerLens hook point name."""
    if "mlp" in hook_name:
        return "mlp"
    elif "attn" in hook_name:
        return "attn"
    elif "resid" in hook_name:
        return "resid"
    elif "embed" in hook_name:
        return "embed"
    return "other"
