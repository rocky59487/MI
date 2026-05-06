from __future__ import annotations

import torch

from src.reip.lrp_rules import LNRule, IdentityRule, HalfRule, ZeroRule


def test_identity_rule_backward_passthrough():
    x = torch.randn(4, 8, requires_grad=True)
    rule = IdentityRule(torch.nn.functional.gelu)
    y = rule(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-6)


def test_half_rule_splits_gradient_evenly():
    gate = torch.randn(2, 3, requires_grad=True)
    linear = torch.randn(2, 3, requires_grad=True)
    rule = HalfRule()
    out = rule(gate, linear)
    out.sum().backward()

    assert gate.grad is not None and linear.grad is not None
    assert torch.allclose(gate.grad, 0.5 * linear.detach(), atol=1e-6)
    assert torch.allclose(linear.grad, 0.5 * gate.detach(), atol=1e-6)


def test_zero_rule_matches_linear_forward_shape():
    x = torch.randn(5, 7, requires_grad=True)
    weight = torch.randn(11, 7, requires_grad=True)
    bias = torch.randn(11, requires_grad=True)

    rule = ZeroRule(weight=weight, bias=bias)
    out = rule(x)
    assert out.shape == (5, 11)


def test_ln_rule_output_shape():
    ln = torch.nn.LayerNorm(16)
    rule = LNRule(ln, is_rms=False)
    x = torch.randn(3, 4, 16, requires_grad=True)
    out = rule(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None
