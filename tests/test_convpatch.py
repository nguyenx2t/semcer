"""The CPU shift-and-sum conv matches ``F.conv1d`` + SiLU within 1e-5."""

from __future__ import annotations

import torch

from semcer import _convpatch


def test_shift_sum_matches_conv1d_with_silu():
    g = torch.Generator().manual_seed(1)
    for channels, seq in ((8, 1), (64, 5), (96, 130)):
        x = torch.randn(3, channels, seq, generator=g)
        w = torch.randn(channels, 4, generator=g)
        b = torch.randn(channels, generator=g)
        for act in (None, "silu"):
            got = _convpatch.shift_sum_conv1d(x, w, b, act)
            ref = _convpatch._reference_conv1d(x, w, b, act)
            assert (got - ref).abs().max() <= 1e-5


def test_self_test_and_idempotent_install():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    assert _convpatch.self_test() <= 1e-5
    _convpatch.install()
    fn = mq.causal_conv1d_fn
    assert getattr(fn, "__semcer_patch__", False)
    _convpatch.install()
    assert mq.causal_conv1d_fn is fn
    assert _convpatch.conv_path("cpu") == "shift_sum"


def test_uninstall_restores_the_original():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    _convpatch.install()
    _convpatch.uninstall()
    try:
        assert not getattr(mq.causal_conv1d_fn, "__semcer_patch__", False)
        assert not getattr(mq.torch_chunk_gated_delta_rule, "__semcer_patch__", False)
        assert _convpatch.conv_path("cpu") == "torch_conv1d"
        assert _convpatch.gdn_path("cpu") == "torch"
    finally:
        _convpatch.install()
