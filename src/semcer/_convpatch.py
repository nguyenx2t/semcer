"""Kernels for the Qwen3.5 Gated DeltaNet (GDN) layers.

:func:`install` replaces two transformers torch fallbacks in the Qwen3.5 module.
The convolution below runs on CPU tensors only (a GPU keeps ``F.conv1d``). The
delta rule below runs on every device, except on a GPU with the ``fla`` package
installed, whose fused kernel is kept. Installing is idempotent and
process-wide, and self-tests each kernel against its fallback first.

1. :func:`shift_sum_conv1d`, the depthwise causal convolution: with only 4 taps
   a shift-and-sum is much faster than ``F.conv1d(groups=C)`` on CPU (different
   fp32 summation order, <= 1e-5 against the fallback).
2. :func:`chunk_gated_delta_rule`, the chunked delta rule: the same maths with
   one ``solve_triangular`` for each chunk's ``(I - A)^-1`` instead of a
   row-by-row loop, and a sequence shorter than the chunk size run as one
   exact-size chunk instead of padded to 64 (relative difference ~1e-7).
"""

from __future__ import annotations

import contextlib
import importlib.util
import threading

import torch
import torch.nn.functional as F

_TOL = 1e-5
_HAS_FLA = importlib.util.find_spec("fla") is not None
_installed: dict[str, object] = {}  # "original" / "gdn_original": the replaced fallbacks
_discard = threading.local()


@contextlib.contextmanager
def discard_final_state():
    """Within this block the patched delta rule returns no final recurrent state
    (``None``) and skips computing it. For forwards whose cache is thrown away (the
    engine's suffix batches on a read-only fork); outputs are unchanged."""
    prev = getattr(_discard, "on", False)
    _discard.on = True
    try:
        yield
    finally:
        _discard.on = prev


def shift_sum_conv1d(hidden_states, weight, bias=None, activation=None):
    """Depthwise causal conv1d over ``(B, C, L)`` with ``(C, K)`` taps, as a shift-and-sum."""
    from transformers.activations import ACT2FN

    seq_len = hidden_states.shape[-1]
    k = weight.shape[-1]
    x = F.pad(hidden_states.to(weight.dtype), (k - 1, 0))
    out = x[:, :, 0:seq_len] * weight[:, 0, None]
    for j in range(1, k):
        out = out + x[:, :, j : j + seq_len] * weight[:, j, None]
    if bias is not None:
        out = out + bias[None, :, None]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def _reference_conv1d(hidden_states, weight, bias=None, activation=None):
    """The transformers torch fallback, reproduced for the self-test."""
    from transformers.activations import ACT2FN

    seq_len = hidden_states.shape[-1]
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=weight.shape[-1] - 1,
        groups=hidden_states.shape[1],
    )[:, :, :seq_len]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def self_test(channels: int = 64, taps: int = 4, seq_len: int = 37, seed: int = 0) -> float:
    """Max abs difference between the shift-and-sum and ``F.conv1d`` + SiLU on one
    random input; raises ``RuntimeError`` above the 1e-5 tolerance."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, channels, seq_len, generator=g)
    w = torch.randn(channels, taps, generator=g)
    diff = float((shift_sum_conv1d(x, w, None, "silu") - _reference_conv1d(x, w, None, "silu")).abs().max())
    if not diff <= _TOL:
        raise RuntimeError(f"shift-and-sum conv self-test failed: max abs diff {diff:.3g} > {_TOL}")
    return diff


def chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    """Drop-in for transformers' ``torch_chunk_gated_delta_rule`` (see module doc)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    b, h, length, dk = key.shape
    dv = value.shape[-1]
    c = min(chunk_size, length)
    pad = (c - length % c) % c
    if pad:
        query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
        beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    query = query * (1 / dk**0.5)
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = (
        x.reshape(b, h, -1, c, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(b, h, -1, c).cumsum(dim=-1)
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    upper = torch.triu(torch.ones(c, c, dtype=torch.bool, device=query.device), diagonal=0)
    a = -((k_beta @ key.transpose(-1, -2)) * decay).masked_fill(upper, 0)  # strictly lower
    eye = torch.eye(c, dtype=a.dtype, device=a.device)
    t = torch.linalg.solve_triangular(eye - a, eye.expand_as(a), upper=False, unitriangular=True)
    value = t @ v_beta
    k_cumdecay = t @ (k_beta * g.exp().unsqueeze(-1))
    state = (
        torch.zeros(b, h, dk, dv, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    out = torch.empty_like(value)
    strict = torch.triu(torch.ones(c, c, dtype=torch.bool, device=query.device), diagonal=1)
    n_chunks = value.shape[2]
    for i in range(n_chunks):
        q_i, k_i, v_i, g_i = query[:, :, i], key[:, :, i], value[:, :, i], g[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay[:, :, i]).masked_fill_(strict, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ state
        out[:, :, i] = (q_i * g_i[..., None].exp()) @ state + attn @ v_new
        if i == n_chunks - 1 and not output_final_state:
            break  # the final state is not wanted: skip its update
        state = (
            state * g_i[:, :, -1, None, None].exp()
            + (k_i * (g_i[:, :, -1, None] - g_i).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    out = out.reshape(b, h, -1, dv)[:, :, :length].transpose(1, 2).contiguous().to(initial_dtype)
    return out, (state if output_final_state else None)


def gdn_self_test(seed: int = 0) -> float:
    """Max relative difference, output and final state, between :func:`chunk_gated_delta_rule`
    and the transformers fallback on random inputs (a short suffix batch with an initial
    state, and a longer multi-chunk row); raises above 1e-5."""
    fallback = _installed.get("gdn_original") or _qwen35_module().torch_chunk_gated_delta_rule
    gen = torch.Generator().manual_seed(seed)
    worst = 0.0
    for bsz, length, init in ((3, 29, True), (1, 130, False)):
        q, k, v = (torch.randn(bsz, length, 4, 16, generator=gen) for _ in range(3))
        g = -torch.rand(bsz, length, 4, generator=gen) * 2
        beta = torch.rand(bsz, length, 4, generator=gen)
        s0 = torch.randn(bsz, 4, 16, 16, generator=gen) * 0.1 if init else None
        kw = dict(initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True)
        a, sa = fallback(q, k, v, g, beta, **kw)
        x, sx = chunk_gated_delta_rule(q, k, v, g, beta, **kw)
        worst = max(
            worst,
            float((a - x).abs().max() / a.abs().max()),
            float((sa - sx).abs().max() / sa.abs().max()),
        )
    if not worst <= _TOL:
        raise RuntimeError(f"GDN chunk-rule self-test failed: relative diff {worst:.3g} > {_TOL}")
    return worst


def _qwen35_module():
    from transformers.models.qwen3_5 import modeling_qwen3_5

    return modeling_qwen3_5


def uses_gated_deltanet(model) -> bool:
    """True when ``model`` holds at least one Qwen3.5 Gated DeltaNet layer."""
    gdn = _qwen35_module().Qwen3_5GatedDeltaNet
    return any(isinstance(m, gdn) for m in model.modules())


def install() -> None:
    """Install both CPU kernels on the Qwen3.5 module (idempotent)."""
    if "original" in _installed:
        return
    mq = _qwen35_module()
    self_test()
    original = mq.causal_conv1d_fn
    gdn_original = mq.torch_chunk_gated_delta_rule

    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        if hidden_states.device.type == "cpu":
            return shift_sum_conv1d(hidden_states, weight, bias, activation)
        return original(hidden_states, weight, bias=bias, activation=activation, **kwargs)

    def torch_chunk_gated_delta_rule(query, *args, **kwargs):
        if query.device.type != "cpu" and _HAS_FLA:
            return gdn_original(query, *args, **kwargs)  # the fused kernel
        if getattr(_discard, "on", False):
            kwargs["output_final_state"] = False
        return chunk_gated_delta_rule(query, *args, **kwargs)

    causal_conv1d_fn.__semcer_patch__ = True  # type: ignore[attr-defined]
    torch_chunk_gated_delta_rule.__semcer_patch__ = True  # type: ignore[attr-defined]
    _installed["original"] = original
    mq.causal_conv1d_fn = causal_conv1d_fn
    _installed["gdn_original"] = gdn_original
    gdn_self_test()
    mq.torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule


def uninstall() -> None:
    """Restore transformers' fallbacks (tests use this for the reference path)."""
    mq = _qwen35_module()
    if "original" in _installed:
        mq.causal_conv1d_fn = _installed.pop("original")
    if "gdn_original" in _installed:
        mq.torch_chunk_gated_delta_rule = _installed.pop("gdn_original")


def gdn_path(device: str) -> str:
    """The chunked delta-rule implementation a GDN layer runs on ``device``, for the signature."""
    if not str(device).startswith("cpu") and _HAS_FLA:
        return "fla"
    return "chunk_solve" if "gdn_original" in _installed else "torch"


def conv_path(device: str) -> str:
    """The conv implementation a GDN layer runs on ``device``, for the signature."""
    if str(device).startswith("cpu"):
        return "shift_sum" if "original" in _installed else "torch_conv1d"
    return "causal_conv1d" if importlib.util.find_spec("causal_conv1d") else "torch_conv1d"


__all__ = [
    "chunk_gated_delta_rule",
    "conv_path",
    "discard_final_state",
    "gdn_path",
    "gdn_self_test",
    "install",
    "self_test",
    "shift_sum_conv1d",
    "uninstall",
    "uses_gated_deltanet",
]
