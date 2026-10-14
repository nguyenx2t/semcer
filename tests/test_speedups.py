"""Exact speed-ups: shared-prefix tokenization, the operation-header cache level,
the CPU GDN chunk-rule kernel and the CPU batch default. Each must leave every
score unchanged (tokens exactly; severities within the 1e-6 fp32 tolerance)."""

from __future__ import annotations

import pytest
import torch
from _pairs import dense_pair, make_pairs

from semcer import SemCER, _convpatch
from semcer._prep import PairTokenizer, prepare_pair, tokenize_edits_full

TOL = 1e-6


def _key(te):
    return (te.index, te.ids, te.ref_span, te.hyp_span, te.status)


def test_shared_prefix_tokenization_equals_full(scorer):
    tok = scorer._tokenizer
    pairs = [(p["reference"], p["hypothesis"]) for p in make_pairs(48)]
    pairs += [dense_pair(n) for n in (100, 600)]
    pairs += [("Total: 15!", "Total 75!!"), ("a", "a."), ("x\n", "y\n\n"), ("abc", "")]
    n = 0
    for ref, hyp in pairs:
        prep = prepare_pair(ref, hyp)
        idx = range(len(prep.edits))
        for ml in (2048, 300, 64):
            pt = PairTokenizer(prep, tok, ml)
            got = pt(idx)
            assert [_key(t) for t in got] == [_key(t) for t in tokenize_edits_full(prep, tok, ml, idx)]
            if prep.edits:
                assert pt.shared is True
            n += len(prep.edits)
    assert n > 3_000


def test_streamed_chunks_equal_one_call(scorer):
    prep = prepare_pair(*dense_pair(300))
    pt = PairTokenizer(prep, scorer._tokenizer, 2048)
    chunks = [t for s in range(0, len(prep.edits), 7) for t in pt(range(s, min(len(prep.edits), s + 7)))]
    whole = tokenize_edits_full(prep, scorer._tokenizer, 2048, range(len(prep.edits)))
    assert [_key(t) for t in chunks] == [_key(t) for t in whole]


def test_unverified_split_falls_back_to_full_tokenization(tiny_bundle, scorer, monkeypatch, pairs):
    sub = [p for p in pairs if p["reference"] != p["hypothesis"]][:12]
    want = scorer.score_batch([p["reference"] for p in sub], [p["hypothesis"] for p in sub])
    monkeypatch.setattr(PairTokenizer, "_verify", lambda self, text, ids: False)
    s = SemCER.from_pretrained(tiny_bundle)
    got = s.score_batch([p["reference"] for p in sub], [p["hypothesis"] for p in sub])
    for a, b in zip(want, got):
        for x, y in zip(a.edits, b.edits):
            assert abs(x.severity - y.severity) <= TOL


def test_op_header_level_is_used_and_exact(tiny_bundle, pairs):
    sub = [p for p in pairs if p["reference"] != p["hypothesis"]][:30]
    on = SemCER.from_pretrained(tiny_bundle)
    off = SemCER.from_pretrained(tiny_bundle, cache_op_header=False)
    a = on.score_batch([p["reference"] for p in sub], [p["hypothesis"] for p in sub])
    b = off.score_batch([p["reference"] for p in sub], [p["hypothesis"] for p in sub])
    assert on.stats.header_forwards > 0 and on.stats.header_rows > 0
    assert off.stats.header_forwards == 0
    assert on.stats.fallback_rows == 0  # shared-prefix split: no edit needs the uncached path
    for x, y in zip(a, b):
        for e, f in zip(x.edits, y.edits):
            assert abs(e.severity - f.severity) <= TOL


@pytest.mark.parametrize("bsz,length,init", [(1, 200, False), (8, 36, True), (4, 64, True), (2, 1, True)])
def test_gdn_chunk_rule_matches_fallback(bsz, length, init):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    fallback = _convpatch._installed.get("gdn_original") or mq.torch_chunk_gated_delta_rule
    g = torch.Generator().manual_seed(length)
    q, k, v = (torch.randn(bsz, length, 4, 32, generator=g) for _ in range(3))
    gate = -torch.rand(bsz, length, 4, generator=g) * 2
    beta = torch.rand(bsz, length, 4, generator=g)
    s0 = torch.randn(bsz, 4, 32, 32, generator=g) * 0.1 if init else None
    kw = dict(initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True)
    a, sa = fallback(q, k, v, gate, beta, **kw)
    b, sb = _convpatch.chunk_gated_delta_rule(q, k, v, gate, beta, **kw)
    assert float((a - b).abs().max() / a.abs().max()) <= 1e-5
    assert float((sa - sb).abs().max() / sa.abs().max()) <= 1e-5


def test_gdn_patch_is_installed_on_cpu_and_recorded(scorer):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    assert getattr(mq.torch_chunk_gated_delta_rule, "__semcer_patch__", False)
    assert scorer.signature["gdn"] == "chunk_solve"


def test_cpu_batch_default(tiny_bundle):
    assert SemCER.from_pretrained(tiny_bundle)._engine.config.max_batch_edits == 8
    assert SemCER.from_pretrained(tiny_bundle, batch_size=3)._engine.config.max_batch_edits == 3


def _cache_tensors(cache):
    out = []
    for layer in cache.layers:
        for d in (getattr(layer, "conv_states", {}), getattr(layer, "recurrent_states", {})):
            out += [t for t in d.values() if t is not None]
        if getattr(layer, "is_initialized", False):
            out += [layer.keys, layer.values]
    return out


@pytest.mark.parametrize("share", [True, False])
def test_read_only_cache_fork_equals_deepcopy_broadcast_and_never_writes_base(scorer, share):
    """The suffix-batch fork gives the same hidden states as the old deep copy +
    ``reorder_cache`` broadcast, and leaves the prefix cache untouched."""
    import copy

    from semcer._engine import _fork_cache

    eng = scorer._engine
    bb = eng.model.backbone
    g = torch.Generator().manual_seed(0)
    vocab = bb.config.vocab_size
    prefix = torch.randint(0, vocab, (1, 23), generator=g)
    with torch.inference_mode():
        base = bb(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True).past_key_values
        before = [t.clone() for t in _cache_tensors(base)]
        b, w, k = 5, 9, prefix.shape[1]
        ids = torch.randint(0, vocab, (b, w), generator=g)
        smask = torch.ones((b, w), dtype=torch.long)
        smask[1, 6:] = 0
        full = torch.cat([torch.ones((b, k), dtype=torch.long), smask], dim=1)
        pos = (torch.arange(w) + k).unsqueeze(0).expand(b, w)

        def run(cache):
            return bb(
                input_ids=ids, attention_mask=full, position_ids=pos, past_key_values=cache, use_cache=True
            ).last_hidden_state

        old = copy.deepcopy(base)
        old.reorder_cache(torch.zeros(b, dtype=torch.long))
        want = run(old)
        got = run(_fork_cache(base, b, share=share))
        again = run(_fork_cache(base, b, share=share))  # the base still holds the prefix state
    assert torch.equal(got, again)
    assert float((got - want).abs().max()) <= TOL
    after = _cache_tensors(base)
    assert len(after) == len(before) and all(torch.equal(x, y) for x, y in zip(before, after))


@pytest.mark.parametrize("bsz,length", [(5, 29), (3, 130), (1, 7)])
def test_gdn_chunk_rule_shared_initial_state_and_no_final_state(bsz, length):
    """A broadcast (stride-0) initial state is read in place; skipping
    the final state leaves the output unchanged. Both within fp32 rounding of the
    transformers fallback on a materialised state."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    fallback = _convpatch._installed.get("gdn_original") or mq.torch_chunk_gated_delta_rule
    gen = torch.Generator().manual_seed(bsz * 1000 + length)
    q, k, v = (torch.randn(bsz, length, 4, 16, generator=gen) for _ in range(3))
    gate = -torch.rand(bsz, length, 4, generator=gen) * 2
    beta = torch.rand(bsz, length, 4, generator=gen)
    s0 = (torch.randn(1, 4, 16, 16, generator=gen) * 0.1).expand(bsz, 4, 16, 16)
    kw = dict(use_qk_l2norm_in_kernel=True)
    want, want_state = fallback(q, k, v, gate, beta, initial_state=s0.contiguous(), output_final_state=True, **kw)
    rule = _convpatch.chunk_gated_delta_rule
    got, got_state = rule(q, k, v, gate, beta, initial_state=s0, output_final_state=True, **kw)
    out_only, none = rule(q, k, v, gate, beta, initial_state=s0, output_final_state=False, **kw)
    assert none is None
    scale = float(want.abs().max())
    assert float((got - want).abs().max()) <= 1e-5 * scale
    assert float((got_state - want_state).abs().max()) <= 1e-5 * float(want_state.abs().max())
    assert torch.equal(out_only, got)


def test_discard_final_state_context_only_affects_cpu_kernel_state():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    gen = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(2, 9, 4, 16, generator=gen) for _ in range(3))
    gate, beta = -torch.rand(2, 9, 4, generator=gen), torch.rand(2, 9, 4, generator=gen)
    a, sa = mq.torch_chunk_gated_delta_rule(q, k, v, gate, beta, output_final_state=True)
    with _convpatch.discard_final_state():
        b, sb = mq.torch_chunk_gated_delta_rule(q, k, v, gate, beta, output_final_state=True)
    c, sc = mq.torch_chunk_gated_delta_rule(q, k, v, gate, beta, output_final_state=True)
    assert sa is not None and sb is None and sc is not None
    assert torch.equal(a, b) and torch.equal(a, c)
