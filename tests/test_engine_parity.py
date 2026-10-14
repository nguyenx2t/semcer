"""The engine reproduces the reference implementation's scoring path (T0).

- the model input is byte-identical to the reference ``build_pair_records``;
- per-edit E[d]/4 of the prefix-cached and uncached paths agree within 1e-6
  (fp32, CPU) over 48 pairs in all 16 languages, and match the reference
  implementation's frozen outputs (``tests/fixtures/reference_parity.json``);
- ``unscored="impute_one"`` reproduces the reference ``score_pair_student``;
- scalar, batch and stream calls agree; a partial pair is reported, not hidden.
"""

from __future__ import annotations

import pytest
from conftest import load_golden

from semcer import SemCER, _convpatch
from semcer._prep import prepare_pair

TOL = 1e-6
#: The golden outputs were frozen on one machine; other CPUs' BLAS kernels round
#: differently, so the cross-machine check allows 1e-5 (same-machine parity: <=1e-6).
GOLDEN_TOL = 1e-5


def _sevs(results):
    return [[e.severity for e in r.edits] if r.edits is not None else [] for r in results]


@pytest.fixture(scope="module")
def uncached(tiny_bundle):
    return SemCER.from_pretrained(tiny_bundle, cache_prefix=False)


def test_cached_path_is_used_and_matches_uncached(scorer, uncached, pairs):
    before = scorer.stats.cached_rows
    a = scorer.score_batch(
        [p["reference"] for p in pairs], [p["hypothesis"] for p in pairs], ids=[p["id"] for p in pairs]
    )
    b = uncached.score_batch(
        [p["reference"] for p in pairs], [p["hypothesis"] for p in pairs], ids=[p["id"] for p in pairs]
    )
    assert scorer.stats.cached_rows > before
    assert scorer.signature["path"] == "prefix_cached" and uncached.signature["path"] == "uncached"
    n_edits = 0
    for ra, rb in zip(a, b):
        assert ra.id == rb.id and ra.n_edits == rb.n_edits and ra.status == rb.status
        for ea, eb in zip(ra.edits, rb.edits):
            n_edits += 1
            assert (ea.severity is None) == (eb.severity is None)
            if ea.severity is not None:
                assert abs(ea.severity - eb.severity) <= TOL
    assert n_edits > 300
    langs = {p["language"] for p in pairs}
    assert len(langs) == 16


def test_matches_reference_outputs(tiny_bundle):
    gold = load_golden("reference_parity.json")["pairs"]
    _convpatch.uninstall()  # the reference outputs were computed with transformers' F.conv1d fallback
    try:
        ref = SemCER.from_pretrained(tiny_bundle, unscored="impute_one")
        _convpatch.uninstall()
        ours = ref.score_batch([g["reference"] for g in gold], [g["hypothesis"] for g in gold])
    finally:
        _convpatch.install()
    worst, n = 0.0, 0
    for g, r in zip(gold, ours):
        # identical model inputs
        prep = prepare_pair(g["reference"], g["hypothesis"])
        mine = [[t, list(rs), list(hs)] for t, rs, hs, _ in (prep.edit_input(k) for k in range(len(prep.edits)))]
        assert mine == g["records"], g["id"]
        # identical per-edit severities
        assert len(r.edits or []) == len(g["severities"])
        for e, s in zip(r.edits or [], g["severities"]):
            assert (e.severity is None) == (s is None)
            if s is not None:
                worst = max(worst, abs(e.severity - s))
                n += 1
        if r.n_ref_graphemes:
            assert r.semcer == pytest.approx(g["semcer_impute_one"], abs=GOLDEN_TOL), g["id"]
    assert n > 300
    assert worst <= GOLDEN_TOL, worst


def test_conv_patch_parity_on_the_model(tiny_bundle, pairs):
    patched = SemCER.from_pretrained(tiny_bundle)
    sub = pairs[:32]
    a = patched.score_batch([p["reference"] for p in sub], [p["hypothesis"] for p in sub])
    _convpatch.uninstall()
    try:
        b = patched.score_batch([p["reference"] for p in sub], [p["hypothesis"] for p in sub])
    finally:
        _convpatch.install()
    for x, y in zip(_sevs(a), _sevs(b)):
        for s, t in zip(x, y):
            assert abs(s - t) <= 1e-5
    assert patched.signature["conv"] == "shift_sum"


def test_scalar_batch_and_stream_agree(scorer, pairs):
    sub = pairs[:30]
    single = [scorer.score(p["reference"], p["hypothesis"], id=p["id"]) for p in sub]
    batch = scorer.score_batch(
        [p["reference"] for p in sub], [p["hypothesis"] for p in sub], ids=[p["id"] for p in sub]
    )
    stream = list(scorer.score_iter(iter(sub)))
    assert [r.id for r in stream] == [p["id"] for p in sub]
    for a, b, c in zip(single, batch, stream):
        assert _sevs([a]) == _sevs([b]) == _sevs([c])
        assert a.semcer == b.semcer == c.semcer


def test_micro_batch_caps_do_not_change_scores(tiny_bundle, scorer, pairs):
    small = SemCER.from_pretrained(tiny_bundle, batch_size=3, max_batch_tokens=64, chunk_edits=5)
    sub = [p for p in pairs if p["reference"] != p["hypothesis"]][:25]
    a = scorer.score_batch([p["reference"] for p in sub], [p["hypothesis"] for p in sub])
    b = small.score_batch([p["reference"] for p in sub], [p["hypothesis"] for p in sub])
    for x, y in zip(_sevs(a), _sevs(b)):
        for s, t in zip(x, y):
            assert abs(s - t) <= TOL


def _partial_window(scorer, ref, hyp):
    """A max_length that truncates some but not all of this pair's edits."""
    from semcer._prep import PairTokenizer

    prep = prepare_pair(ref, hyp)
    for m in range(40, 2048, 3):
        st = [t.status for t in PairTokenizer(prep, scorer._tokenizer, m)(range(len(prep.edits)))]
        if 0 < st.count("truncated") < len(st) and "zero_token" not in st:
            return m
    raise AssertionError("no partial window found")


def test_overflow_is_partial_and_impute_one_charges_one(tiny_bundle, scorer):
    ref = "Invoice 15: the total is 15 dollars, paid on 3 March."
    hyp = "Invoice 16: the total is 75 dolars, paid on 8 March!!"
    m = _partial_window(scorer, ref, hyp)
    short = SemCER.from_pretrained(tiny_bundle, max_length=m)
    r = short.score(ref, hyp)
    assert r.status == "partial" and r.semcer is None
    assert 0 < r.n_unscored < r.n_edits
    assert r.coverage == pytest.approx((r.n_edits - r.n_unscored) / r.n_edits)
    assert sum(e.severity is None for e in r.edits) == r.n_unscored
    assert all(e.status == "truncated" for e in r.edits if e.severity is None)
    imp = SemCER.from_pretrained(tiny_bundle, max_length=m, unscored="impute_one").score(ref, hyp)
    assert imp.status == "partial"
    assert imp.semcer == pytest.approx((r.severity_sum + r.n_unscored) / r.n_ref_graphemes)
    assert imp.effective_severity_sum == pytest.approx(r.severity_sum + r.n_unscored)
    with pytest.raises(Exception, match="could not be scored"):
        SemCER.from_pretrained(tiny_bundle, max_length=m, unscored="raise").score(ref, hyp)


def test_tokenizer_boundary_fallback_is_exact(scorer, uncached):
    # A hypothesis ending in punctuation lets the byte-level pre-tokenizer glue it
    # to the "\n\n" separator, so the prefix tokens are not a prefix of the full
    # encoding; the engine must still be exact (shorter split point or fallback).
    ref, hyp = "Total: 15!", "Total 75!!"
    a, b = scorer.score(ref, hyp), uncached.score(ref, hyp)
    for x, y in zip(a.edits, b.edits):
        assert abs(x.severity - y.severity) <= TOL


def test_runaway_pair_past_the_window_is_partial_without_model_work(tiny_bundle):
    from _pairs import dense_pair

    ref, hyp = dense_pair(1500)  # the pair layer alone overflows 2,048 tokens
    s = SemCER.from_pretrained(tiny_bundle)
    r = s.score(ref, hyp, details="scores")
    assert r.status == "partial" and r.semcer is None
    assert r.n_unscored == r.n_edits > 400 and r.coverage == 0.0
    assert s.stats.forwards == 0


def test_runaway_pair_streams_with_bounded_batches(tiny_bundle):
    import psutil
    from _pairs import dense_pair

    ref, hyp = dense_pair(900)
    s = SemCER.from_pretrained(tiny_bundle, max_length=8192, batch_size=8, max_batch_tokens=1024, chunk_edits=64)
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    r = s.score(ref, hyp, details="scores")
    growth = proc.memory_info().rss - rss0
    assert r.status == "ok" and r.n_edits > 500
    st = s.stats
    assert st.max_forward_rows <= 8
    assert st.max_forward_tokens <= max(1024, max_suffix(s, ref, hyp))
    assert st.forwards >= r.n_edits // 8
    assert growth < 512 * 2**20  # bounded by the batch caps, not by the edit count


def max_suffix(s, ref, hyp):
    """The longest single row, which may run alone above the token cap."""
    from semcer._prep import PairTokenizer

    prep = prepare_pair(ref, hyp)
    toks = PairTokenizer(prep, s._tokenizer, s.max_length)(range(min(64, len(prep.edits))))
    return max(len(t.ids) for t in toks)
