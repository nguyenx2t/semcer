"""The cross-pair path (GPU batching) gives the per-pair path's values.

Pairs are grouped by prefix length, a group's prefixes run as one unpadded
forward, and edits from different pairs share suffix forwards. Per-edit
severities must agree with the per-pair path within the 1e-6 fp32 tolerance,
statuses and order exactly, for every grouping the caps allow.
"""

from __future__ import annotations

import pytest
from _pairs import dense_pair, make_pairs

from semcer import SemCER
from semcer._prep import prepare_pair

TOL = 1e-6


def _texts(pairs):
    return [p["reference"] for p in pairs], [p["hypothesis"] for p in pairs]


@pytest.fixture(scope="module")
def many():
    """96 pairs in 16 languages plus a dense pair and one whose split cannot be shared."""
    pairs = make_pairs(96)
    for n, (ref, hyp) in enumerate([dense_pair(200), ("Total: 15!", "Total 75!!")]):
        pairs.append({"id": f"extra-{n}", "language": "en", "reference": ref, "hypothesis": hyp})
    return pairs


@pytest.mark.parametrize("slack,rows,prefill", [(0, 4, 3), (16, 64, 32), (10_000, 16, 96)])
def test_engine_score_pairs_equals_score_pair(scorer, many, slack, rows, prefill):
    eng = scorer._engine
    preps = [prepare_pair(p["reference"], p["hypothesis"]) for p in many]
    want = [eng.score_pair(prep) for prep in preps]
    saved = (eng.config.prefix_slack, eng.config.max_batch_edits, eng.config.max_prefill_pairs)
    eng.config.prefix_slack, eng.config.max_batch_edits, eng.config.max_prefill_pairs = slack, rows, prefill
    try:
        before = (eng.stats.prefix_forwards, eng.stats.prefill_rows)
        got = eng.score_pairs(preps)
    finally:
        eng.config.prefix_slack, eng.config.max_batch_edits, eng.config.max_prefill_pairs = saved
    n = 0
    for a, b in zip(want, got):
        assert [x.status for x in a] == [y.status for y in b]
        assert [x.context for x in a] == [y.context for y in b]
        for x, y in zip(a, b):
            assert abs(x.severity - y.severity) <= TOL
            assert max(abs(p - q) for p, q in zip(x.probs, y.probs)) <= TOL
            n += 1
    assert n > 700
    prefills, rows_prefilled = eng.stats.prefix_forwards - before[0], eng.stats.prefill_rows - before[1]
    assert rows_prefilled > prefills or slack == 0  # prefixes shared a forward


def test_scorer_cross_pair_matches_per_pair(tiny_bundle, scorer, many):
    cross = SemCER.from_pretrained(tiny_bundle, batch_pairs=24)
    assert cross.signature["path"] == "prefix_cached+cross_pair"
    assert scorer.signature["path"] == "prefix_cached"
    refs, hyps = _texts(many)
    ids = [p["id"] for p in many]
    want = scorer.score_batch(refs, hyps, ids=ids)
    got = cross.score_batch(refs, hyps, ids=ids)
    assert [r.id for r in got] == ids
    for a, b in zip(want, got):
        assert (a.status, a.n_edits, a.n_ref_graphemes, a.cer) == (b.status, b.n_edits, b.n_ref_graphemes, b.cer)
        assert a.semcer == pytest.approx(b.semcer, abs=1e-6)
        for x, y in zip(a.edits, b.edits):
            assert (x.op, x.ref_span, x.hyp_span) == (y.op, y.ref_span, y.hyp_span)
            assert abs(x.severity - y.severity) <= TOL
    one = cross.score(refs[5], hyps[5])
    assert one.semcer == pytest.approx(want[5].semcer, abs=1e-6)
    assert cross.stats.forwards < scorer.stats.forwards


def test_cross_pair_truncation_statuses_match(tiny_bundle):
    ref = "Invoice 15: the total is 15 dollars, paid on 3 March."
    hyp = "Invoice 16: the total is 75 dolars, paid on 8 March!!"
    pairs = make_pairs(16) + [{"reference": ref, "hypothesis": hyp}]
    refs, hyps = _texts(pairs)
    for max_length in (48, 64, 96):
        a = SemCER.from_pretrained(tiny_bundle, max_length=max_length).score_batch(refs, hyps)
        b = SemCER.from_pretrained(tiny_bundle, max_length=max_length, batch_pairs=8).score_batch(refs, hyps)
        assert any(r.status == "partial" for r in a)
        for x, y in zip(a, b):
            assert (x.status, x.n_unscored) == (y.status, y.n_unscored)
            assert [e.status for e in x.edits] == [e.status for e in y.edits]
            for e, f in zip(x.edits, y.edits):
                assert (e.severity is None) == (f.severity is None)
                if e.severity is not None:
                    assert abs(e.severity - f.severity) <= TOL


def test_cross_pair_stream_keeps_order_and_isolates_bad_records(tiny_bundle, many):
    cross = SemCER.from_pretrained(tiny_bundle, batch_pairs=8, empty_reference="null")
    records = [dict(p) for p in many[:20]]
    records.insert(3, {"id": "bad-type", "reference": None, "hypothesis": "x"})
    records.insert(9, {"id": "empty-ref", "reference": "", "hypothesis": "x"})
    out = list(cross.score_iter(iter(records), on_error="record"))
    assert [r.id for r in out] == [r["id"] for r in records]
    assert out[3].status == "error" and "TypeError" in out[3].cause
    assert out[9].status == "undefined"
    assert all(r.status in ("ok", "partial") for k, r in enumerate(out) if k not in (3, 9))

    seen = []
    with pytest.raises(TypeError):
        for r in cross.score_iter(iter(records)):
            seen.append(r.id)
    assert seen == [r["id"] for r in records[:3]]  # everything before the bad record was delivered


def test_cross_pair_window_failure_falls_back_to_single_pairs(tiny_bundle, scorer, many, monkeypatch):
    cross = SemCER.from_pretrained(tiny_bundle, batch_pairs=8)
    sub = many[:16]
    want = scorer.score_batch(*_texts(sub))

    def boom(preps):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(cross._engine, "score_pairs", boom)
    got = cross.score_batch(*_texts(sub))
    assert cross.stats.window_fallbacks == 2
    for a, b in zip(want, got):
        assert a.semcer == pytest.approx(b.semcer, abs=1e-6)


def test_window_is_bounded_by_edits_as_well_as_pairs(tiny_bundle, monkeypatch):
    cross = SemCER.from_pretrained(tiny_bundle, batch_pairs=1000)
    monkeypatch.setattr(SemCER, "_WINDOW_EDITS", 40)
    sizes = []
    real = cross._engine.score_pairs
    monkeypatch.setattr(cross._engine, "score_pairs", lambda preps: sizes.append(len(preps)) or real(preps))
    pairs = [p for p in make_pairs(60) if p["reference"] != p["hypothesis"]]
    cross.score_batch(*_texts(pairs))
    assert len(sizes) > 3 and sum(sizes) == len(pairs)


def test_device_auto_resolves_to_an_available_device(tiny_bundle):
    import torch

    s = SemCER.from_pretrained(tiny_bundle, device="auto")
    assert s.device == ("cuda" if torch.cuda.is_available() else "cpu")
    assert s.signature["device"] == s.device
    if s.device == "cpu":
        assert s.batch_pairs == 32
