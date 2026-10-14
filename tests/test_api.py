"""Public API and result contract.

- ``0 <= semcer <= cer`` for complete rows; contributions sum to semcer; vector
  lengths match; equal strings score zero with no model work;
- an empty reference raises by default; ``null`` is explicit;
- JSON round trip, never NaN; display spans place insertions and deletions;
- ``import semcer`` imports no torch and makes no network call.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

import semcer
from semcer import SemCER, UndefinedMetricError
from semcer.result import PairResult, severity_level


def test_invariants_over_all_pairs(scorer, pairs):
    results = scorer.score_batch(
        [p["reference"] for p in pairs],
        [p["hypothesis"] for p in pairs],
        ids=[p["id"] for p in pairs],
        languages=[p["language"] for p in pairs],
    )
    assert [r.id for r in results] == [p["id"] for p in pairs]
    for r in results:
        assert r.status == "ok"
        assert 0.0 <= r.semcer <= r.cer + 1e-12
        assert len(r.severities) == r.n_edits == len(r.edits)
        assert math.fsum(r.contributions) == pytest.approx(r.semcer, abs=1e-12)
        assert r.severity_sum == pytest.approx(r.semcer * r.n_ref_graphemes)
        assert r.coverage == 1.0 and r.n_unscored == 0
        assert r.cer == pytest.approx(r.n_edits / r.n_ref_graphemes)
        assert r.cer == pytest.approx(semcer.cer(r.reference, r.hypothesis))
        assert r.metadata["language"] in {p["language"] for p in pairs}


def test_exact_match_is_zero_without_model_work(scorer):
    before = scorer.stats.forwards
    r = scorer.score("Straße 15", "Straße 15")
    assert (r.status, r.semcer, r.cer, r.n_edits, r.edits, r.coverage) == ("ok", 0.0, 0.0, 0, [], 1.0)
    # NFC-equal strings are an exact match too
    assert scorer.score("é", "é").n_edits == 0
    assert scorer.stats.forwards == before


def test_empty_hypothesis_is_all_deletions(scorer):
    r = scorer.score("abc", "")
    assert r.n_edits == 3 and {e.op for e in r.edits} == {"DEL"} and r.cer == 1.0


def test_empty_reference_policies(tiny_bundle, scorer):
    with pytest.raises(UndefinedMetricError):
        scorer.score("", "abc")
    r = SemCER.from_pretrained(tiny_bundle, empty_reference="null").score("", "abc")
    assert r.status == "undefined" and r.semcer is None and r.cer is None and r.coverage is None


def test_invalid_inputs_are_rejected(scorer):
    with pytest.raises(TypeError):
        scorer.score(None, "abc")
    with pytest.raises(TypeError):
        scorer.score("abc", 5)
    with pytest.raises(ValueError, match="differ in length"):
        scorer.score_batch(["a", "b"], ["a"])
    assert scorer.score_batch([], []) == []
    with pytest.raises(ValueError, match="details"):
        scorer.score("a", "b", details="everything")


def test_score_iter_error_records(tiny_bundle, scorer):
    recs = [
        {"id": 1, "reference": "abc", "hypothesis": "abd"},
        {"id": 2, "reference": "", "hypothesis": "x"},
        {"id": 3, "reference": None, "hypothesis": "x"},
        {"id": 4, "reference": "abc", "hypothesis": "abc"},
    ]
    out = list(scorer.score_iter(recs, on_error="record"))
    assert [r.id for r in out] == [1, 2, 3, 4]
    assert [r.status for r in out] == ["ok", "error", "error", "ok"]
    assert "UndefinedMetricError" in out[1].cause and "TypeError" in out[2].cause
    with pytest.raises(UndefinedMetricError):
        list(scorer.score_iter(recs))


def test_details_modes(scorer):
    ref, hyp = "The total is 15", "The total is 75!"
    full = scorer.score(ref, hyp)
    lean = scorer.score(ref, hyp, details="scores")
    dist = scorer.score(ref, hyp, details="distributions")
    assert lean.edits is None and lean.semcer == full.semcer and "edits" not in lean.to_dict()
    with pytest.raises(ValueError):
        lean.severities
    assert full.edits[0].probs is None
    for e in dist.edits:
        assert len(e.probs) == 5 and math.fsum(e.probs) == pytest.approx(1.0)
        assert e.severity == pytest.approx(sum(k * p for k, p in enumerate(e.probs)) / 4, abs=1e-6)


def test_json_round_trip_without_nan(scorer, tiny_bundle):
    r = scorer.score("Tổng cộng là 15 đồng.", "Tong cong la 75 dong", id="vi-1", details="distributions")
    text = r.to_json()
    back = PairResult.from_dict(json.loads(text))
    assert back.to_dict() == r.to_dict()
    u = SemCER.from_pretrained(tiny_bundle, empty_reference="null").score("", "x")
    d = json.loads(u.to_json())
    assert d["semcer"] is None and d["cer"] is None
    bad = PairResult(
        status="ok", semcer=float("nan"), cer=float("inf"), severity_sum=None, n_edits=0, n_ref_graphemes=1
    )
    assert json.loads(bad.to_json())["semcer"] is None


def test_display_spans_place_gaps_and_graphemes(scorer):
    r = scorer.score("abc", "aXbc")  # insertion between a and b
    (e,) = r.edits
    assert e.op == "INS" and e.ref_span == (1, 1) and e.hyp_span == (1, 2)
    assert r.hypothesis[slice(*e.hyp_char_span)] == "X"
    r = scorer.score("abc", "ac")  # deletion of b
    (e,) = r.edits
    assert e.op == "DEL" and e.ref_span == (1, 2) and e.hyp_span == (1, 1)
    assert r.reference[slice(*e.ref_char_span)] == "b"
    r = scorer.score("नमस्ते दुनिया", "नमस्ते दुनियाा")  # a combining grapheme
    for e in r.edits:
        if e.ref_grapheme:
            assert r.reference[slice(*e.ref_char_span)] == e.ref_grapheme
        if e.hyp_grapheme:
            assert r.hypothesis[slice(*e.hyp_char_span)] == e.hyp_grapheme
    r = scorer.score("ab", "aXXb")  # repeated insertions at one boundary stay separate
    assert [e.op for e in r.edits] == ["INS", "INS"] and [e.hyp_span for e in r.edits] == [(1, 2), (2, 3)]


def test_display_spans_consistent_over_all_pairs(scorer, pairs):
    for p in pairs:
        r = scorer.score(p["reference"], p["hypothesis"])
        for e in r.edits:
            if e.op != "INS":
                assert r.reference[slice(*e.ref_char_span)] == e.ref_grapheme
            if e.op != "DEL":
                assert r.hypothesis[slice(*e.hyp_char_span)] == e.hyp_grapheme


def test_worst_edits_explain_and_levels(scorer):
    r = scorer.score("The total is 15 dollars", "Tha tota1 is 75 dollar")
    worst = r.worst_edits(2)
    assert len(worst) == 2 and worst[0].severity >= worst[1].severity
    assert r.worst_edits(0) == [] and len(r.worst_edits(100)) == r.n_edits
    text = r.explain()
    assert text.count("\n") == r.n_edits
    assert "\033[" not in text and "\033[" in r.explain(color=True)
    assert [severity_level(x) for x in (0.1, 0.25, 0.75, 0.76, None)] == [
        "harmless",
        "partially recoverable",
        "partially recoverable",
        "meaning lost",
        None,
    ]


def test_import_is_torch_free_and_offline():
    code = (
        "import sys, socket\n"
        "def boom(*a, **k): raise SystemExit('network touched')\n"
        "socket.socket.connect = boom\n"
        "import semcer\n"
        "assert semcer.cer('abc', 'abd') == 1/3\n"
        "assert semcer.cer('A b', 'ab', normalization='full') == 0\n"
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules, "
        "sorted(m for m in sys.modules if m.startswith(('torch', 'transformers')))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_module_level_score_uses_the_default_and_clears(tiny_bundle, monkeypatch):
    monkeypatch.setenv("SEMCER_MODEL", str(tiny_bundle))
    semcer.clear_cache()
    try:
        r = semcer.score(reference="The total is 15", hypothesis="The total is 75")
        assert r.status == "ok" and r.n_edits == 1
        first = semcer._DEFAULT["scorer"]
        semcer.score("a", "b")
        assert semcer._DEFAULT["scorer"] is first
    finally:
        semcer.clear_cache()
    assert "scorer" not in semcer._DEFAULT


def test_batch_size_can_be_set_and_does_not_change_scores(tiny_bundle, scorer, pairs):
    sub = [p for p in pairs if p["reference"] != p["hypothesis"]][:10]
    refs, hyps = [p["reference"] for p in sub], [p["hypothesis"] for p in sub]
    small = SemCER(tiny_bundle, batch_size=2)
    assert small.batch_size == 2 and scorer.batch_size == 8
    for a, b in zip(scorer.score_batch(refs, hyps), small.score_batch(refs, hyps)):
        assert a.semcer == pytest.approx(b.semcer, abs=1e-6)
    assert small.stats.max_forward_rows <= 2
    with pytest.raises(ValueError, match="batch_size"):
        SemCER(tiny_bundle, batch_size=0)


def test_html_view_marks_every_edit_and_escapes_text(scorer):
    r = scorer.score(reference="a <b> cat sat", hypothesis="a <b> cut sat!")
    html = r._repr_html_()
    assert html.count('<span style="background') >= r.n_edits + 3  # every edit, plus the three-level key
    assert "&lt;b&gt;" in html and "<b> cat" not in html  # text is escaped
    assert "severity" in html and "REF" in html and "HYP" in html
    assert "<pre>" in scorer.score(reference="abc", hypothesis="abd", details="scores")._repr_html_()


def test_library_is_silent_by_default_and_progress_is_opt_in(tiny_bundle, capfd):
    s = SemCER(tiny_bundle)
    s.score_batch(["a cat", "hello"], ["a cut", "hello"])
    assert capfd.readouterr() == ("", "")
    s.score_batch(["a cat", "hello"], ["a cut", "hello"], progress=True)
    assert "semcer" in capfd.readouterr().err
