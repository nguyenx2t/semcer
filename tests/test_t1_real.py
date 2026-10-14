"""T1: a real-size bundle (Qwen3.5-0.8B-Base weights). Opt-in, never in CI.

    SEMCER_T1_BUNDLE=bundle/ python -m pytest tests/test_t1_real.py -s

``bundle/`` is any real-size (Qwen3.5-0.8B) bundle.
``SEMCER_T1_PAIRS`` sets how many of the 16-language synthetic pairs are used.
"""

from __future__ import annotations

import os

import pytest

from semcer import SemCER
from semcer._prep import PairTokenizer, prepare_pair, tokenize_edits_full

BUNDLE = os.environ.get("SEMCER_T1_BUNDLE")
N_PAIRS = int(os.environ.get("SEMCER_T1_PAIRS", "48"))
t1 = pytest.mark.skipif(not BUNDLE, reason="set SEMCER_T1_BUNDLE to run T1")
TOL = 1e-6


@pytest.fixture(scope="module")
def pairs():
    from _pairs import make_pairs

    return make_pairs(N_PAIRS)


@t1
def test_real_tokenizer_shared_prefix_split_holds(pairs):
    from _pairs import dense_pair, make_pairs

    s = SemCER.from_pretrained(BUNDLE)
    tok = s._tokenizer
    assert tok.is_fast and type(tok).__name__.startswith("Qwen")
    n = 0
    extra = (dense_pair(300), ("Total: 15!", "Total 75!!"))
    for p in make_pairs(48) + [dict(reference=r, hypothesis=h) for r, h in extra]:
        prep = prepare_pair(p["reference"], p["hypothesis"])
        idx = range(len(prep.edits))
        for ml in (2048, 128):
            pt = PairTokenizer(prep, tok, ml)
            got = [(t.ids, t.ref_span, t.hyp_span, t.status) for t in pt(idx)]
            want = [(t.ids, t.ref_span, t.hyp_span, t.status) for t in tokenize_edits_full(prep, tok, ml, idx)]
            assert got == want
            assert pt.shared is True or not prep.edits
            n += len(prep.edits)
    print(f"\nreal Qwen tokenizer: shared-prefix tokenization == full on {n} edit inputs")


@t1
def test_real_weights_cached_matches_uncached_and_cpu_kernels(pairs):
    s = SemCER.from_pretrained(BUNDLE)  # CPU kernels + shared prefix + header level
    u = SemCER.from_pretrained(BUNDLE, cache_prefix=False)
    refs, hyps = [p["reference"] for p in pairs], [p["hypothesis"] for p in pairs]
    a, b = s.score_batch(refs, hyps), u.score_batch(refs, hyps)
    worst = max(
        (abs(x.severity - y.severity) for ra, rb in zip(a, b) for x, y in zip(ra.edits, rb.edits)),
        default=0.0,
    )
    print(
        f"\nT1 cached(+header, CPU kernels) vs uncached: max |dE[d]/4| = {worst:.2e}; "
        f"header rows {s.stats.header_rows}, fallback rows {s.stats.fallback_rows}"
    )
    assert worst <= TOL
    assert s.stats.fallback_rows == 0
