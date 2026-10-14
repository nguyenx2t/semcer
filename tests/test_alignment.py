"""Alignment: fixed edit scripts (tie-breaks and index conventions the model
input assumes), and the cell budget refuses a runaway pair before any DP table
is allocated. ``tests/test_engine_parity.py`` also checks the full model inputs
of 48 pairs against the reference implementation."""

from __future__ import annotations

import pytest

from semcer import _align
from semcer._prep import PairTooLarge, prepare_pair


@pytest.mark.parametrize(
    "ref,hyp,want",
    [
        ("ab", "b", [("DEL", 0, 0, "a", None)]),
        ("a", "aa", [("INS", 0, 0, None, "a")]),
        ("abc", "axc", [("SUB", 1, 1, "b", "x")]),
        ("", "ab", [("INS", 0, 0, None, "a"), ("INS", 0, 1, None, "b")]),
        ("ab", "", [("DEL", 0, 0, "a", None), ("DEL", 1, 0, "b", None)]),
        ("ab", "ba", [("SUB", 0, 0, "a", "b"), ("SUB", 1, 1, "b", "a")]),
        ("\u00e9", "e\u0301", []),  # equal after NFC
    ],
)
def test_edit_scripts(ref, hyp, want):
    got = [(e.op, e.i, e.j, e.g_r, e.g_h) for e in _align.align_graphemes(ref, hyp)[2]]
    assert got == want


def test_cell_budget_refuses_before_aligning():
    ref, hyp = "a" * 300, "b" * 300
    with pytest.raises(PairTooLarge) as ei:
        prepare_pair(ref, hyp, cell_budget=301 * 300)
    assert ei.value.n_ref == 300 and ei.value.n_hyp == 300
    assert len(prepare_pair(ref, hyp, cell_budget=301 * 301).edits) == 300
    assert prepare_pair(ref, hyp, cell_budget=None).edits


def test_exact_match_skips_alignment_even_over_budget():
    assert prepare_pair("x" * 1000, "x" * 1000, cell_budget=10).edits == []
