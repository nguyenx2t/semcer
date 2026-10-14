"""Comparing two systems on the same references, and CSV input."""

from __future__ import annotations

import json

import pytest

from semcer import SemCER, cli
from semcer.summary import compare


def _row(i, severity_sum, n_edits, n_ref=10, **extra):
    return {
        "id": i, "status": "ok", "semcer": severity_sum / n_ref, "cer": n_edits / n_ref,
        "severity_sum": severity_sum, "n_edits": n_edits, "n_ref_graphemes": n_ref, **extra,
    }  # fmt: skip


def test_compare_reports_ratio_of_sums_and_a_paired_interval():
    a = [_row(i, 1.0, 2) for i in range(40)]
    b = [_row(i, 2.0 if i % 2 else 1.0, 2) for i in range(40)]
    r = compare(a, b, samples=500, seed=1)
    assert r["n_lines"] == 40 and r["resampled"] == "lines"
    assert r["a"]["semcer_micro"] == pytest.approx(0.1) and r["b"]["semcer_micro"] == pytest.approx(0.15)
    assert r["semcer_difference"] == pytest.approx(-0.05) and r["cer_difference"] == 0
    lo, hi = r["semcer_difference_ci95"]
    assert lo <= -0.05 <= hi < 0  # the first system is better, and the interval excludes zero
    assert (r["lines_a_better"], r["lines_b_better"], r["lines_tied"]) == (20, 0, 20)
    assert r == compare(a, b, samples=500, seed=1)  # the seed fixes the draws
    assert compare(a, a, samples=100)["semcer_difference_ci95"] == [0.0, 0.0]


def test_compare_resamples_documents_when_rows_carry_them():
    a = [_row(i, 1.0, 2, document_id=i // 10) for i in range(40)]
    b = [_row(i, 1.5, 2, document_id=i // 10) for i in range(40)]
    assert compare(a, b, samples=50)["resampled"] == "documents"


def test_compare_refuses_different_lines_or_references_and_skips_unscored():
    a = [_row(i, 1.0, 2) for i in range(5)]
    with pytest.raises(ValueError, match="same ids"):
        compare(a, a[:4])
    with pytest.raises(ValueError, match="different reference"):
        compare(a, [_row(i, 1.0, 2, n_ref=11) for i in range(5)])
    b = [dict(r) for r in a]
    b[0].update(status="partial", semcer=None)
    r = compare(a, b, samples=10)
    assert (r["n_lines"], r["n_excluded"]) == (4, 1)


def test_cli_scores_a_csv_and_compares_two_runs(tiny_bundle, tmp_path, capsys):
    table = tmp_path / "pairs.csv"
    table.write_text('reference,hypothesis,language\n"a cat, sat",a cut sat,en\nhello,hello,en\n', encoding="utf-8")
    out = tmp_path / "a.jsonl"
    assert cli.main(["score", "--input", str(table), "--output", str(out), "--model", str(tiny_bundle), "--quiet"]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == [1, 2] and rows[0]["reference"] == "a cat, sat" and rows[0]["language"] == "en"
    bad = tmp_path / "bad.csv"
    bad.write_text("ref,hyp\na,b\n", encoding="utf-8")
    assert (
        cli.main(["score", "--input", str(bad), "--output", str(tmp_path / "x.jsonl"), "--model", str(tiny_bundle)])
        == 2
    )
    capsys.readouterr()
    assert cli.main(["compare", str(out), str(out), "--samples", "20"]) == 0
    text = capsys.readouterr().out
    assert "2 lines scored by both" in text and "difference" in text


def test_constructor_rejects_an_unknown_setting(tiny_bundle):
    assert SemCER(tiny_bundle, chunk_edits=8).batch_size == 8  # an advanced setting is accepted
    with pytest.raises(TypeError, match="max_batch_edits"):
        SemCER(tiny_bundle, max_batch_edits=4)
