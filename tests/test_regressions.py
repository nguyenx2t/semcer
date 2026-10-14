"""Bugs found in review, each pinned by the input that showed it."""

from __future__ import annotations

import json
import logging

import pytest

import semcer
from semcer import SemCER, _loader, cli
from semcer.summary import compare


def _row(i, severity_sum=1.0, n_edits=2, n_ref=10, **extra):
    return {
        "id": i, "status": "ok", "semcer": severity_sum / n_ref, "cer": n_edits / n_ref,
        "severity_sum": severity_sum, "n_edits": n_edits, "n_ref_graphemes": n_ref, **extra,
    }  # fmt: skip


def _score(tmp_path, tiny_bundle, *args):
    return cli.main(["score", *map(str, args), "--model", str(tiny_bundle), "--quiet"])


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_incomplete_cached_snapshot_is_completed_not_trusted(tiny_bundle, tmp_path, monkeypatch):
    import huggingface_hub

    partial = tmp_path / "snapshot"
    partial.mkdir()
    (partial / "semcer_meta.json").write_text("{}")  # the weights never arrived
    calls = []

    def fake(local_files_only=False, **kwargs):
        calls.append(local_files_only)
        return str(partial if local_files_only else tiny_bundle)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert _loader.resolve_bundle("someone/model")[0] == tiny_bundle
    assert calls == [True, False]


def test_compare_accepts_mixed_detail_levels_and_refuses_mixed_settings_or_repeated_ids():
    a = [_row(i, reference=f"line {i}") for i in range(6)]
    b = [_row(i, 2.0) for i in range(6)]  # scored with details="scores": no reference text
    assert compare(a, b, samples=20)["n_lines"] == 6
    with pytest.raises(ValueError, match="different models or settings"):
        compare(a, [dict(r, signature={"model": "other"}) for r in b], samples=20)
    with pytest.raises(ValueError, match="more than once"):
        compare(a + a[:1], b, samples=20)
    with pytest.raises(ValueError, match="samples"):
        compare(a, b, samples=0)
    assert len(compare(a, b, samples=20, top=4)["largest_differences"]) == 4


def test_csv_with_a_byte_order_mark_and_tsv_with_a_leading_quote(tiny_bundle, tmp_path):
    table = tmp_path / "pairs.csv"
    table.write_bytes("﻿reference,hypothesis\nabc,abd\n".encode())
    assert _score(tmp_path, tiny_bundle, "--input", table, "--output", tmp_path / "a.jsonl") == 0
    assert _rows(tmp_path / "a.jsonl")[0]["reference"] == "abc"

    tsv = tmp_path / "pairs.tsv"
    tsv.write_text('reference\thypothesis\n"Hello\t"Hello\nplain\tplane\n', encoding="utf-8")
    assert _score(tmp_path, tiny_bundle, "--input", tsv, "--output", tmp_path / "b.jsonl") == 0
    rows = _rows(tmp_path / "b.jsonl")
    assert [r["reference"] for r in rows] == ['"Hello', "plain"]  # two rows, the quote kept

    refs, hyps = tmp_path / "r.txt", tmp_path / "h.txt"
    refs.write_bytes("﻿abc\n".encode())
    hyps.write_text("abc\n", encoding="utf-8")
    assert (
        _score(tmp_path, tiny_bundle, "--references", refs, "--hypotheses", hyps, "--output", tmp_path / "c.jsonl") == 0
    )
    assert _rows(tmp_path / "c.jsonl")[0]["n_edits"] == 0  # the mark is not a character of line 1


def test_bad_input_and_settings_exit_2_without_a_traceback(tiny_bundle, tmp_path, capsys):
    latin = tmp_path / "latin.csv"
    latin.write_bytes("reference,hypothesis\ncaf\xe9,cafe\n".encode("latin-1"))
    assert _score(tmp_path, tiny_bundle, "--input", latin, "--output", tmp_path / "x.jsonl") == 2
    short = tmp_path / "short.csv"
    short.write_text("reference,hypothesis\nonly one field\n", encoding="utf-8")
    assert _score(tmp_path, tiny_bundle, "--input", short, "--output", tmp_path / "y.jsonl") == 2
    assert "short.csv: row 1" in capsys.readouterr().err
    assert not (tmp_path / "y.jsonl.input.jsonl").exists()  # nothing is left behind
    assert _score(tmp_path, tiny_bundle, "--reference", "a", "--hypothesis", "b", "--batch-size", "0") == 2
    assert cli.main(["compare", str(short), str(short), "--samples", "0"]) == 2


def test_existing_output_is_refused_before_the_paired_input_is_rewritten(tiny_bundle, tmp_path):
    refs, hyps, out = tmp_path / "r.txt", tmp_path / "h.txt", tmp_path / "s.jsonl"
    refs.write_text("abc\n", encoding="utf-8")
    hyps.write_text("abd\n", encoding="utf-8")
    assert _score(tmp_path, tiny_bundle, "--references", refs, "--hypotheses", hyps, "--output", out) == 0
    paired = tmp_path / "s.jsonl.input.jsonl"
    before = paired.read_text(encoding="utf-8")
    refs.write_text("xyz\n", encoding="utf-8")
    assert _score(tmp_path, tiny_bundle, "--references", refs, "--hypotheses", hyps, "--output", out) == 2
    assert paired.read_text(encoding="utf-8") == before


def test_resume_scores_an_error_row_again(tiny_bundle, tmp_path):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    inp.write_text(json.dumps({"id": 1, "reference": "abc", "hypothesis": "abd"}) + "\n", encoding="utf-8")
    assert _score(tmp_path, tiny_bundle, "--input", inp, "--output", out) == 0
    row = _rows(out)[0]
    row.update(status="error", semcer=None, cause="out of memory")
    out.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert _score(tmp_path, tiny_bundle, "--input", inp, "--output", out, "--resume") == 0
    assert [r["status"] for r in _rows(out)] == ["ok"]


def test_a_complete_last_row_without_its_newline_is_read(tmp_path, capsys):
    scores = tmp_path / "s.jsonl"
    scores.write_text("\n".join(json.dumps(_row(i)) for i in range(3)), encoding="utf-8")  # no final newline
    assert cli.main(["compare", str(scores), str(scores), "--samples", "10"]) == 0
    assert "3 lines scored by both" in capsys.readouterr().out


def test_enable_logging_prints_each_message_once_and_takes_a_level_number(tiny_bundle, capsys):
    root = logging.getLogger()
    handler = logging.StreamHandler()
    root.addHandler(handler)
    try:
        semcer.enable_logging(logging.INFO)
        SemCER(tiny_bundle)
    finally:
        root.removeHandler(handler)
    assert capsys.readouterr().err.count("batch size") == 1


def test_html_view_keeps_runs_of_spaces(scorer):
    html = scorer.score(reference="a  b", hypothesis="a  c").to_html()
    assert "white-space:pre-wrap" in html and "a  " in html
    assert scorer.score(reference="a", hypothesis="a")._repr_html_().startswith("<div")
