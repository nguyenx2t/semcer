"""CLI scoring, resume and summaries.

- an interrupted-then-resumed run has one row per ID and the same summary as an
  uninterrupted run;
- a changed hypothesis under the same ID is re-scored on resume;
- invalid input or configuration exits 2 before scoring; partial runs exit 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semcer import cli


def _write(path: Path, recs) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    return path


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def records(pairs):
    return [dict(p, system="sysA") for p in pairs[:30]]


def _score(tiny_bundle, *extra):
    return cli.main(["score", "--model", str(tiny_bundle), *extra])


def test_single_pair_prints_explanation(tiny_bundle, capsys):
    rc = _score(tiny_bundle, "--reference", "The total is 15", "--hypothesis", "The total is 75")
    out = capsys.readouterr().out
    assert rc == 0 and "semcer=" in out and "SUB" in out
    rc = _score(tiny_bundle, "--reference", "abc", "--hypothesis", "abd", "--json")
    assert json.loads(capsys.readouterr().out)["n_edits"] == 1


def test_jsonl_run_resume_equals_uninterrupted(tiny_bundle, records, tmp_path):
    inp = _write(tmp_path / "in.jsonl", records)
    full = tmp_path / "full.jsonl"
    assert _score(tiny_bundle, "--input", str(inp), "--output", str(full), "--summary", str(tmp_path / "s1.json")) == 0
    rows = _rows(full)
    assert [r["id"] for r in rows] == [r["id"] for r in records]
    assert all("input_sha256" in r and r["system"] == "sysA" for r in rows)

    # simulate an interruption: 11 complete rows plus a half-written 12th
    part = tmp_path / "part.jsonl"
    text = full.read_text(encoding="utf-8").splitlines(keepends=True)
    part.write_text("".join(text[:11]) + text[11][: len(text[11]) // 2], encoding="utf-8")
    manifest = json.loads((tmp_path / "full.jsonl.manifest.json").read_text())
    manifest["completed"] = False
    (tmp_path / "part.jsonl.manifest.json").write_text(json.dumps(manifest))

    assert _score(tiny_bundle, "--input", str(inp), "--output", str(part)) == 2  # exists, no flag
    assert (
        _score(
            tiny_bundle, "--input", str(inp), "--output", str(part), "--resume", "--summary", str(tmp_path / "s2.json")
        )
        == 0
    )
    resumed = _rows(part)
    assert sorted(r["id"] for r in resumed) == sorted(r["id"] for r in records)
    assert len({r["id"] for r in resumed}) == len(resumed)
    assert (tmp_path / "part.jsonl.bak").is_file()
    s1 = json.loads((tmp_path / "s1.json").read_text())
    s2 = json.loads((tmp_path / "s2.json").read_text())
    # Lines share model calls, so a resumed run batches them differently: equal within fp32 rounding.
    assert s1["semcer_micro"] == pytest.approx(s2["semcer_micro"], abs=1e-6)
    assert s1["status_counts"] == s2["status_counts"]
    assert json.loads((tmp_path / "part.jsonl.manifest.json").read_text())["completed"] is True


def test_resume_rescores_a_changed_hypothesis(tiny_bundle, records, tmp_path):
    inp = _write(tmp_path / "in.jsonl", records[:6])
    out = tmp_path / "out.jsonl"
    assert _score(tiny_bundle, "--input", str(inp), "--output", str(out)) == 0
    before = {r["id"]: r for r in _rows(out)}
    changed = [dict(r) for r in records[:6]]
    changed[2]["hypothesis"] = changed[2]["hypothesis"] + " EXTRA"
    _write(inp, changed)
    assert _score(tiny_bundle, "--input", str(inp), "--output", str(out), "--resume") == 0
    after = {r["id"]: r for r in _rows(out)}
    assert len(_rows(out)) == 6
    cid = changed[2]["id"]
    assert after[cid]["input_sha256"] != before[cid]["input_sha256"]
    assert after[cid]["n_edits"] > before[cid]["n_edits"]
    for rid in after:
        if rid != cid:
            assert after[rid] == before[rid]


def test_resume_refuses_a_different_configuration(tiny_bundle, records, tmp_path):
    inp = _write(tmp_path / "in.jsonl", records[:3])
    out = tmp_path / "out.jsonl"
    assert _score(tiny_bundle, "--input", str(inp), "--output", str(out)) == 0
    assert _score(tiny_bundle, "--input", str(inp), "--output", str(out), "--resume", "--unscored", "impute_one") == 2
    assert (
        _score(tiny_bundle, "--input", str(inp), "--output", str(out), "--overwrite", "--unscored", "impute_one") == 0
    )


@pytest.mark.parametrize(
    "bad,msg",
    [
        ('{"id": 1, "reference": "a", "hypothesis": null}\n', "must be a string"),
        ('{"reference": "a", "hypothesis": "b"}\n', "'id'"),
        (
            '{"id": 1, "reference": "a", "hypothesis": "b"}\n{"id": 1, "reference": "a", "hypothesis": "c"}\n',
            "duplicate id",
        ),
        ("not json\n", "not valid JSON"),
    ],
)
def test_invalid_input_exits_2_before_scoring(tiny_bundle, tmp_path, capsys, bad, msg):
    inp = tmp_path / "in.jsonl"
    inp.write_text(bad, encoding="utf-8")
    assert _score(tiny_bundle, "--input", str(inp), "--output", str(tmp_path / "o.jsonl")) == 2
    assert msg in capsys.readouterr().err
    assert not (tmp_path / "o.jsonl").exists()


def test_invalid_utf8_exits_2(tiny_bundle, tmp_path):
    inp = tmp_path / "in.jsonl"
    inp.write_bytes(b'{"id": 1, "reference": "\xff", "hypothesis": "b"}\n')
    assert _score(tiny_bundle, "--input", str(inp), "--output", str(tmp_path / "o.jsonl")) == 2


def test_error_rows_exit_1_and_keep_going(tiny_bundle, tmp_path):
    recs = [
        {"id": "a", "reference": "abc", "hypothesis": "abd"},
        {"id": "big", "reference": "x" * 60, "hypothesis": "y" * 60},
        {"id": "empty", "reference": "", "hypothesis": "q"},
        {"id": "c", "reference": "abc", "hypothesis": "abc"},
    ]
    inp = _write(tmp_path / "in.jsonl", recs)
    out = tmp_path / "o.jsonl"
    rc = _score(tiny_bundle, "--input", str(inp), "--output", str(out), "--cell-budget", "1000")
    rows = {r["id"]: r for r in _rows(out)}
    assert rc == 1
    assert rows["big"]["status"] == "error" and rows["big"]["cause"].startswith("too_large")
    assert rows["empty"]["status"] == "undefined"
    assert rows["a"]["status"] == rows["c"]["status"] == "ok"


def test_progress_never_echoes_text(tiny_bundle, records, tmp_path, capsys):
    inp = _write(tmp_path / "in.jsonl", records[:5])
    _score(tiny_bundle, "--input", str(inp), "--output", str(tmp_path / "o.jsonl"))
    err = capsys.readouterr().err
    assert "semcer: " in err
    for r in records[:5]:
        assert r["reference"] not in err


def test_summarize_and_inspect(tiny_bundle, records, tmp_path, capsys):
    inp = _write(tmp_path / "in.jsonl", records[:8])
    out = tmp_path / "o.jsonl"
    _score(tiny_bundle, "--input", str(inp), "--output", str(out))
    capsys.readouterr()
    assert cli.main(["summarize", "--input", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    rows = _rows(out)
    num = sum(r["severity_sum"] for r in rows)
    den = sum(r["n_ref_graphemes"] for r in rows)
    assert summary["semcer_micro"] == pytest.approx(num / den)
    assert summary["cer_micro"] == pytest.approx(sum(r["n_edits"] for r in rows) / den)
    assert set(summary["by_language"]) == {r["language"] for r in records[:8]}
    assert summary["by_system"]["sysA"]["n_rows"] == 8
    rid = records[3]["id"]
    assert cli.main(["inspect", "--input", str(out), "--id", rid]) == 0
    assert "semcer=" in capsys.readouterr().out


def test_inspect_needs_edit_detail(tiny_bundle, records, tmp_path, capsys):
    inp = _write(tmp_path / "in.jsonl", records[:2])
    out = tmp_path / "o.jsonl"
    _score(tiny_bundle, "--input", str(inp), "--output", str(out), "--details", "scores")
    assert "edits" not in _rows(out)[0]
    assert cli.main(["inspect", "--input", str(out), "--id", records[0]["id"]]) == 2
    assert "--details edits" in capsys.readouterr().err


def test_summarize_refuses_mixed_signatures(tmp_path):
    rows = [
        {
            "status": "ok",
            "semcer": 0.1,
            "cer": 0.2,
            "severity_sum": 1.0,
            "n_edits": 2,
            "n_ref_graphemes": 10,
            "signature": {"model": "a"},
        },
        {
            "status": "ok",
            "semcer": 0.1,
            "cer": 0.2,
            "severity_sum": 1.0,
            "n_edits": 2,
            "n_ref_graphemes": 10,
            "signature": {"model": "b"},
        },
    ]
    p = _write(tmp_path / "s.jsonl", rows)
    assert cli.main(["summarize", "--input", str(p)]) == 2
    assert cli.main(["summarize", "--input", str(p), "--allow-mixed"]) == 0


def test_progress_reports_time_left_when_the_total_is_known():
    import io

    from semcer.cli import Progress

    out = io.StringIO()
    p = Progress(total=100, every=0.0, stream=out)
    p.tick("ok")
    assert "scored 1/100" in out.getvalue() and "min left" in out.getvalue()
    out = io.StringIO()
    Progress(total=None, every=0.0, stream=out).tick("ok")
    assert "min left" not in out.getvalue()


def test_two_text_files_are_paired_line_by_line(tiny_bundle, tmp_path, capsys):
    refs, hyps, out = tmp_path / "refs.txt", tmp_path / "hyps.txt", tmp_path / "scores.jsonl"
    refs.write_text("The total is 15\n a cat \n", encoding="utf-8")
    hyps.write_text("The total is 75\n a cat \n", encoding="utf-8")
    assert (
        cli.main(
            [
                "score",
                "--references",
                str(refs),
                "--hypotheses",
                str(hyps),
                "--output",
                str(out),
                "--model",
                str(tiny_bundle),
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == [1, 2] and rows[0]["n_edits"] == 1
    assert rows[1]["reference"] == " a cat " and rows[1]["n_edits"] == 0  # spaces are kept

    hyps.write_text("only one line\n", encoding="utf-8")
    assert (
        cli.main(
            [
                "score",
                "--references",
                str(refs),
                "--hypotheses",
                str(hyps),
                "--output",
                str(tmp_path / "x.jsonl"),
                "--model",
                str(tiny_bundle),
            ]
        )
        == 2
    )
    assert "2 lines" in capsys.readouterr().err
