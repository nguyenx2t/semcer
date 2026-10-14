"""``semcer`` command line.

    semcer score --reference "The total is 15" --hypothesis "The total is 75" --model DIR
    semcer score --input pairs.jsonl --output scores.jsonl [--resume | --overwrite]
    semcer score --references refs.txt --hypotheses hyps.txt --output scores.jsonl
    semcer score --input pairs.csv --output scores.jsonl
    semcer compare system_a.jsonl system_b.jsonl
    semcer summarize --input scores.jsonl [--output summary.json]
    semcer inspect --input scores.jsonl --id example-001

JSONL in, JSONL out. Input rows need ``id``, ``reference`` and ``hypothesis``
(strings; whitespace is meaningful); ``language``, ``document_id``, ``system`` and
``metadata`` are carried through. IDs must be unique within a run.

Output rows are written and flushed one at a time. ``<output>.manifest.json``
records the scoring configuration; ``--resume`` requires the same configuration,
keeps every completed row whose input text hash still matches, re-scores changed
or missing ones, and recovers an incomplete trailing line (the original file is
kept as ``<output>.bak``). Progress goes to stderr and never echoes text.

Exit codes: 0 all rows scored without error; 1 interrupted or some rows are
``error``; 2 invalid usage, configuration or input (before any scoring).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

EXIT_OK, EXIT_PARTIAL, EXIT_USAGE = 0, 1, 2


class UsageError(Exception):
    pass


# -- helpers -------------------------------------------------------------------
def input_hash(reference: str, hypothesis: str) -> str:
    return hashlib.sha256(json.dumps([reference, hypothesis], ensure_ascii=False).encode("utf-8")).hexdigest()


def read_jsonl_records(path: Path) -> list[tuple[Any, str]]:
    """First pass over the input: validate every record and return ``[(id, hash)]``.

    Raises :class:`UsageError` on invalid UTF-8, JSON, fields or duplicate IDs.
    """
    seen: set = set()
    out: list[tuple[Any, str]] = []
    with open(path, "rb") as f:
        for n, raw in enumerate(f, 1):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise UsageError(f"{path}:{n}: not valid UTF-8 ({exc})") from exc
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise UsageError(f"{path}:{n}: not valid JSON ({exc})") from exc
            if not isinstance(rec, dict):
                raise UsageError(f"{path}:{n}: each line must be a JSON object")
            if "id" not in rec or not isinstance(rec["id"], (str, int)) or isinstance(rec["id"], bool):
                raise UsageError(f"{path}:{n}: 'id' (string or integer) is required")
            for key in ("reference", "hypothesis"):
                if not isinstance(rec.get(key), str):
                    raise UsageError(f"{path}:{n}: {key!r} must be a string (null is rejected)")
            if rec["id"] in seen:
                raise UsageError(f"{path}:{n}: duplicate id {rec['id']!r}; ids must be unique per run")
            seen.add(rec["id"])
            out.append((rec["id"], input_hash(rec["reference"], rec["hypothesis"])))
    return out


def iter_records(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_output_rows(path: Path) -> tuple[list[dict], bool]:
    """Complete rows of an existing output, and whether a broken tail was dropped."""
    rows: list[dict] = []
    broken = False
    data = path.read_bytes()
    lines = data.split(b"\n")
    for k, raw in enumerate(lines):
        if not raw.strip():
            continue
        last = k == len(lines) - 1  # no trailing newline: the write was cut off
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if last:
                broken = True
                continue
            raise UsageError(f"{path}: line {k + 1} is corrupt (not the trailing line); refusing to resume")
        rows.append(row)  # a complete object is a complete row, even without its newline
    return rows, broken


def config_hash(signature: dict, details: str) -> str:
    from semcer.summary import signature_key

    return hashlib.sha256((signature_key(signature) + "|" + details).encode()).hexdigest()


def _manifest_path(output: Path) -> Path:
    return output.with_name(output.name + ".manifest.json")


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _color() -> bool:
    """Color terminal output unless it is piped or ``NO_COLOR`` is set."""
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


class Progress:
    def __init__(self, total: int | None, every: float = 2.0, stream=sys.stderr, enabled: bool = True):
        self.total, self.every, self.stream = total, every if enabled else float("inf"), stream
        self.done = self.errors = self.skipped = 0
        self.t0 = self.last = time.monotonic()

    def tick(self, status: str | None = None, *, skipped: bool = False) -> None:
        if skipped:
            self.skipped += 1
        else:
            self.done += 1
            if status == "error":
                self.errors += 1
        now = time.monotonic()
        if now - self.last >= self.every:
            self.last = now
            rate = self.done / max(now - self.t0, 1e-9)
            of = eta = ""
            if self.total is not None:
                of = f"/{self.total}"
                left = (self.total - self.done - self.skipped) / max(rate, 1e-9)
                eta = f", about {left / 60:.0f} min left" if self.done else ""
            print(
                f"semcer: scored {self.done + self.skipped}{of} "
                f"(resumed {self.skipped}, errors {self.errors}) {rate:.2f} pairs/s{eta}",
                file=self.stream,
                flush=True,
            )


# -- model options -------------------------------------------------------------
def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--model", help="bundle directory or Hub id (default: $SEMCER_MODEL / release)")
    g.add_argument("--revision", help="Hub revision (commit) to pin")
    g.add_argument("--local-files-only", action="store_true", help="never touch the network")
    g.add_argument("--device", default="auto", help="auto (CUDA when available, else CPU), cpu or cuda[:N]")
    g.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    g.add_argument("--unscored", default="partial", choices=["partial", "impute_one", "raise"])
    g.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="edits per model call (default 8 on CPU; 16 to 64 on a GPU by its memory)",
    )
    g.add_argument(
        "--batch-pairs", type=int, default=None, help="lines prepared together (default 32 on CPU; 64 or 256 on a GPU)"
    )
    g.add_argument("--cell-budget", type=int, default=None, help="alignment table cells before a line is refused")


def _load_scorer(args, *, empty_reference: str):
    from semcer.metric import SemCER

    advanced = {k: v for k in ("batch_pairs", "cell_budget") if (v := getattr(args, k)) is not None}
    return SemCER(
        args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        unscored=args.unscored,
        empty_reference=empty_reference,
        local_files_only=args.local_files_only,
        **advanced,
    )


# -- commands ------------------------------------------------------------------
def _text_files_to_jsonl(references: Path, hypotheses: Path, output: Path) -> Path:
    """Pair two text files line by line into ``<output>.input.jsonl`` (ids are
    1-based line numbers). Line endings are normalized and a leading byte-order mark
    is dropped; spaces inside a line are kept."""
    sides = []
    for path in (references, hypotheses):
        if not path.is_file():
            raise UsageError(f"{path} does not exist")
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UsageError(f"{path}: not valid UTF-8 ({exc})") from exc
        sides.append(text.removesuffix("\n").split("\n"))
    refs, hyps = sides
    if len(refs) != len(hyps):
        raise UsageError(f"{references} has {len(refs)} lines and {hypotheses} has {len(hyps)}")
    paired = output.with_name(output.name + ".input.jsonl")
    paired.parent.mkdir(parents=True, exist_ok=True)
    with open(paired, "w", encoding="utf-8") as f:
        for n, (r, h) in enumerate(zip(refs, hyps), 1):
            f.write(json.dumps({"id": n, "reference": r, "hypothesis": h}, ensure_ascii=False) + "\n")
    return paired


def _table_to_jsonl(table: Path, output: Path) -> Path:
    """A CSV or TSV file with ``reference`` and ``hypothesis`` columns (and
    optionally ``id``, ``language``, ``system``, ``document_id``) as ``<output>.input.jsonl``."""
    import csv

    if not table.is_file():
        raise UsageError(f"input {table} does not exist")
    tsv = table.suffix.lower() == ".tsv"
    records = []
    try:
        with open(table, encoding="utf-8-sig", newline="") as src:
            # A TSV field is taken literally: a leading quote must not swallow the tabs after it.
            reader = csv.DictReader(
                src, delimiter="\t" if tsv else ",", quoting=csv.QUOTE_NONE if tsv else csv.QUOTE_MINIMAL
            )
            missing = {"reference", "hypothesis"} - set(reader.fieldnames or ())
            if missing:
                raise UsageError(f"{table} needs the columns {sorted(missing)}")
            for n, row in enumerate(reader, 1):
                if row["reference"] is None or row["hypothesis"] is None:
                    raise UsageError(f"{table}: row {n} is missing a reference or a hypothesis")
                rec = {"id": row.get("id") or n, "reference": row["reference"], "hypothesis": row["hypothesis"]}
                rec.update({k: row[k] for k in ("language", "system", "document_id") if row.get(k)})
                records.append(rec)
    except UnicodeDecodeError as exc:
        raise UsageError(f"{table}: not valid UTF-8 ({exc})") from exc
    paired = output.with_name(output.name + ".input.jsonl")
    paired.parent.mkdir(parents=True, exist_ok=True)
    with open(paired, "w", encoding="utf-8") as dst:
        dst.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
    return paired


def cmd_score(args) -> int:
    if args.resume and args.overwrite:
        raise UsageError("--resume and --overwrite are exclusive")
    if args.output and Path(args.output).exists() and not (args.resume or args.overwrite):
        raise UsageError(f"{args.output} exists; pass --resume to continue it or --overwrite to replace it")
    if args.input and Path(args.input).suffix.lower() in (".csv", ".tsv") and args.output:
        args.input = str(_table_to_jsonl(Path(args.input), Path(args.output)))
    if args.references or args.hypotheses:
        if not (args.references and args.hypotheses and args.output) or args.input:
            raise UsageError("--references and --hypotheses go together, with --output and without --input")
        args.input = str(_text_files_to_jsonl(Path(args.references), Path(args.hypotheses), Path(args.output)))
    if args.input is None:
        if args.reference is None or args.hypothesis is None:
            raise UsageError(
                "give --reference and --hypothesis, --references and --hypotheses FILES, or --input FILE.jsonl"
            )
        scorer = _load_scorer(args, empty_reference="null")
        r = scorer.score(args.reference, args.hypothesis, details=args.details)
        print(r.to_json() if args.json else r.explain(color=_color()))
        return EXIT_OK if r.status != "error" else EXIT_PARTIAL
    if args.output is None:
        raise UsageError("--input needs --output")
    inp, out = Path(args.input), Path(args.output)
    if not inp.is_file():
        raise UsageError(f"input {inp} does not exist")
    manifest_path = _manifest_path(out)

    index = read_jsonl_records(inp)  # validates everything before scoring
    n_total = len(index)
    scorer = _load_scorer(args, empty_reference="null")
    chash = config_hash(scorer.signature, args.details)

    kept: dict[Any, dict] = {}
    if out.exists() and args.resume:
        if not manifest_path.is_file():
            raise UsageError(f"cannot resume: {manifest_path} is missing")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config_hash") != chash:
            raise UsageError(
                "cannot resume: the scoring configuration differs from the one that wrote "
                f"{out} (model, precision, policies or details); use --overwrite or a new output"
            )
        rows, broken = read_output_rows(out)
        want = dict(index)
        for row in rows:
            unchanged = row.get("id") in want and row.get("input_sha256") == want[row.get("id")]
            if unchanged and row.get("status") != "error":  # an error row is scored again
                kept[row["id"]] = row
        if broken or len(kept) != len(rows):
            shutil.copy(out, out.with_name(out.name + ".bak"))
            if broken and not args.quiet:
                print("semcer: recovered an incomplete trailing record", file=sys.stderr)
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                if row.get("id") in kept and kept[row["id"]] is row:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(tmp, out)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")

    manifest = {
        "schema_version": "1",
        "config_hash": chash,
        "details": args.details,
        "signature": scorer.signature,
        "input": str(inp),
        "n_input_records": n_total,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": False,
    }
    _write_json(manifest_path, manifest)

    progress = Progress(n_total, enabled=not args.quiet)
    todo = (rec for rec in iter_records(inp) if rec["id"] not in kept)
    for _ in kept:
        progress.tick(skipped=True)
    interrupted = False
    n_error = sum(1 for r in kept.values() if r.get("status") == "error")
    try:
        with open(out, "a", encoding="utf-8") as f:
            for rec, row in _score_stream(scorer, todo, args.details):
                row["input_sha256"] = input_hash(rec["reference"], rec["hypothesis"])
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
                n_error += row["status"] == "error"
                progress.tick(row["status"])
    except KeyboardInterrupt:
        interrupted = True
        if not args.quiet:
            print("semcer: interrupted; completed rows are kept, rerun with --resume", file=sys.stderr)
    try:
        rows, _ = read_output_rows(out)
    except UsageError:
        rows = []
    from semcer.summary import summarize

    summary = summarize(rows, allow_mixed=True)
    manifest.update(
        completed=not interrupted,
        n_output_rows=len(rows),
        finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    _write_json(manifest_path, manifest)
    if args.summary:
        _write_json(Path(args.summary), summary)
    sem = summary["semcer_micro"]
    if not args.quiet:
        print(
            f"semcer: {len(rows)}/{n_total} rows; status {summary['status_counts']}; "
            f"semcer(micro)={'n/a' if sem is None else f'{sem:.4f}'} over {summary['n_population']} rows",
            file=sys.stderr,
        )
    return EXIT_PARTIAL if (interrupted or n_error or len(rows) < n_total) else EXIT_OK


def _score_stream(scorer, records, details):
    """``(record, row)`` pairs in input order; a failing record becomes an ``error`` row."""
    pending: deque = deque()

    def feed():
        for rec in records:
            pending.append(rec)
            yield rec

    for result in scorer.score_iter(feed(), details=details, on_error="record"):
        yield pending.popleft(), result.to_dict()


def cmd_summarize(args) -> int:
    from semcer.summary import MixedSignatureError, summarize

    rows, broken = read_output_rows(Path(args.input))
    if broken:
        print("semcer: ignoring an incomplete trailing record", file=sys.stderr)
    try:
        summary = summarize(rows, allow_mixed=args.allow_mixed)
    except MixedSignatureError as exc:
        raise UsageError(str(exc)) from exc
    text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return EXIT_OK


def cmd_compare(args) -> int:
    from semcer.summary import compare

    rows_a, rows_b = (read_output_rows(Path(p))[0] for p in (args.a, args.b))
    result = compare(rows_a, rows_b, samples=args.samples, seed=args.seed, top=args.top)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return EXIT_OK
    lo, hi = result["semcer_difference_ci95"]
    print(f"{result['n_lines']} lines scored by both ({result['n_excluded']} excluded)")
    print(f"{'':<12}{'semcer':>10}{'CER':>10}")
    for name, key in (("first", "a"), ("second", "b")):
        print(f"{name:<12}{result[key]['semcer_micro']:>10.4f}{result[key]['cer_micro']:>10.4f}")
    print(
        f"difference  {result['semcer_difference']:>+10.4f}{result['cer_difference']:>+10.4f}"
        f"   semcer 95% interval [{lo:+.4f}, {hi:+.4f}], {result['resampled']} resampled"
    )
    print(
        f"lower semcer on {result['lines_a_better']} lines for the first, "
        f"{result['lines_b_better']} for the second, {result['lines_tied']} tied"
    )
    print("largest differences (id, semcer first, semcer second):")
    for d in result["largest_differences"]:
        print(f"  {d['id']!s:<24}{d['semcer_a']:>8.4f}{d['semcer_b']:>8.4f}")
    return EXIT_OK


def cmd_inspect(args) -> int:
    from semcer.result import PairResult

    rows, _ = read_output_rows(Path(args.input))
    match = [r for r in rows if str(r.get("id")) == str(args.id)]
    if not match:
        raise UsageError(f"id {args.id!r} not found in {args.input}")
    result = PairResult.from_dict(match[0])
    if result.edits is None:
        raise UsageError(
            f"{args.input} was scored with --details scores; inspection needs per-edit detail "
            "(rescore with --details edits)"
        )
    print(result.explain(color=_color()))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="semcer", description="semcer: recoverability-weighted CER.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("score", help="score one pair or a JSONL file")
    p.add_argument("--reference")
    p.add_argument("--hypothesis")
    p.add_argument("--references", help="text file, one reference per line")
    p.add_argument("--hypotheses", help="text file, one hypothesis per line")
    p.add_argument("--input", help="JSONL, CSV or TSV file with reference and hypothesis (and id)")
    p.add_argument("--output", help="JSONL file of results")
    p.add_argument("--summary", help="also write the corpus summary JSON here")
    p.add_argument("--details", default="edits", choices=["edits", "scores", "distributions"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--json", action="store_true", help="single pair: print the JSON record")
    p.add_argument("--quiet", action="store_true", help="no progress or status messages")
    _add_model_args(p)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("summarize", help="corpus summary of a scores JSONL")
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    p.add_argument("--allow-mixed", action="store_true", help="allow rows from different signatures")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("compare", help="compare two systems scored on the same references")
    p.add_argument("a", help="scores JSONL of the first system")
    p.add_argument("b", help="scores JSONL of the second system")
    p.add_argument("--samples", type=int, default=1000, help="bootstrap draws for the interval")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=10, help="lines to list")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("inspect", help="explain one scored pair")
    p.add_argument("--input", required=True)
    p.add_argument("--id", required=True)
    p.set_defaults(func=cmd_inspect)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    from semcer import enable_logging

    enable_logging("WARNING" if getattr(args, "quiet", False) else "INFO")
    try:
        return args.func(args)
    except (UsageError, ValueError, FileNotFoundError) as exc:  # bad input or settings, before any scoring
        print(f"semcer: error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
