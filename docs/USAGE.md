# Usage guide

Install and a first score are in the [README](../README.md). This page covers
the rest.

## Results

`score`, `score_batch` and `score_iter` return `PairResult` objects.

| Field | Meaning |
|---|---|
| `semcer`, `cer` | The pair's scores. `0 <= semcer <= cer` |
| `status` | `ok`, `partial` (some edits were not scored), `undefined` (empty reference) or `error` |
| `edits` | One record per edit: `op` (SUB, DEL, INS), the two graphemes, their positions, `severity`, `level` |
| `severities` | One value per edit, in alignment order |
| `contributions` | `severity / n_ref_graphemes` per edit; sums to `semcer` |
| `severity_sum`, `n_edits`, `n_ref_graphemes`, `n_unscored` | Sufficient statistics for aggregating a corpus |
| `signature` | What produced the number (see Reproducibility) |

`result.explain()` prints the edits, `result.worst_edits(k)` returns the `k` most
severe, and `result.to_dict()` / `to_json()` serialize losslessly.

A severity is graded recoverability, not a probability. Read it as one of three
levels: harmless below 0.25, partially recoverable from 0.25 to 0.75, meaning
lost above 0.75. The model does not resolve differences below about 0.1.

Pass the texts by name. Swapping them gives a wrong number without an error,
because the denominator is the reference length.

### Unscored edits

The model reads up to 2,048 tokens. An edit that falls past that window is not
scored: its severity is `None`, the pair's status is `partial`, and `semcer` is
`None`. Choose another policy when you build the scorer:

```python
semcer.SemCER(unscored="impute_one")   # charge each unscored edit 1.0
semcer.SemCER(unscored="raise")        # stop instead
```

An empty reference raises by default; `empty_reference="null"` returns a result
with status `undefined`.

## Scoring a corpus

```python
scorer = semcer.SemCER()
results = scorer.score_batch(references=refs, hypotheses=hyps, ids=ids)

for result in scorer.score_iter(records, on_error="record"):   # dicts with id, reference, hypothesis
    ...
```

`score_iter` streams with bounded memory and returns results in input order.
With `on_error="record"` a failing record becomes a result with status `error`
instead of stopping the run. `details="scores"` drops the per-edit records and
`details="distributions"` adds the model's five rubric probabilities per edit.

### Command line

```bash
semcer score --references refs.txt --hypotheses hyps.txt --output scores.jsonl
semcer score --input pairs.csv --output scores.jsonl
semcer score --input pairs.jsonl --output scores.jsonl --summary summary.json
semcer score --input pairs.jsonl --output scores.jsonl --resume
semcer summarize --input scores.jsonl
semcer inspect --input scores.jsonl --id line-0042
semcer score --reference "The total is 15" --hypothesis "The total is 75"
```

With two text files, the row with `id` n scores line n of each file. A CSV or
TSV file needs `reference` and `hypothesis` columns. A JSONL input needs `id`,
`reference` and `hypothesis` on each line; `language`, `system`,
`document_id` and `metadata` are carried through, and the summary breaks scores
down by language and system. `--resume` keeps every finished row whose text has
not changed and refuses to continue under a different model or settings. The
exit code is 0 when every row was scored, 1 when the run was interrupted or some
rows are errors, and 2 for invalid input.

Corpus scores are ratios of sums, `sum(severity_sum) / sum(n_ref_graphemes)`,
not means of per-line scores. The summary reports both.

## Comparing two systems

Score each system's output against the same references, then:

```bash
semcer compare system_a.jsonl system_b.jsonl
```

This prints each system's corpus semcer and CER, their difference with a 95%
paired bootstrap interval, how many lines each system wins, and the lines where
they differ most. Lines are resampled together, or documents when the rows
carry `document_id`. `--json` prints the full result, and
`semcer.summary.compare(rows_a, rows_b)` does the same from Python.

## CER without the model

```python
semcer.cer(reference, hypothesis)                          # NFC grapheme CER
semcer.cer(reference, hypothesis, normalization="light")
```

| Profile | Applied to both texts before aligning |
|---|---|
| `raw` | nothing |
| `ws` | whitespace removed |
| `light` | casefold, whitespace runs collapsed |
| `omni2025` | punctuation, symbols and whitespace removed |
| `full` | `omni2025` plus casefold |

These load no model and import no tensor library.

## Devices

`SemCER(device=...)` and `--device` take `"auto"` (the default: CUDA when
available, else CPU), `"cpu"` or `"cuda"`.

- **Batch size.** `SemCER(batch_size=...)` or `--batch-size` sets the number of
  edits per model call. Defaults follow the card's memory: 16 edits and 64
  lines below 24 GB, 32 and 64 from 24 GB, 64 and 256 from 32 GB (18 GB peak on
  short lines). On CPU they are 8 and 32. Lower `batch_size` if a GPU runs out
  of memory.
- **Precision.** `dtype="bf16"` is about twice as fast on an A100 (374 against
  193 edits per second on 1,000 lines), halves the download, and moves a
  severity by up to about 0.006. fp32 is the default because it gives the same
  scores on every device.
- **Lines share model calls.** `score_batch`, `score_iter` and the CLI prepare
  `batch_pairs` lines at a time and batch their edits together, so pass a
  corpus to them rather than calling `score` per line. Which lines share a call
  moves a severity by less than 1e-06; `batch_pairs=1` scores each line on its
  own, as `score` always does.

## Messages and progress

The library prints nothing by default. To see what it is doing (model download,
device, batch size):

```python
semcer.enable_logging()                       # or configure the "semcer" logger yourself
scorer.score_batch(refs, hyps, progress=True) # a progress bar
```

The CLI shows status and progress by default; `--quiet` turns them off.

In Jupyter or Colab a result displays as the two texts with each edit coloured
by its level; `result.to_html()` returns that HTML.

## Models and offline use

With no argument the scorer uses the released model at a pinned revision,
downloads it to the Hugging Face cache once, and reads the cache afterwards.

```python
semcer.SemCER("path/to/bundle")                  # a local copy
semcer.SemCER(local_files_only=True)             # never touch the network
```

`hf download nguyenx2t/semcer-qwen3.5-0.8b --local-dir ./semcer-model` makes a
local copy, and the environment variable `SEMCER_MODEL` sets the default to a
directory or a Hub id.

## Reproducibility

Every result carries a signature: package version, model revision and weights
hash, input template and alignment versions, device, precision, and the versions
of torch, transformers, tokenizers and regex. transformers, tokenizers and regex
are pinned exactly, because each can change a score; torch 2.11 to 2.13 give
identical scores. Report the signature with your numbers:

```python
scorer.signature_string()
```

## Development

```bash
uv venv && uv pip install -e '.[test]'
python -m pytest -n auto tests
ruff check src tools tests
```

The tests run offline on a tiny random-weight model in `tests/fixtures` and
check parity and invariants, not quality. `tools/bench.py` measures speed and
parity on a device, and `tools/export_bundle.py` turns a training checkpoint
into a model bundle.
