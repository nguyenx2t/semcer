"""Corpus aggregation from saved result rows. Torch-free.

Corpus micro scores are ratios of sums, never means of per-pair ratios::

    semcer_micro = sum(severity numerator) / sum(n_ref_graphemes)
    cer_micro    = sum(n_edits)            / sum(n_ref_graphemes)

over one explicit population: rows that carry a semcer value (status ``ok``, or
``partial`` under the ``impute_one`` policy, whose numerator is the effective
sum including the imputed charges). Rows of other statuses are counted, never
silently folded in. Pair-macro means are reported separately. Rows scored under
different signatures are refused unless ``allow_mixed``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

#: Signature keys that do not change a score's identity (execution details).
_NON_IDENTITY = ("conv", "path", "torch", "transformers")


class MixedSignatureError(ValueError):
    """The rows were scored under more than one scoring signature."""


def signature_key(signature: dict) -> str:
    return json.dumps({k: v for k, v in sorted(signature.items()) if k not in _NON_IDENTITY}, sort_keys=True)


def _in_population(row: dict) -> bool:
    return row.get("semcer") is not None and row.get("status") in ("ok", "partial")


def _numerator(row: dict) -> float:
    eff = row.get("effective_severity_sum")
    return eff if eff is not None else row["severity_sum"]


def _block(rows: list[dict]) -> dict[str, Any]:
    pop = [r for r in rows if _in_population(r)]
    n_ref = sum(r["n_ref_graphemes"] for r in pop)
    num = math.fsum(_numerator(r) for r in pop)
    n_edits = sum(r["n_edits"] for r in pop)
    aligned = [r for r in rows if r.get("n_edits") is not None]
    total_edits = sum(r["n_edits"] for r in aligned)
    unscored = sum(r.get("n_unscored") or 0 for r in aligned)
    status = defaultdict(int)
    for r in rows:
        status[r.get("status", "missing")] += 1
    return {
        "n_rows": len(rows),
        "status_counts": dict(sorted(status.items())),
        "pair_completion": (status["ok"] / len(rows)) if rows else None,
        "n_population": len(pop),
        "n_ref_graphemes": n_ref,
        "semcer_micro": num / n_ref if n_ref else None,
        "cer_micro": n_edits / n_ref if n_ref else None,
        "semcer_pair_macro": (math.fsum(r["semcer"] for r in pop) / len(pop)) if pop else None,
        "cer_pair_macro": (math.fsum(r["cer"] for r in pop) / len(pop)) if pop else None,
        "n_edits": total_edits,
        "n_unscored_edits": unscored,
        "edit_coverage": ((total_edits - unscored) / total_edits) if total_edits else None,
        "n_imputed_rows": sum(1 for r in pop if r.get("status") == "partial"),
    }


def summarize(rows: Iterable[dict], *, allow_mixed: bool = False) -> dict[str, Any]:
    """Summary of saved result rows (dicts as written by ``semcer score``)."""
    rows = list(rows)
    sigs: dict[str, dict] = {}
    for r in rows:
        sig = r.get("signature") or {}
        sigs.setdefault(signature_key(sig), sig)
    if len(sigs) > 1 and not allow_mixed:
        raise MixedSignatureError(
            f"rows carry {len(sigs)} different scoring signatures; summarize them separately or pass allow_mixed=True"
        )
    out: dict[str, Any] = {
        "population": "rows with a semcer value: status ok, or partial under unscored=impute_one",
        "aggregation": "micro = ratio of sums over the population; pair_macro = mean of per-pair ratios",
        **_block(rows),
    }
    for key in ("language", "system"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r.get(key) is not None:
                groups[str(r[key])].append(r)
        if groups:
            out[f"by_{key}"] = {g: _block(v) for g, v in sorted(groups.items())}
        if key == "language" and groups:
            macro = [b["semcer_micro"] for b in out["by_language"].values() if b["semcer_micro"] is not None]
            out["semcer_language_macro"] = math.fsum(macro) / len(macro) if macro else None
    out["signatures"] = list(sigs.values())
    return out


def _micro(rows: list[dict]) -> tuple[float, float]:
    n_ref = sum(r["n_ref_graphemes"] for r in rows)
    return math.fsum(_numerator(r) for r in rows) / n_ref, sum(r["n_edits"] for r in rows) / n_ref


def compare(rows_a: Iterable[dict], rows_b: Iterable[dict], *, samples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Compare two systems scored on the same references.

    The rows are matched by ``id``; a line counts only when both systems have a
    semcer value for it, and the references must agree. Returns each system's
    corpus semcer and CER on those lines, the difference ``a - b`` with a 95%
    paired bootstrap interval (lines resampled together, or documents when every
    row carries ``document_id``; each replicate recomputes the ratio of sums),
    and the lines on which the two systems' semcer differs most.
    """
    import random

    a, b = ({r["id"]: r for r in rows} for rows in (rows_a, rows_b))
    if len(a) == 0 or set(a) != set(b):
        raise ValueError(f"the two files do not score the same ids ({len(a)} and {len(b)} rows)")
    ids = [i for i in a if _in_population(a[i]) and _in_population(b[i])]
    for i in ids:
        same_text = a[i].get("reference") == b[i].get("reference")
        if a[i]["n_ref_graphemes"] != b[i]["n_ref_graphemes"] or not same_text:
            raise ValueError(f"id {i!r} has a different reference in the two files")
    if not ids:
        raise ValueError("no line has a semcer value in both files")
    clustered = all(a[i].get("document_id") is not None for i in ids)
    groups: dict[Any, list] = defaultdict(list)
    for i in ids:
        groups[a[i]["document_id"] if clustered else i].append(i)
    units = list(groups.values())

    def diff(lines: list) -> tuple[float, float]:
        (sa, ca), (sb, cb) = _micro([a[i] for i in lines]), _micro([b[i] for i in lines])
        return sa - sb, ca - cb

    rng = random.Random(seed)
    draws = sorted(diff([i for unit in rng.choices(units, k=len(units)) for i in unit])[0] for _ in range(samples))
    (sem_a, cer_a), (sem_b, cer_b) = _micro([a[i] for i in ids]), _micro([b[i] for i in ids])
    lo, hi = (draws[int(q * (samples - 1))] for q in (0.025, 0.975)) if samples else (None, None)
    per_line = sorted(ids, key=lambda i: -abs(a[i]["semcer"] - b[i]["semcer"]))
    return {
        "n_lines": len(ids),
        "n_excluded": len(a) - len(ids),
        "resampled": "documents" if clustered else "lines",
        "a": {"semcer_micro": sem_a, "cer_micro": cer_a},
        "b": {"semcer_micro": sem_b, "cer_micro": cer_b},
        "semcer_difference": sem_a - sem_b,
        "semcer_difference_ci95": [lo, hi],
        "cer_difference": cer_a - cer_b,
        "lines_a_better": sum(a[i]["semcer"] < b[i]["semcer"] for i in ids),
        "lines_b_better": sum(a[i]["semcer"] > b[i]["semcer"] for i in ids),
        "lines_tied": sum(a[i]["semcer"] == b[i]["semcer"] for i in ids),
        "largest_differences": [
            {
                "id": i,
                "semcer_a": a[i]["semcer"],
                "semcer_b": b[i]["semcer"],
                "cer_a": a[i]["cer"],
                "cer_b": b[i]["cer"],
            }
            for i in per_line[:10]
        ],
    }


__all__ = ["MixedSignatureError", "compare", "signature_key", "summarize"]
