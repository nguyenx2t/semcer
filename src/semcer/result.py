"""Per-edit records and per-pair results (torch-free; no inference here).

Statuses
    ``ok``         every edit scored (an exact match is ``ok`` with no edits);
    ``partial``    some edits fell past the model's input window; ``semcer`` is
                   ``None`` unless the ``impute_one`` policy charged each 1.0;
    ``undefined``  empty (NFC) reference: the ratio has no denominator;
    ``error``      the pair was not scored (``cause`` names why, e.g. ``too_large``).

JSON never carries NaN or infinity: undefined metrics are ``null``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1"


def severity_level(severity: float | None) -> str | None:
    """Display level: harmless < 0.25 ≤ partially recoverable ≤ 0.75 < meaning lost."""
    if severity is None:
        return None
    if severity < 0.25:
        return "harmless"
    if severity <= 0.75:
        return "partially recoverable"
    return "meaning lost"


@dataclass
class EditRecord:
    """One aligned grapheme edit.

    ``ref_index``/``hyp_index`` are the aligner's indices (a deletion's
    ``hyp_index`` is a neighbouring index, not a gap). ``ref_span``/``hyp_span``
    are half-open grapheme spans on the NFC strings, empty at the gap for an
    insertion/deletion; ``*_char_span`` are the same spans in code points of the
    NFC text (not of the raw input).
    """

    op: str
    ref_index: int
    hyp_index: int
    ref_grapheme: str | None
    hyp_grapheme: str | None
    ref_span: tuple[int, int]
    hyp_span: tuple[int, int]
    ref_char_span: tuple[int, int]
    hyp_char_span: tuple[int, int]
    severity: float | None  # None when the edit could not be scored
    status: str = "ok"  # "ok" | "truncated" | "zero_token"
    probs: list[float] | None = None  # 5-way digit distribution (details="distributions")
    context: str | None = None

    @property
    def level(self) -> str | None:
        return severity_level(self.severity)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "op": self.op,
            "ref_index": self.ref_index,
            "hyp_index": self.hyp_index,
            "ref_grapheme": self.ref_grapheme,
            "hyp_grapheme": self.hyp_grapheme,
            "ref_span": list(self.ref_span),
            "hyp_span": list(self.hyp_span),
            "ref_char_span": list(self.ref_char_span),
            "hyp_char_span": list(self.hyp_char_span),
            "severity": self.severity,
            "status": self.status,
        }
        if self.probs is not None:
            d["probs"] = list(self.probs)
        if self.context is not None:
            d["context"] = self.context
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EditRecord:
        return cls(
            op=d["op"],
            ref_index=d["ref_index"],
            hyp_index=d["hyp_index"],
            ref_grapheme=d["ref_grapheme"],
            hyp_grapheme=d["hyp_grapheme"],
            ref_span=tuple(d["ref_span"]),
            hyp_span=tuple(d["hyp_span"]),
            ref_char_span=tuple(d["ref_char_span"]),
            hyp_char_span=tuple(d["hyp_char_span"]),
            severity=d["severity"],
            status=d.get("status", "ok"),
            probs=d.get("probs"),
            context=d.get("context"),
        )


@dataclass
class PairResult:
    """Per-pair result with its sufficient statistics and provenance."""

    status: str
    semcer: float | None
    cer: float | None
    severity_sum: float | None  # sum over SCORED edits
    n_edits: int | None
    n_ref_graphemes: int | None
    n_unscored: int = 0
    coverage: float | None = None
    edits: list[EditRecord] | None = None  # None when details="scores"
    signature: dict[str, Any] = field(default_factory=dict)
    id: Any = None
    cause: str | None = None
    unscored_policy: str | None = None
    effective_severity_sum: float | None = None  # severity_sum + imputed charges
    metadata: dict[str, Any] | None = None
    reference: str | None = None  # NFC text, kept only with edit detail
    hypothesis: str | None = None

    # -- derived views -------------------------------------------------------
    @property
    def severities(self) -> list[float | None]:
        """Edit-ordered severity vector (``None`` for an unscored edit)."""
        if self.edits is None:
            raise ValueError("per-edit detail was not kept (details='scores')")
        return [e.severity for e in self.edits]

    @property
    def contributions(self) -> list[float | None]:
        """``severity / n_ref_graphemes`` per edit; sums to ``semcer`` on complete rows."""
        if self.edits is None:
            raise ValueError("per-edit detail was not kept (details='scores')")
        n = self.n_ref_graphemes
        return [None if (e.severity is None or not n) else e.severity / n for e in self.edits]

    @property
    def complete(self) -> bool:
        return self.status == "ok"

    def worst_edits(self, k: int = 5) -> list[EditRecord]:
        """The ``k`` highest-severity scored edits, stable descending; ``k <= 0`` → []."""
        if k <= 0 or not self.edits:
            return []
        scored = [e for e in self.edits if e.severity is not None]
        return sorted(scored, key=lambda e: -e.severity)[:k]

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        if self.id is not None:
            d["id"] = self.id
        if self.metadata:
            d.update({k: v for k, v in self.metadata.items() if k not in d})
        d.update(
            {
                "status": self.status,
                "semcer": _finite(self.semcer),
                "cer": _finite(self.cer),
                "severity_sum": _finite(self.severity_sum),
                "n_edits": self.n_edits,
                "n_ref_graphemes": self.n_ref_graphemes,
                "n_unscored": self.n_unscored,
                "coverage": _finite(self.coverage),
            }
        )
        if self.cause is not None:
            d["cause"] = self.cause
        if self.unscored_policy is not None and self.n_unscored:
            d["unscored_policy"] = self.unscored_policy
        if self.effective_severity_sum is not None:
            d["effective_severity_sum"] = _finite(self.effective_severity_sum)
        if self.edits is not None:
            d["edits"] = [e.to_dict() for e in self.edits]
            if self.reference is not None:
                d["reference"] = self.reference
                d["hypothesis"] = self.hypothesis
        d["signature"] = self.signature
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PairResult:
        known = {
            "schema_version",
            "id",
            "status",
            "semcer",
            "cer",
            "severity_sum",
            "n_edits",
            "n_ref_graphemes",
            "n_unscored",
            "coverage",
            "cause",
            "unscored_policy",
            "effective_severity_sum",
            "edits",
            "signature",
            "reference",
            "hypothesis",
            "input_sha256",
        }
        edits = d.get("edits")
        return cls(
            status=d["status"],
            semcer=d.get("semcer"),
            cer=d.get("cer"),
            severity_sum=d.get("severity_sum"),
            n_edits=d.get("n_edits"),
            n_ref_graphemes=d.get("n_ref_graphemes"),
            n_unscored=d.get("n_unscored", 0),
            coverage=d.get("coverage"),
            edits=None if edits is None else [EditRecord.from_dict(e) for e in edits],
            signature=d.get("signature", {}),
            id=d.get("id"),
            cause=d.get("cause"),
            unscored_policy=d.get("unscored_policy"),
            effective_severity_sum=d.get("effective_severity_sum"),
            metadata={k: v for k, v in d.items() if k not in known} or None,
            reference=d.get("reference"),
            hypothesis=d.get("hypothesis"),
        )

    # -- human-readable views --------------------------------------------------
    def explain(self, k: int | None = None, *, color: bool = False) -> str:
        """Text account of the score and its edits (the ``k`` worst if given).

        ``color=True`` colors each severity by level with ANSI escapes.
        """
        head = f"[{self.id}] " if self.id is not None else ""
        if self.status == "undefined":
            return f"{head}semcer undefined: empty reference ({self.cause or 'no denominator'})"
        if self.status == "error":
            return f"{head}not scored: {self.cause}"
        sem = "n/a" if self.semcer is None else f"{self.semcer:.4f}"
        lines = [
            f"{head}semcer={sem}  CER={self.cer:.4f}  "
            f"({self.n_edits} edit(s) over {self.n_ref_graphemes} reference graphemes; lower is better)"
        ]
        if self.status == "partial":
            how = "each charged 1.0 (impute_one)" if self.unscored_policy == "impute_one" else "semcer withheld"
            lines.append(f"  PARTIAL: {self.n_unscored} edit(s) past the model window were not scored; {how}")
        if self.edits is None:
            return "\n".join(lines)
        edits = self.edits if k is None else self.worst_edits(k)
        for e in edits:
            sev = "unscored" if e.severity is None else f"{e.severity:.2f} ({e.level})"
            if color:
                sev = f"\033[{_ANSI[e.level]}m{sev}\033[0m"
            lines.append(
                f"  {e.op:<3} ref[{e.ref_span[0]}:{e.ref_span[1]}] {e.ref_grapheme!r} -> "
                f"{e.hyp_grapheme!r}  severity {sev}"
            )
        return "\n".join(lines)


_ANSI = {"harmless": "32", "partially recoverable": "33", "meaning lost": "31", None: "2"}


def _finite(x: float | None) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def display_spans(edits, r_graphs: list[str], h_graphs: list[str]):
    """``(ref_span, hyp_span, ref_char_span, hyp_char_span)`` for each aligned edit.

    Stretches between edits are matched diagonals, so a cursor over consumed
    (ref, hyp) graphemes places every gap.
    """
    r_off = [0]
    for g in r_graphs:
        r_off.append(r_off[-1] + len(g))
    h_off = [0]
    for g in h_graphs:
        h_off.append(h_off[-1] + len(g))
    ri = hi = 0
    out = []
    for e in edits:
        hi += e.i - ri  # matched diagonal up to this edit's reference position
        ri = e.i
        if e.op == "SUB":
            rs, hs = (ri, ri + 1), (hi, hi + 1)
            ri, hi = ri + 1, hi + 1
        elif e.op == "DEL":
            rs, hs = (ri, ri + 1), (hi, hi)
            ri += 1
        else:  # INS
            rs, hs = (ri, ri), (hi, hi + 1)
            hi += 1
        out.append((rs, hs, (r_off[rs[0]], r_off[rs[1]]), (h_off[hs[0]], h_off[hs[1]])))
    return out


__all__ = ["SCHEMA_VERSION", "EditRecord", "PairResult", "display_spans", "severity_level"]
