"""The ``SemCER`` scorer: load once, score pairs.

    scorer = SemCER()                       # downloads the released model once
    result = scorer.score(reference="The total is 15", hypothesis="The total is 75")
    results = scorer.score_batch(references=refs, hypotheses=hyps)
    for result in scorer.score_iter(records):   # dicts with id / reference / hypothesis
        ...

Policies (recorded in every result's signature):

``unscored``        edits whose locus falls past the model window: ``"partial"``
                    (default; ``semcer`` null), ``"impute_one"`` (charge 1.0 each),
                    or ``"raise"``. Both non-raising policies give status ``partial``.
``empty_reference`` ``"raise"`` (default) or ``"null"`` (status ``undefined``,
                    metrics null).

With ``batch_pairs > 1`` (the default), ``score_batch`` and
``score_iter`` prepare that many pairs at a time and share forwards across them
(:meth:`semcer._engine.Engine.score_pairs`); results come back in input order.
The values equal the pair-by-pair ones within the fp32 tolerance (1e-6).
``score`` always scores its one pair on its own.
"""

from __future__ import annotations

import logging
import sys
import unicodedata
from collections.abc import Iterable, Iterator
from typing import Any

from semcer._prep import (
    ALIGNMENT_VERSION,
    DEFAULT_CELL_BUDGET,
    INPUT_TEMPLATE_VERSION,
    PairTooLarge,
    prepare_pair,
)
from semcer.profiles import UndefinedMetricError
from semcer.profiles import cer as _cer
from semcer.result import EditRecord, PairResult, display_spans

__all__ = ["SemCER", "PairResult", "EditRecord", "UnscorableEditError", "UndefinedMetricError"]

logger = logging.getLogger("semcer")

DETAILS = ("edits", "scores", "distributions")
UNSCORED_POLICIES = ("partial", "impute_one", "raise")
EMPTY_REFERENCE_POLICIES = ("raise", "null")
#: Engine settings ``SemCER`` accepts by keyword, with their defaults (``None``:
#: taken from the model or the device).
ADVANCED = {
    "batch_pairs": None,  # lines prepared and scored together (32 on CPU; 64 or 256 on a GPU)
    "max_length": None,  # input window in tokens (the model's own, 2,048)
    "max_batch_tokens": 4096,  # padded tokens per model call (single-line path only)
    "chunk_edits": 256,  # edits tokenized at a time within a line (single-line path only)
    "cache_prefix": True,  # reuse the line's encoding across its edits
    "cache_op_header": True,  # a second cache level per operation (single-line path)
    "cell_budget": DEFAULT_CELL_BUDGET,  # alignment table cells before a line is refused
}


class UnscorableEditError(RuntimeError):
    """An edit could not be scored under ``unscored="raise"``."""


def _gpu_batches(device: str) -> tuple[int, int]:
    """``(batch_size, batch_pairs)`` for a GPU, by its memory. Measured peaks on
    short lines: 11 GB at (16, 64), 13 GB at (32, 64), 18 GB at (64, 256)."""
    import torch

    if not str(device).startswith("cuda"):
        return 16, 64
    gb = torch.cuda.get_device_properties(torch.device(device)).total_memory / 2**30
    return (64, 256) if gb >= 32 else (32, 64) if gb >= 24 else (16, 64)


def _dist_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover
        return None


class SemCER:
    """Score ``(reference, hypothesis)`` pairs into per-edit severities and semcer."""

    def __init__(
        self,
        model: str | None = None,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "fp32",
        batch_size: int | None = None,
        unscored: str = "partial",
        empty_reference: str = "raise",
        local_files_only: bool = False,
        **advanced,
    ):
        """Load the model and get ready to score.

        model             a bundle directory or a Hub id; with none, ``$SEMCER_MODEL``
                          or the released model, downloaded once and cached
        revision          Hub commit, branch or tag of ``model``
        device            ``"auto"`` (CUDA when available, else CPU), ``"cpu"``, ``"cuda[:N]"``
        dtype             ``"fp32"`` or ``"bf16"``
        batch_size        edits per model call (default 8 on CPU; 16 to 64 on a GPU
                          by its memory); lower it if a GPU runs out of memory
        unscored          ``"partial"``, ``"impute_one"`` or ``"raise"`` (module doc)
        empty_reference   ``"raise"`` or ``"null"``
        local_files_only  never touch the network

        Engine settings that rarely need changing are accepted by keyword too:
        see :data:`ADVANCED`.
        """
        unknown = sorted(set(advanced) - set(ADVANCED))
        if unknown:
            raise TypeError(f"unexpected argument(s) {unknown}; advanced settings are {sorted(ADVANCED)}")
        adv = {**ADVANCED, **advanced}
        max_length, batch_pairs, cell_budget = adv["max_length"], adv["batch_pairs"], adv["cell_budget"]
        from semcer._convpatch import conv_path, gdn_path
        from semcer._engine import Engine, EngineConfig
        from semcer._loader import load_bundle, resolve_device

        if unscored not in UNSCORED_POLICIES:
            raise ValueError(f"unscored must be one of {UNSCORED_POLICIES}, got {unscored!r}")
        if empty_reference not in EMPTY_REFERENCE_POLICIES:
            raise ValueError(f"empty_reference must be one of {EMPTY_REFERENCE_POLICIES}, got {empty_reference!r}")
        device = resolve_device(device)
        bundle = load_bundle(model, revision=revision, device=device, dtype=dtype, local_files_only=local_files_only)
        model, tokenizer, meta = bundle.model, bundle.tokenizer, bundle.meta
        source, revision = bundle.source, bundle.revision
        if max_length is None:
            max_length = meta["max_length"]
        self._model = model
        self._tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_length = int(max_length)
        self.unscored = unscored
        self.empty_reference = empty_reference
        self.cell_budget = cell_budget
        on_cpu = str(device).startswith("cpu")
        if batch_size is None:
            # Larger CPU batches spill the caches (each row carries recurrent state).
            batch_size = 8 if on_cpu else _gpu_batches(device)[0]
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = int(batch_size)
        if batch_pairs is None:
            # Sharing model calls across lines is 1.8x faster on CPU (7.1 to 13.0
            # edits/s, Apple M-series) and moves a severity by at most 7e-7.
            batch_pairs = 32 if on_cpu else _gpu_batches(device)[1]
        if batch_pairs < 1:
            raise ValueError(f"batch_pairs must be >= 1, got {batch_pairs}")
        self.batch_pairs = int(batch_pairs)
        self._engine = Engine(
            model,
            tokenizer,
            device,
            EngineConfig(
                max_length=self.max_length,
                max_batch_tokens=adv["max_batch_tokens"],
                max_batch_edits=self.batch_size,
                chunk_edits=adv["chunk_edits"],
                cache_prefix=adv["cache_prefix"],
                cache_op_header=adv["cache_op_header"],
            ),
        )
        import torch
        import transformers

        sha = meta.get("source_checkpoint_sha256")
        self.signature: dict[str, Any] = {
            "semcer": _dist_version("semcer") or "0+unknown",
            "model": source,
            "revision": revision,
            "weights": sha[:16] if sha else None,
            "dev_checkpoint": bool(meta.get("dev_checkpoint", False)),
            "label": meta.get("label"),
            "head": meta.get("head_type", getattr(model, "head_type", None)),
            "readout": meta.get("readout", getattr(model, "readout", None)),
            "max_length": self.max_length,
            "template": INPUT_TEMPLATE_VERSION,
            "alignment": ALIGNMENT_VERSION,
            "device": str(device),
            "precision": dtype,
            "conv": conv_path(device),
            "gdn": gdn_path(device),
            "path": self._engine.path + ("+cross_pair" if self._cross_pair else ""),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            # Versions that define the text units: NFC, grapheme clusters, tokens.
            "tokenizers": _dist_version("tokenizers"),
            "regex": _dist_version("regex"),
            "python": sys.version.split()[0],
            "unicode": unicodedata.unidata_version,
            "unscored": unscored,
            "empty_reference": empty_reference,
        }

        logger.info(
            "model %s on %s (%s), batch size %d, %d line(s) per window",
            source,
            device,
            dtype,
            self.batch_size,
            self.batch_pairs,
        )

    @property
    def _cross_pair(self) -> bool:
        return self.batch_pairs > 1 and self._engine.path == "prefix_cached"

    @classmethod
    def from_pretrained(cls, model: str | None = None, **kwargs) -> SemCER:
        """Same as ``SemCER(model, **kwargs)``."""
        return cls(model, **kwargs)

    # -- scoring ----------------------------------------------------------------
    def score(
        self,
        reference: str,
        hypothesis: str,
        *,
        id: Any = None,
        language: str | None = None,
        details: str = "edits",
    ) -> PairResult:
        """Score one pair."""
        if details not in DETAILS:
            raise ValueError(f"details must be one of {DETAILS}, got {details!r}")
        rec = {"id": id, "reference": reference, "hypothesis": hypothesis}
        if language is not None:
            rec["language"] = language
        return self._score_record(rec, details=details)

    def score_batch(
        self,
        references: list[str],
        hypotheses: list[str],
        *,
        ids: list | None = None,
        languages: list[str | None] | None = None,
        details: str = "edits",
        progress: bool = False,
    ) -> list[PairResult]:
        """Score positional pairs; results come back in input order.

        ``progress=True`` shows a progress bar.
        """
        if len(references) != len(hypotheses):
            raise ValueError(f"references and hypotheses differ in length: {len(references)} != {len(hypotheses)}")
        for name, seq in (("ids", ids), ("languages", languages)):
            if seq is not None and len(seq) != len(references):
                raise ValueError(f"{name} has {len(seq)} entries for {len(references)} pairs")
        records = []
        for n, (r, h) in enumerate(zip(references, hypotheses)):
            rec = {"id": ids[n] if ids is not None else None, "reference": r, "hypothesis": h}
            if languages is not None and languages[n] is not None:
                rec["language"] = languages[n]
            records.append(rec)
        results = self.score_iter(records, details=details)
        if progress:
            from tqdm.auto import tqdm

            results = tqdm(results, total=len(records), desc="semcer", unit="line")
        return list(results)

    def score_iter(
        self, records: Iterable[dict], *, details: str = "edits", on_error: str = "raise"
    ) -> Iterator[PairResult]:
        """One result per record (``reference``, ``hypothesis``; optionally ``id``,
        ``language``, ``document_id``, ``system``, ``metadata``), in input order.

        ``on_error="record"`` turns a record's failure into a status ``error``
        result naming the cause instead of raising.
        """
        if details not in DETAILS:
            raise ValueError(f"details must be one of {DETAILS}, got {details!r}")
        if on_error not in ("raise", "record"):
            raise ValueError(f"on_error must be 'raise' or 'record', got {on_error!r}")
        if self._cross_pair:
            yield from self._score_windows(records, details, on_error)
            return
        for rec in records:
            try:
                yield self._score_record(rec, details=details)
            except Exception as exc:
                if on_error == "raise":
                    raise
                yield self._error(rec, f"{type(exc).__name__}: {exc}")

    #: Edits prepared ahead of the model on the cross-pair path (bounds memory
    #: when a window holds runaway pairs).
    _WINDOW_EDITS = 8192

    def _score_windows(self, records, details: str, on_error: str) -> Iterator[PairResult]:
        """The cross-pair path: prepare up to ``batch_pairs`` pairs, score them
        together, and yield their results in input order."""
        window: list = []  # (record, PreparedPair | PairResult | Exception)
        n_edits = 0
        for rec in records:
            try:
                item = self._prepare(rec)
            except Exception as exc:
                item = exc
            window.append((rec, item))
            failed = isinstance(item, Exception) and on_error == "raise"
            n_edits += len(getattr(item, "edits", None) or ())
            if failed or len(window) >= self.batch_pairs or n_edits >= self._WINDOW_EDITS:
                yield from self._flush(window, details, on_error)
                window, n_edits = [], 0
        yield from self._flush(window, details, on_error)

    def _flush(self, window: list, details: str, on_error: str) -> Iterator[PairResult]:
        from semcer._prep import PreparedPair

        preps = [item for _, item in window if isinstance(item, PreparedPair)]
        try:
            scored = iter(self._engine.score_pairs(preps) if preps else ())
        except Exception as exc:
            # One pair's failure (or a batch too large for the device) must not
            # take its neighbours with it: redo the window one pair at a time.
            scored = None
            self._engine.stats.window_fallbacks += 1
            logger.warning("a batch of lines failed (%s); scoring them one by one", type(exc).__name__)
            if str(self.device).startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
        for rec, item in window:
            try:
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, PairResult):
                    yield item
                    continue
                scores = next(scored) if scored is not None else self._engine.score_pair(item)
                yield self._assemble(rec, item, scores, details)
            except Exception as exc:
                if on_error == "raise":
                    raise
                yield self._error(rec, f"{type(exc).__name__}: {exc}")

    def _error(self, rec, cause: str) -> PairResult:
        rec = rec if isinstance(rec, dict) else {}
        return self._unscored_result(rec, "error", cause)

    # -- one pair -------------------------------------------------------------
    def _result(self, rec, **fields) -> PairResult:
        meta = {k: rec[k] for k in ("language", "document_id", "system") if rec.get(k) is not None}
        if isinstance(rec.get("metadata"), dict):
            meta["metadata"] = rec["metadata"]
        return PairResult(id=rec.get("id"), metadata=meta or None, **fields)

    def _prepare(self, rec: dict):
        """The aligned pair for ``rec``, or its finished result when no model run
        is needed (too large to align, or an empty reference under ``"null"``)."""
        _check_record(rec)
        try:
            prep = prepare_pair(rec["reference"], rec["hypothesis"], cell_budget=self.cell_budget)
        except PairTooLarge as exc:
            return self._too_large(rec, exc.n_ref, exc.n_hyp, exc.budget)
        if prep.n_ref == 0:
            return self._empty_reference(rec)
        return prep

    def _score_record(self, rec: dict, *, details: str) -> PairResult:
        prep = self._prepare(rec)
        if isinstance(prep, PairResult):
            return prep
        return self._assemble(rec, prep, self._engine.score_pair(prep), details)

    def _assemble(self, rec: dict, prep, scores, details: str) -> PairResult:
        sig = dict(self.signature)
        n_ref = prep.n_ref
        sevs = [s.severity for s in scores]
        if self.unscored == "raise":
            for k, s in enumerate(scores):
                if s.severity is None:
                    e = prep.edits[k]
                    raise UnscorableEditError(
                        f"pair {rec.get('id')!r}: edit {k} ({e.op} at ref {e.i}) could not be "
                        f"scored: locus {s.status} (max_length={self.max_length})"
                    )

        from semcer._engine import pair_numbers

        nums = pair_numbers(sevs, n_ref, self.unscored)
        edits = None
        if details != "scores":
            spans = display_spans(prep.edits, prep.r_graphs, prep.h_graphs)
            edits = [
                EditRecord(
                    op=e.op,
                    ref_index=e.i,
                    hyp_index=e.j,
                    ref_grapheme=e.g_r,
                    hyp_grapheme=e.g_h,
                    ref_span=sp[0],
                    hyp_span=sp[1],
                    ref_char_span=sp[2],
                    hyp_char_span=sp[3],
                    severity=sev,
                    status=s.status,
                    probs=s.probs if details == "distributions" else None,
                    context=s.context,
                )
                for e, sp, sev, s in zip(prep.edits, spans, sevs, scores)
            ]
        return self._result(
            rec,
            status=nums["status"],
            semcer=nums["semcer"],
            cer=len(prep.edits) / n_ref,
            severity_sum=nums["severity_sum"],
            n_edits=len(prep.edits),
            n_ref_graphemes=n_ref,
            n_unscored=nums["n_unscored"],
            coverage=nums["coverage"],
            edits=edits,
            signature=sig,
            unscored_policy=self.unscored,
            effective_severity_sum=nums["effective_severity_sum"],
            reference=prep.reference if edits is not None else None,
            hypothesis=prep.hypothesis if edits is not None else None,
        )

    def _unscored_result(self, rec, status, cause, *, n_ref=None) -> PairResult:
        """A result with no scored edits (error or undefined)."""
        return self._result(
            rec,
            status=status,
            semcer=None,
            cer=None,
            severity_sum=None,
            n_edits=None,
            n_ref_graphemes=n_ref,
            coverage=None,
            signature=dict(self.signature),
            cause=cause,
        )

    def _too_large(self, rec, n_ref, n_hyp, budget) -> PairResult:
        cause = f"too_large: {n_ref}x{n_hyp} graphemes exceeds cell_budget {budget}"
        return self._unscored_result(rec, "error", cause, n_ref=n_ref)

    def _empty_reference(self, rec) -> PairResult:
        if self.empty_reference == "raise":
            raise UndefinedMetricError(
                f"pair {rec.get('id')!r}: semcer is undefined for an empty reference "
                "(use empty_reference='null' to get a record instead)"
            )
        return self._unscored_result(rec, "undefined", "empty_reference", n_ref=0)

    # -- model-free helpers -----------------------------------------------------
    def cer(self, reference: str, hypothesis: str, normalization: str = "raw") -> float:
        """Grapheme CER under a named profile (model-free; raises on an empty reference)."""
        return _cer(reference, hypothesis, normalization)

    @property
    def stats(self):
        """Engine counters (forwards, cached/uncached rows, padding)."""
        return self._engine.stats

    def signature_string(self) -> str:
        s = self.signature
        model = s["model"] or "custom"
        rev = f"@{s['revision'][:12]}" if s.get("revision") else ""
        dev = "+DEV" if s["dev_checkpoint"] else ""
        return (
            f"semcer:{s['semcer']}|model:{model}{rev}{dev}|w:{s['weights']}|tpl:{s['template']}"
            f"|align:{s['alignment']}|{s['device']}:{s['precision']}|unscored:{s['unscored']}"
        )


def _check_record(rec: dict) -> None:
    if not isinstance(rec, dict):
        raise TypeError(f"records must be dicts, got {type(rec).__name__}")
    for key in ("reference", "hypothesis"):
        if key not in rec:
            raise KeyError(f"record {rec.get('id')!r} has no {key!r}")
        if not isinstance(rec[key], str):
            raise TypeError(f"record {rec.get('id')!r}: {key} must be str, got {type(rec[key]).__name__}")
