"""semcer — recoverability-weighted character error rate.

    import semcer

    result = semcer.score(reference="The total is 15", hypothesis="The total is 75")
    result.semcer, result.cer, result.severities
    print(result.explain())

    scorer = semcer.SemCER()                # downloads the released model once
    results = scorer.score_batch(references=refs, hypotheses=hyps)

    semcer.cer(reference, hypothesis, normalization="light")   # model-free

``semcer = sum(per-edit severity) / n_ref_graphemes`` over NFC extended grapheme
clusters, with each edit's severity predicted by a distilled local model. Lower
is better; a severity is graded recoverability, not a probability. A reference
transcription is always required.

Importing this package imports no tensor runtime and makes no network call; the
model loads on first use.
"""

from __future__ import annotations

from typing import Any

from semcer._align import align_graphemes
from semcer.graphemes import graphemes, nfc
from semcer.profiles import PROFILES, UndefinedMetricError, cer, cer_stats, normalize
from semcer.result import EditRecord, PairResult

__all__ = [
    "PROFILES",
    "EditRecord",
    "PairResult",
    "SemCER",
    "UndefinedMetricError",
    "align_graphemes",
    "cer",
    "cer_stats",
    "clear_cache",
    "enable_logging",
    "graphemes",
    "nfc",
    "normalize",
    "score",
]

_DEFAULT: dict[str, Any] = {}


def score(reference: str, hypothesis: str, **kwargs) -> PairResult:
    """Score one pair with the lazily loaded default scorer.

    The default model is the published release bundle (``$SEMCER_MODEL``
    overrides it with a bundle directory or Hub id). It is loaded once and reused;
    :func:`clear_cache` releases it. For other devices or settings build a
    :class:`SemCER` explicitly.
    """
    scorer = _DEFAULT.get("scorer")
    if scorer is None:
        from semcer.metric import SemCER

        scorer = _DEFAULT["scorer"] = SemCER()
    return scorer.score(reference, hypothesis, **kwargs)


def enable_logging(level: str = "INFO") -> None:
    """Print semcer's log messages (model download, device, batch size) to stderr.

    The library is silent by default. It logs to the standard ``"semcer"`` logger,
    so ``logging`` configuration works as well.
    """
    import logging

    logger = logging.getLogger("semcer")
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("semcer: %(message)s"))
        logger.addHandler(handler)


def clear_cache() -> None:
    """Drop the module-level default scorer (and its model memory)."""
    _DEFAULT.clear()


def __getattr__(name: str):  # PEP 562: the torch-backed class loads on first use
    if name == "SemCER":
        from semcer.metric import SemCER

        return SemCER
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
