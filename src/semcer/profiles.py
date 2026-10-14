"""Model-free CER with named, versioned normalization profiles.

``cer(reference, hypothesis)`` with profile ``raw`` is NFC grapheme CER on
semcer's own alignment: the same edits and denominator as ``result.cer``. The
other profiles transform **both** strings and re-align, so their edits and
denominators differ; they never touch the severity model's input.

==========  ==================================================================
profile     transformation (after NFC)
==========  ==================================================================
``raw``     nothing
``ws``      drop every whitespace character (``str.isspace``); case kept
``light``   casefold; whitespace runs -> one space; strip ends
``omni2025`` drop every char whose Unicode category starts with P/S/Z, or that
            is whitespace; case kept
``full``    casefold, then the ``omni2025`` drop
==========  ==================================================================

Importing this module imports no tensor runtime.
"""

from __future__ import annotations

import re
import unicodedata

from semcer._align import align_grapheme_lists
from semcer.graphemes import graphemes, nfc

PROFILES: tuple[str, ...] = ("raw", "ws", "light", "omni2025", "full")
PROFILE_VERSION = "1"

_WHITESPACE_RUN = re.compile(r"\s+")


class UndefinedMetricError(ValueError):
    """The (normalized) reference is empty, so an error *rate* has no denominator."""


def _drop_punct_symbol_space(text: str) -> str:
    return "".join(c for c in text if not (unicodedata.category(c)[0] in "PSZ" or c.isspace()))


def normalize(text: str, profile: str = "raw") -> str:
    """Apply a named profile (NFC first). Unknown names raise ``ValueError``."""
    s = nfc(text)
    if profile == "raw":
        return s
    if profile == "ws":
        return "".join(c for c in s if not c.isspace())
    if profile == "light":
        return _WHITESPACE_RUN.sub(" ", s.casefold()).strip()
    if profile == "omni2025":
        return _drop_punct_symbol_space(s)
    if profile == "full":
        return _drop_punct_symbol_space(s.casefold())
    raise ValueError(f"unknown normalization profile {profile!r}; expected one of {PROFILES}")


def cer_stats(reference: str, hypothesis: str, normalization: str = "raw") -> dict:
    """Sufficient statistics for one pair under a profile.

    ``{"profile", "profile_version", "n_edits", "n_ref_graphemes", "cer"}``;
    ``cer`` is ``None`` (and nothing raises) when the normalized reference is empty.
    """
    for name, value in (("reference", reference), ("hypothesis", hypothesis)):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be str, got {type(value).__name__}")
    r = graphemes(normalize(reference, normalization))
    h = graphemes(normalize(hypothesis, normalization))
    n_edits = len(align_grapheme_lists(r, h)) if r != h else 0
    return {
        "profile": normalization,
        "profile_version": PROFILE_VERSION,
        "n_edits": n_edits,
        "n_ref_graphemes": len(r),
        "cer": n_edits / len(r) if r else None,
    }


def cer(reference: str, hypothesis: str, normalization: str = "raw") -> float:
    """Grapheme CER under ``normalization``; raises :class:`UndefinedMetricError`
    on an empty normalized reference."""
    stats = cer_stats(reference, hypothesis, normalization)
    if stats["cer"] is None:
        raise UndefinedMetricError(f"CER is undefined: the reference is empty after {normalization!r} normalization")
    return stats["cer"]


__all__ = ["PROFILES", "PROFILE_VERSION", "UndefinedMetricError", "cer", "cer_stats", "normalize"]
