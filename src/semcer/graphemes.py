"""NFC normalization and UAX #29 grapheme cluster splitting."""

from __future__ import annotations

import unicodedata

import regex

_GRAPHEME_RE = regex.compile(r"\X")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def graphemes(s: str) -> list[str]:
    """The extended grapheme clusters of ``nfc(s)``."""
    return _GRAPHEME_RE.findall(nfc(s))
