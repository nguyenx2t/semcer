"""Named CER profiles reproduce the reference normalizers (frozen in ``tests/fixtures/profiles.json``)."""

from __future__ import annotations

import pytest
from conftest import load_golden

import semcer
from semcer.profiles import PROFILES, UndefinedMetricError, cer_stats, normalize


def test_profiles_match_reference_normalized_cer():
    gold = load_golden("profiles.json")
    assert tuple(gold["levels"]) == PROFILES
    assert len(gold["pairs"]) >= 48
    for (r, h), per_level in zip(gold["pairs"], gold["out"]):
        for level, (nr, nh, c) in zip(PROFILES, per_level):
            assert normalize(r, level) == nr and normalize(h, level) == nh
            assert cer_stats(r, h, level)["cer"] == c


def test_cer_raises_on_empty_normalized_reference():
    with pytest.raises(UndefinedMetricError):
        semcer.cer("!!!", "abc", normalization="full")
    assert cer_stats("!!!", "abc", "full")["cer"] is None
    with pytest.raises(ValueError, match="unknown normalization"):
        semcer.cer("a", "b", normalization="nfkc")
    with pytest.raises(TypeError):
        semcer.cer(None, "b")
