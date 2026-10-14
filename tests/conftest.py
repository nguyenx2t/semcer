"""Shared fixtures: the committed tiny bundle, a scorer on it, synthetic pairs, and
the reference implementation's frozen outputs on that bundle (``tests/fixtures``).

The tiny bundle is a few-layer random-init Qwen3.5 text model (Gated DeltaNet +
attention) with a random head and a small byte-level BPE tokenizer; everything runs
offline. Real-size checks are opt-in (``tests/test_t1_real.py``). Tests assert
parity and invariants, never quality numbers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(Path(__file__).resolve().parent))  # test helpers

from _pairs import make_pairs  # noqa: E402

if os.environ.get("PYTEST_XDIST_WORKER"):
    # One torch thread per xdist worker: N workers x all-core BLAS oversubscribes
    # the CPU and makes the suite many times slower.
    import torch

    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture(scope="session")
def tiny_bundle() -> Path:
    """A tiny random-weight Qwen3.5 bundle (committed; tests must not modify it)."""
    return FIXTURES / "tiny-bundle"


@pytest.fixture(scope="session")
def scorer(tiny_bundle):
    from semcer import SemCER

    return SemCER(tiny_bundle, batch_pairs=1)  # the single-line path; tests/test_cross_pair.py covers the default


@pytest.fixture(scope="session")
def pairs():
    """48 synthetic pairs: 3 per language, all 16 languages."""
    return make_pairs(48)


def load_golden(name: str) -> dict:
    """A JSON of the reference implementation's outputs on ``tiny_bundle``."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _quiet_semcer_logger():
    """The CLI turns logging on for the process; start every test from the library's silent default."""
    import logging

    logger = logging.getLogger("semcer")
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    yield
