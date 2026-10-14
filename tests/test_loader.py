"""Bundle loading: layout and checksums, meta validation before any weight is
read, offline loading, dtype, the default-model and environment overrides, and
the provenance the signature carries."""

from __future__ import annotations

import hashlib
import json
import shutil
import socket

import pytest
import torch

from semcer import SemCER, _loader
from semcer._loader import BundleError


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_layout_and_checksums(tiny_bundle):
    names = {p.name for p in tiny_bundle.iterdir()}
    for f in (
        "model.safetensors",
        "model.bf16.safetensors",
        "config.json",
        "tokenizer.json",
        "semcer_meta.json",
        "LICENSE",
        "NOTICE",
        "README.md",
        "SHA256SUMS",
    ):
        assert f in names
    meta = json.loads((tiny_bundle / "semcer_meta.json").read_text())
    assert meta["bundle_version"] == 1 and meta["dev_checkpoint"] is True
    sums = dict(line.split("  ")[::-1] for line in (tiny_bundle / "SHA256SUMS").read_text().splitlines())
    for name, digest in sums.items():
        assert _sha256(tiny_bundle / name) == digest
    assert json.loads((tiny_bundle / "config.json").read_text())["model_type"] == "qwen3_5_text"


def _copy_bundle(src, dst, **meta_changes):
    shutil.copytree(src, dst)
    meta = json.loads((dst / "semcer_meta.json").read_text())
    meta.update(meta_changes)
    (dst / "semcer_meta.json").write_text(json.dumps(meta))
    return dst


@pytest.mark.parametrize(
    "change,match",
    [
        ({"bundle_version": 99}, "bundle_version"),
        ({"readout": "span3_mean"}, "readout"),
        ({"head_type": "ordinal"}, "head_type"),
        ({"digit_order": [4, 3, 2, 1, 0]}, "digit_order"),
    ],
)
def test_unknown_meta_raises_before_weights_load(tiny_bundle, tmp_path, monkeypatch, change, match):
    b = _copy_bundle(tiny_bundle, tmp_path / "b", **change)
    import safetensors.torch

    def boom(*a, **k):
        raise AssertionError("weights were read")

    monkeypatch.setattr(safetensors.torch, "load_file", boom)
    with pytest.raises(BundleError, match=match):
        SemCER.from_pretrained(b)


def test_missing_meta_and_missing_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(BundleError, match="not a semcer bundle"):
        SemCER.from_pretrained(tmp_path / "empty")
    with pytest.raises(BundleError, match="does not exist"):
        SemCER.from_pretrained(str(tmp_path / "nope"))


def test_no_default_model_yet_gives_a_clear_message(monkeypatch):
    monkeypatch.delenv("SEMCER_MODEL", raising=False)
    monkeypatch.setattr(_loader, "DEFAULT_MODEL_ID", None)
    with pytest.raises(BundleError, match="no default semcer model"):
        SemCER.from_pretrained()


def test_env_override_selects_the_bundle(tiny_bundle, monkeypatch):
    monkeypatch.setenv("SEMCER_MODEL", str(tiny_bundle))
    s = SemCER.from_pretrained()
    assert s.max_length == 2048


def test_loads_and_scores_with_the_network_disabled(tiny_bundle, monkeypatch):
    def no_network(*a, **k):
        raise OSError("network disabled in this test")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    s = SemCER.from_pretrained(tiny_bundle, local_files_only=True)
    assert s.score("abc", "abd").status == "ok"


def test_max_length_comes_from_meta_and_can_be_overridden(tiny_bundle):
    assert SemCER.from_pretrained(tiny_bundle).max_length == 2048
    assert SemCER.from_pretrained(tiny_bundle, max_length=512).max_length == 512


def test_bf16_bundle_loads(tiny_bundle):
    s = SemCER.from_pretrained(tiny_bundle, dtype="bf16")
    assert next(s._model.parameters()).dtype == torch.bfloat16
    r = s.score("The total is 15", "The total is 75")
    assert r.signature["precision"] == "bf16" and 0.0 <= r.edits[0].severity <= 1.0


def test_bad_dtype_raises_before_loading(tiny_bundle):
    with pytest.raises(ValueError, match="dtype"):
        SemCER.from_pretrained(tiny_bundle, dtype="fp16")


def test_signature_carries_provenance(scorer, tiny_bundle):
    sig = scorer.signature
    meta = json.loads((tiny_bundle / "semcer_meta.json").read_text())
    assert sig["weights"] == meta["source_checkpoint_sha256"][:16]
    assert sig["dev_checkpoint"] is True
    assert sig["template"] == meta["input_template_version"]
    assert sig["alignment"] == meta["alignment_version"]
    assert sig["precision"] == "fp32" and sig["device"] == "cpu"
    assert "+DEV" in scorer.signature_string()


def test_signature_records_the_versions_that_can_change_scores(scorer):
    import sys
    import unicodedata
    from importlib.metadata import version

    sig = scorer.signature
    assert sig["python"] == sys.version.split()[0]
    assert sig["unicode"] == unicodedata.unidata_version  # NFC
    assert sig["regex"] == version("regex")  # grapheme clusters
    assert sig["tokenizers"] == version("tokenizers")
    assert sig["torch"] and sig["transformers"]


def test_label_reaches_the_signature(tiny_bundle, tmp_path):
    b = _copy_bundle(tiny_bundle, tmp_path / "b", label="example label, not a release")
    s = SemCER.from_pretrained(b)
    assert s.score("abc", "abd").signature["label"] == "example label, not a release"


def test_first_download_is_logged_once_and_nothing_is_printed(tiny_bundle, monkeypatch, capsys, caplog):
    import huggingface_hub

    calls = []

    def fake(local_files_only=False, **kwargs):
        calls.append(local_files_only)
        if local_files_only and len(calls) == 1:
            raise FileNotFoundError("not cached")
        return str(tiny_bundle)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    with caplog.at_level("INFO", logger="semcer"):
        _loader.resolve_bundle("someone/some-model")
    assert calls == [True, False]
    assert "downloading the model someone/some-model (3 GB), once" in caplog.text
    caplog.clear()
    with caplog.at_level("INFO", logger="semcer"):
        _loader.resolve_bundle("someone/some-model")
    assert calls == [True, False, True] and caplog.text == ""
    assert capsys.readouterr() == ("", "")  # the library itself prints nothing
    calls.clear()
    with pytest.raises(FileNotFoundError):
        _loader.resolve_bundle("someone/some-model", local_files_only=True)
