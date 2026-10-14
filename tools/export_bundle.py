"""Training checkpoint -> semcer inference bundle (maintainers only).

    python tools/export_bundle.py CKPT.pt OUT_DIR --training-commit SHA [--model-card CARD.md]
        [--label TEXT] [--no-bf16] [--allow-dev]

Accepts exactly one checkpoint contract:

- ``state["model"]`` holds a full :class:`semcer._model.SeverityStudent` state dict;
- ``state["meta"]`` names the backbone in :data:`CONTRACT` at the pinned
  revision, ``head_type == "distribution"``, ``readout == "span2_last"`` and
  ``max_length == 2048``;
- optimizer, scheduler, RNG or EMA state may be present and is ignored.

Anything else is refused before a file is written. A checkpoint whose meta says
``dev_checkpoint: true`` (random head) is refused unless ``--allow-dev`` is set.

The write is atomic: every artifact is staged in a sibling temp directory, the
staged fp32 weights are reloaded and compared bit-for-bit with the checkpoint,
``SHA256SUMS`` is written last, and only then is the stage renamed into place.
The config, tokenizer and LICENSE come from the backbone's Hub repo at the
pinned revision, so an export needs the network once (or a warm Hub cache).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from semcer._loader import validate_meta
from semcer._model import HEAD_TYPE, READOUT, SeverityStudent
from semcer._prep import ALIGNMENT_VERSION, INPUT_TEMPLATE_VERSION

BUNDLE_VERSION = 1
CONTRACT = {
    "backbone": "Qwen/Qwen3.5-0.8B-Base",
    "backbone_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    "head_type": HEAD_TYPE,
    "readout": READOUT,
    "max_length": 2048,
}

NOTICE = """\
semcer severity model
=====================

Copyright (c) 2026 semcer authors.

These weights are a modified version of {backbone} (text model only; the
vision tower is not included), Copyright the Qwen team, Alibaba Cloud, licensed
under the Apache License, Version 2.0. Modifications: the language-model
weights were fine-tuned to predict a five-way edit-severity distribution, and a
span readout with a small MLP head was added. The bundled config.json and
tokenizer files originate from https://huggingface.co/{backbone} at revision
{revision}.

The weights are distributed under the Apache License, Version 2.0 (see
LICENSE).

The training targets are severity distributions produced by an LLM judge
(DeepSeek). They are AI-generated labels, not human annotations.
"""


class ExportError(ValueError):
    """The checkpoint does not satisfy the contract, or the export failed."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_contract(state: dict, *, allow_dev: bool = False) -> dict:
    """Return the checkpoint meta, or raise :class:`ExportError`."""
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise ExportError('checkpoint has no state["model"] state dict')
    meta = state.get("meta")
    if not isinstance(meta, dict):
        raise ExportError('checkpoint has no state["meta"]')
    for key, want in CONTRACT.items():
        if meta.get(key) != want:
            raise ExportError(f"meta[{key!r}] = {meta.get(key)!r}; the exporter only accepts {want!r}")
    if meta.get("ema_decay") is not None:
        raise ExportError('checkpoint was trained with EMA; the release reads state["model"] (EMA off)')
    if meta.get("dev_checkpoint") and not allow_dev:
        raise ExportError("this is a dev checkpoint (random head); pass --allow-dev")
    return meta


def _default_card(bundle_meta: dict) -> str:
    lines = ["# semcer severity model" + (" (DEV BUNDLE)" if bundle_meta["dev_checkpoint"] else ""), ""]
    if bundle_meta.get("label"):
        lines += [f"> **{bundle_meta['label']}**", ""]
    if bundle_meta["dev_checkpoint"]:
        lines += ["> **Random head, not a metric.** Its scores mean nothing. Do not publish it.", ""]
    lines += [
        "Predicts, for each aligned grapheme edit between a reference and a hypothesis",
        "transcription, an LLM judge's 5-way severity distribution; severity = E[digit]/4",
        "in [0, 1]. semcer = sum(severity) / reference graphemes.",
        "",
        f"- backbone: `{bundle_meta['backbone']}` @ `{bundle_meta['backbone_revision']}` (Apache-2.0)",
        f"- head / readout: `{bundle_meta['head_type']}` / `{bundle_meta['readout']}`",
        f"- max input tokens: {bundle_meta['max_length']}",
        f"- input template: `{bundle_meta['input_template_version']}`",
        f"- alignment: `{bundle_meta['alignment_version']}`",
        f"- source checkpoint sha256: `{bundle_meta['source_checkpoint_sha256']}`",
    ]
    return "\n".join(lines) + "\n"


def export_bundle(
    checkpoint: str | Path,
    out_dir: str | Path,
    *,
    training_commit: str | None = None,
    model_card: str | Path | None = None,
    label: str | None = None,
    write_bf16: bool = True,
    allow_dev: bool = False,
) -> Path:
    """Export ``checkpoint`` to the bundle directory ``out_dir`` (absent or empty)."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ExportError(f"{out_dir} exists and is not empty")
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False, mmap=True)
    meta = check_contract(state, allow_dev=allow_dev)
    if state.get("ema") is not None:
        warnings.warn('checkpoint carries EMA weights; the bundle uses state["model"]')
    weights = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in state["model"].items()}
    step = state.get("step")
    del state

    backbone, revision = meta["backbone"], meta["backbone_revision"]
    config = AutoConfig.from_pretrained(backbone, revision=revision)
    get_text = getattr(config, "get_text_config", None)
    config = get_text() if callable(get_text) else config
    head_hidden = int(weights["head.0.weight"].shape[0])
    model = SeverityStudent.from_config(config, head_hidden=head_hidden)
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise ExportError(f"state dict does not match SeverityStudent: {exc}") from exc
    del model
    tokenizer = AutoTokenizer.from_pretrained(backbone, revision=revision)
    if not tokenizer.is_fast:
        raise ExportError("the tokenizer must be a fast tokenizer")

    bundle_meta = {
        "bundle_version": BUNDLE_VERSION,
        "head_type": meta["head_type"],
        "readout": meta["readout"],
        "max_length": meta["max_length"],
        "backbone": backbone,
        "backbone_revision": revision,
        "input_template_version": INPUT_TEMPLATE_VERSION,
        "alignment_version": ALIGNMENT_VERSION,
        "digit_order": [0, 1, 2, 3, 4],
        "head_hidden": head_hidden,
        "weights": "state_dict[model] (EMA off)",
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_checkpoint_step": step,
        "training_commit": training_commit,
        "dev_checkpoint": bool(meta.get("dev_checkpoint", False)),
    }
    if label:
        bundle_meta["label"] = label
    validate_meta(bundle_meta)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=".export-"))
    try:
        fp32_path = staging / "model.safetensors"
        save_file(weights, str(fp32_path), metadata={"format": "pt"})
        if write_bf16:
            save_file(
                {k: v.to(torch.bfloat16) for k, v in weights.items()},
                str(staging / "model.bf16.safetensors"),
                metadata={"format": "pt"},
            )
        reloaded = load_file(str(fp32_path))
        for k, v in weights.items():
            if k not in reloaded or not torch.equal(reloaded[k], v):
                raise ExportError(f"reloaded weight {k!r} does not match the checkpoint")
        del reloaded
        config.save_pretrained(str(staging))
        tokenizer.save_pretrained(str(staging))
        (staging / "chat_template.jinja").unlink(missing_ok=True)  # the model is never prompted
        (staging / "semcer_meta.json").write_text(
            json.dumps(bundle_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copy(hf_hub_download(backbone, "LICENSE", revision=revision), staging / "LICENSE")
        (staging / "NOTICE").write_text(NOTICE.format(backbone=backbone, revision=revision), encoding="utf-8")
        card = Path(model_card).read_text(encoding="utf-8") if model_card else _default_card(bundle_meta)
        (staging / "README.md").write_text(card, encoding="utf-8")
        lines = [
            f"{sha256_file(p)}  {p.name}" for p in sorted(staging.iterdir()) if p.is_file() and p.name != "SHA256SUMS"
        ]
        (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if out_dir.exists():
            os.rmdir(out_dir)
        os.replace(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return out_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Export a training checkpoint to a semcer bundle.")
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    ap.add_argument("--training-commit", help="commit of the training code, recorded in the meta")
    ap.add_argument("--model-card", help="Markdown file written as the bundle's README.md")
    ap.add_argument("--label", help="shown in every result signature, e.g. for a non-release checkpoint")
    ap.add_argument("--no-bf16", action="store_true", help="skip model.bf16.safetensors")
    ap.add_argument("--allow-dev", action="store_true", help="export a random-head dev checkpoint")
    args = ap.parse_args(argv)
    print(
        export_bundle(
            args.checkpoint,
            args.out_dir,
            training_commit=args.training_commit,
            model_card=args.model_card,
            label=args.label,
            write_bf16=not args.no_bf16,
            allow_dev=args.allow_dev,
        )
    )


if __name__ == "__main__":
    main()
