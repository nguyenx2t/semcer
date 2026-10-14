"""Bundle loader behind ``SemCER.from_pretrained``.

A bundle is a directory (local, or a Hugging Face Hub snapshot) holding
``model.safetensors`` (fp32, optionally ``model.bf16.safetensors``), the backbone
``config.json``, the tokenizer files and ``semcer_meta.json`` (see
:data:`REQUIRED_META`). Head, readout and ``max_length`` come from the meta,
which is validated before any weight is read.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Bundle format versions this package can read.
SUPPORTED_BUNDLE_VERSIONS = frozenset({1})
#: The published default model, pinned to one Hub commit so a release of this
#: package always scores with the same weights.
DEFAULT_MODEL_ID: str | None = "nguyenx2t/semcer-qwen3.5-0.8b"
DEFAULT_REVISION: str | None = "b612a672c2d3f4101a734e64f28370bd7d872f67"
#: Environment override for the default model (a bundle directory or a Hub id).
MODEL_ENV = "SEMCER_MODEL"

META_FILE = "semcer_meta.json"
REQUIRED_META = (
    "bundle_version",
    "head_type",
    "readout",
    "max_length",
    "backbone",
    "backbone_revision",
    "input_template_version",
    "alignment_version",
    "digit_order",
)
KNOWN_HEADS = ("distribution",)
KNOWN_READOUTS = ("span2_last",)


class BundleError(ValueError):
    """The bundle is missing, malformed, or of a kind this package cannot run."""


@dataclass
class LoadedBundle:
    model: Any
    tokenizer: Any
    config: Any
    meta: dict
    path: Path
    source: str  # what the user asked for (a path or a Hub id)
    revision: str | None  # resolved Hub commit, or None for a local directory


def validate_meta(meta: dict) -> dict:
    """Raise :class:`BundleError` unless ``meta`` describes a runnable bundle."""
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise BundleError(f"{META_FILE} is missing {missing}")
    if meta["bundle_version"] not in SUPPORTED_BUNDLE_VERSIONS:
        raise BundleError(
            f"bundle_version {meta['bundle_version']!r} is not supported by this semcer "
            f"(supports {sorted(SUPPORTED_BUNDLE_VERSIONS)}); upgrade the package"
        )
    if meta["head_type"] not in KNOWN_HEADS:
        raise BundleError(f"unknown head_type {meta['head_type']!r}; known {KNOWN_HEADS}")
    if meta["readout"] not in KNOWN_READOUTS:
        raise BundleError(f"unknown readout {meta['readout']!r}; known {KNOWN_READOUTS}")
    if not isinstance(meta["max_length"], int) or meta["max_length"] <= 0:
        raise BundleError(f"max_length must be a positive int, got {meta['max_length']!r}")
    if list(meta["digit_order"]) != [0, 1, 2, 3, 4]:
        raise BundleError(f"unsupported digit_order {meta['digit_order']!r}")
    return meta


def read_meta(bundle_dir: Path) -> dict:
    path = Path(bundle_dir) / META_FILE
    if not path.is_file():
        raise BundleError(f"not a semcer bundle: {path} does not exist")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"{path} is not valid JSON: {exc}") from exc
    return validate_meta(meta)


def resolve_bundle(
    name_or_path: str | os.PathLike | None,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    dtype: str = "fp32",
) -> tuple[Path, str, str | None]:
    """``(directory, source, resolved_revision)`` for a local path or a Hub id.

    A local directory is used as is; a Hub id goes through ``snapshot_download``,
    which fetches only the weights file ``dtype`` needs.
    """
    if name_or_path is None:
        name_or_path = os.environ.get(MODEL_ENV) or DEFAULT_MODEL_ID
        if revision is None and name_or_path == DEFAULT_MODEL_ID:
            revision = DEFAULT_REVISION
    if name_or_path is None:
        raise BundleError(
            "no default semcer model has been published yet; pass a bundle directory or "
            f"Hub id to from_pretrained(), or set ${MODEL_ENV}"
        )
    path = Path(name_or_path).expanduser()
    if path.is_dir():
        return path, str(name_or_path), None
    if path.exists() or str(name_or_path).startswith((".", "/", "~")):
        raise BundleError(f"bundle directory {path} does not exist")
    from huggingface_hub import snapshot_download

    other = "model.safetensors" if dtype == "bf16" else "model.bf16.safetensors"
    local = snapshot_download(
        repo_id=str(name_or_path), revision=revision, local_files_only=local_files_only, ignore_patterns=[other]
    )
    local = Path(local)
    return local, str(name_or_path), local.name  # snapshot dirs are named by commit


def _select_weights(bundle_dir: Path, dtype: str) -> tuple[Path, bool]:
    """The weights file for ``dtype`` and whether it must be cast after loading."""
    fp32 = bundle_dir / "model.safetensors"
    bf16 = bundle_dir / "model.bf16.safetensors"
    if dtype == "bf16" and bf16.is_file():
        return bf16, False
    if fp32.is_file():
        return fp32, dtype == "bf16"
    raise BundleError(f"no model.safetensors in {bundle_dir}")


def resolve_device(device: str) -> str:
    """``"auto"`` becomes ``"cuda"`` when CUDA is available, else ``"cpu"``."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def check_dtype(dtype: str, device: str) -> None:
    import torch

    if dtype not in ("fp32", "bf16"):
        raise ValueError(f"unsupported dtype {dtype!r}: use 'fp32' or 'bf16'")
    if dtype == "bf16" and str(device).startswith("cuda"):
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            raise ValueError("dtype='bf16' needs a bf16-capable CUDA device; use dtype='fp32'")


def load_bundle(
    name_or_path: str | os.PathLike | None = None,
    *,
    revision: str | None = None,
    device: str = "cpu",
    dtype: str = "fp32",
    local_files_only: bool = False,
) -> LoadedBundle:
    """Resolve, validate and load a bundle onto ``device`` in eval mode."""
    device = resolve_device(device)
    check_dtype(dtype, device)
    bundle_dir, source, resolved = resolve_bundle(
        name_or_path, revision=revision, local_files_only=local_files_only, dtype=dtype
    )
    meta = read_meta(bundle_dir)  # validated before any weight is touched

    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer

    from semcer._model import SeverityStudent

    config = AutoConfig.from_pretrained(str(bundle_dir), local_files_only=True)
    weights_path, cast = _select_weights(bundle_dir, dtype)
    state = load_file(str(weights_path))
    head_hidden = int(meta.get("head_hidden") or state["head.0.weight"].shape[0])
    model = SeverityStudent.from_config(config, head_hidden=head_hidden)
    if dtype == "bf16" and not cast:
        model = model.to(torch.bfloat16)
    model.load_state_dict(state, strict=True)
    del state
    if cast:
        model = model.to(torch.bfloat16)
    if dtype == "fp32" and str(device).startswith("cuda"):
        # fp32 parity: TF32 matmuls would change results beyond the fp32 tolerance.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir), local_files_only=True)
    if not getattr(tokenizer, "is_fast", False):
        raise BundleError("semcer needs a fast tokenizer (offset mapping) in the bundle")

    from semcer import _convpatch

    if _convpatch.uses_gated_deltanet(model):
        _convpatch.install()
    return LoadedBundle(model, tokenizer, config, meta, bundle_dir, source, resolved)


__all__ = ["BundleError", "LoadedBundle", "load_bundle"]
