"""The severity model: a HuggingFace backbone, a span readout and a small MLP head.

The REF and HYP locus token spans are max-pooled over the backbone's last hidden
state and concatenated with the last real token's state (readout ``span2_last``,
``3H`` features). The head maps that to five digit logits (``distribution`` head);
severity = E[digit] / 4.
"""

from __future__ import annotations

import inspect

import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig

HEAD_TYPE = "distribution"
READOUT = "span2_last"


def _text_config(config: PretrainedConfig) -> PretrainedConfig:
    """The language-model part of a (possibly multimodal) config."""
    get = getattr(config, "get_text_config", None)
    return get() if callable(get) else config


def masked_span_maxpool(hidden: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
    """Max-pool ``hidden (B, L, H)`` over each row's half-open span ``spans (B, 2)``.

    An empty span (``start == end``) pools the single position ``start``, so the
    result is never ``-inf``.
    """
    _, length, _ = hidden.shape
    start = spans[:, 0]
    end = torch.maximum(spans[:, 1], start + 1)
    positions = torch.arange(length, device=hidden.device).unsqueeze(0)  # (1, L)
    in_span = (positions >= start.unsqueeze(1)) & (positions < end.unsqueeze(1))
    neg_inf = torch.tensor(float("-inf"), dtype=hidden.dtype, device=hidden.device)
    pooled, _ = torch.where(in_span.unsqueeze(-1), hidden, neg_inf).max(dim=1)
    return pooled


def last_token_state(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """``(B, H)``: each row's hidden state at its last ``attention_mask == 1`` position."""
    positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    last = (positions * (attention_mask > 0)).amax(dim=1)  # (B,)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def distribution_severity(logits: torch.Tensor) -> torch.Tensor:
    """``E[d] / 4`` under ``softmax(logits)`` over digits 0..4, in float32; ``(B,)``."""
    probs = torch.softmax(logits.float(), dim=-1)
    digits = torch.arange(5, dtype=torch.float32, device=logits.device)
    return (probs * digits).sum(dim=-1) / 4.0


class SeverityStudent(nn.Module):
    """Backbone + span readout + MLP head -> five digit logits."""

    head_type = HEAD_TYPE
    readout = READOUT

    def __init__(self, backbone: nn.Module, *, head_hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        # Decoder backbones would otherwise build a KV cache on every forward.
        self._no_cache = "use_cache" in inspect.signature(backbone.forward).parameters
        hidden_size = int(_text_config(backbone.config).hidden_size)
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 5),
        )

    @classmethod
    def from_config(cls, config: PretrainedConfig, *, head_hidden: int = 256) -> SeverityStudent:
        """Build a random-init float32 backbone (text model only) from ``config``.

        The explicit dtype matters: transformers would otherwise build in the
        config's dtype (bfloat16 for Qwen3.5) and round fp32 weights on load.
        """
        backbone = AutoModel.from_config(_text_config(config), dtype=torch.float32)
        return cls(backbone, head_hidden=head_hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ref_span: torch.Tensor,
        hyp_span: torch.Tensor,
    ) -> torch.Tensor:
        """Digit logits ``(B, 5)`` for a batch of full (uncached) edit inputs."""
        extra = {"use_cache": False} if self._no_cache else {}
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **extra)
        return self.head_logits(out.last_hidden_state, ref_span, hyp_span, attention_mask)

    def head_logits(
        self,
        hidden: torch.Tensor,
        ref_span: torch.Tensor,
        hyp_span: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Readout + head over hidden states ``(B, L, H)``; spans index into ``L``."""
        feats = [
            masked_span_maxpool(hidden, ref_span),
            masked_span_maxpool(hidden, hyp_span),
            last_token_state(hidden, attention_mask),
        ]
        return self.head(torch.cat(feats, dim=-1))


__all__ = [
    "HEAD_TYPE",
    "READOUT",
    "SeverityStudent",
    "distribution_severity",
    "last_token_state",
    "masked_span_maxpool",
]
