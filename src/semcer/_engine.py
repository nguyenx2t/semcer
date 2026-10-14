"""Scoring engine: batched, prefix-cached per-edit inference.

Every edit input is ``pair_layer + "\n\n" + edit_block``. For a causal backbone
the shared prefix runs once and each edit continues from its cache; this is
exact because the readout only reads positions inside the edit block. A second
cache level holds the per-operation header (``"Operation: SUB\nREF:"``). Edits
that cannot continue the prefix, and non-causal backbones, take the uncached
path (one full forward per edit), which is also the parity reference.

:meth:`Engine.score_pair` runs one pair at a time, which suits a CPU.
:meth:`Engine.score_pairs` shares forwards across pairs, which a GPU needs: a
single pair rarely has enough edits to fill one. Both give the same values.
"""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass, field

import torch

from semcer._convpatch import discard_final_state
from semcer._model import distribution_severity
from semcer._prep import PairTokenizer, PreparedPair, TokenizedEdit, tokenize_prefix

#: Backbone ``model_type``s that are causal, so prefix-state caching is exact.
CAUSAL_MODEL_TYPES = frozenset({"qwen3_5_text", "qwen3_5"})


class NonFiniteScoreError(FloatingPointError):
    """The model produced a NaN/inf output; nothing is aggregated from it."""


@dataclass
class EngineConfig:
    max_length: int
    #: Padded tokens per forward: rows x longest row (suffix tokens on the cached path).
    max_batch_tokens: int = 4096
    #: Edits (rows) per forward.
    max_batch_edits: int = 16
    #: Edits rendered + tokenized at a time within one pair.
    chunk_edits: int = 256
    #: Use prefix-state caching when the backbone is causal.
    cache_prefix: bool = True
    #: Also cache the per-operation edit-block header, one level below the prefix.
    cache_op_header: bool = True
    #: Cross-pair path: pairs whose prefixes run in one forward.
    max_prefill_pairs: int = 16
    #: Cross-pair path: tokens a pair's prefix may run past its group's shared
    #: length; each of the pair's edits re-runs that excess.
    prefix_slack: int = 16
    #: Cross-pair path: cached + new tokens per forward (rows x (prefix + suffix)),
    #: which is what the attention cache of one forward holds.
    max_cache_tokens: int = 16384


@dataclass
class EditScore:
    severity: float | None
    probs: list[float] | None
    status: str
    context: str


@dataclass
class EngineStats:
    forwards: int = 0
    prefix_forwards: int = 0
    cached_rows: int = 0
    uncached_rows: int = 0
    fallback_rows: int = 0  # edits of a cached pair that had to run uncached
    real_tokens: int = 0
    padded_tokens: int = 0
    header_forwards: int = 0
    header_rows: int = 0  # edits scored on top of an operation-header cache
    max_forward_rows: int = 0
    max_forward_tokens: int = 0  # padded tokens of the largest single forward
    prefill_rows: int = 0  # pairs whose prefix ran in a cross-pair prefill forward
    slack_tokens: int = 0  # prefix tokens re-run with edits on the cross-pair path
    window_fallbacks: int = 0  # cross-pair windows redone one pair at a time

    def record(self, rows: int, width: int, real: int, *, cached: bool) -> None:
        self.forwards += 1
        if cached:
            self.cached_rows += rows
        else:
            self.uncached_rows += rows
        self.real_tokens += real
        self.padded_tokens += rows * width
        self.max_forward_rows = max(self.max_forward_rows, rows)
        self.max_forward_tokens = max(self.max_forward_tokens, rows * width)

    def record_cache_forward(self, n_tokens: int) -> None:
        self.real_tokens += n_tokens
        self.padded_tokens += n_tokens

    @property
    def padding_fraction(self) -> float:
        return 1.0 - self.real_tokens / self.padded_tokens if self.padded_tokens else 0.0


def _lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _micro_batches(rows: list, length_of, max_rows: int, max_tokens: int, offset: int = 0):
    """Greedy longest-first packing under a row cap and a padded-token cap
    (each batch pads to its first row; a row over the token cap runs alone).
    ``offset`` tokens count against the cap for every row."""
    order = sorted(range(len(rows)), key=lambda i: -length_of(rows[i]))
    batch: list[int] = []
    width = 0
    for i in order:
        n = length_of(rows[i])
        new_width = max(width, n)
        if batch and (
            len(batch) + 1 > max_rows or (len(batch) + 1) * (offset + new_width) > max_tokens
        ):
            yield [rows[j] for j in batch]
            batch, new_width = [], n
        batch.append(i)
        width = new_width
    if batch:
        yield [rows[j] for j in batch]


def _padded(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Right-padded ``(ids, mask, width)`` for token lists."""
    width = max(len(x) for x in seqs)
    ids = torch.zeros((len(seqs), width), dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for r, x in enumerate(seqs):
        ids[r, : len(x)] = torch.tensor(x, dtype=torch.long)
        mask[r, : len(x)] = 1
    return ids, mask, width


def _read_only_conv_update(self, conv_states, state_idx=0, **kwargs):
    # ``update_conv_state`` for a layer that already holds a state: the same
    # return value, without the in-place write of the new tail.
    return torch.cat([self.conv_states[state_idx], conv_states], dim=-1)


def _read_only_recurrent_update(self, recurrent_states, state_idx=0, **kwargs):
    return recurrent_states  # the caller ignores the return; nothing reads the new state


def _fork_cache(base, rows, *, share: bool):
    """A read-only view of the prefix cache ``base`` for one suffix forward.

    ``rows`` is a batch size ``b`` (``base`` holds one row, broadcast to ``b``)
    or a 1-D index tensor (``base`` holds several rows; row ``r`` of the fork is
    row ``rows[r]`` of ``base``).

    A suffix forward reads the prefix state and never needs the state it leaves
    behind, so the fork drops the in-place state writes (attention layers
    already ``cat`` rather than write in place). With a batch size,
    ``share=True`` keeps the recurrent states as stride-0 views (the CPU kernels
    only read them) and ``share=False`` materialises them for fused kernels.
    Same values as ``deepcopy`` + ``reorder_cache``.
    """
    from transformers.cache_utils import LinearAttentionCacheLayerMixin

    def grow(t, keep_view):
        if t is None:
            return None
        if not isinstance(rows, int):
            return t.index_select(0, rows)
        v = t.expand(rows, *t.shape[1:])
        return v if keep_view else v.contiguous()

    fork = copy.copy(base)
    fork.layers = []
    for layer in base.layers:
        nl = copy.copy(layer)
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            nl.conv_states = {i: grow(t, True) for i, t in layer.conv_states.items()}  # cat copies
            nl.recurrent_states = {i: grow(t, share) for i, t in layer.recurrent_states.items()}
            nl.has_previous_state = dict(layer.has_previous_state)
            nl.update_conv_state = types.MethodType(_read_only_conv_update, nl)
            nl.update_recurrent_state = types.MethodType(_read_only_recurrent_update, nl)
        if getattr(layer, "is_initialized", False) and getattr(layer, "keys", None) is not None:
            nl.keys, nl.values = grow(layer.keys, True), grow(layer.values, True)  # cat copies
        fork.layers.append(nl)
    return fork


def _locus_start(te: TokenizedEdit) -> int:
    return min(te.ref_span[0], te.hyp_span[0])


def _suffix(te: TokenizedEdit, start: int) -> list[int]:
    """``te``'s tokens from position ``start`` on."""
    if te.split is not None and start >= te.split:
        return te.suffix[start - te.split :]
    return te.ids[start:]


@dataclass
class _PairState:
    k: int | None = None  # prefix split point, set by the pair's first scorable edit
    prefix_ids: list[int] | None = None
    cache: object = None  # prefix cache (k tokens)
    op_nodes: dict = field(default_factory=dict)  # op -> (header_ids, cache) or (None, None)


@dataclass
class Engine:
    model: torch.nn.Module
    tokenizer: object
    device: str
    config: EngineConfig
    stats: EngineStats = field(default_factory=EngineStats)

    def __post_init__(self) -> None:
        self.causal = self.model.backbone.config.model_type in CAUSAL_MODEL_TYPES

    @property
    def path(self) -> str:
        return "prefix_cached" if (self.causal and self.config.cache_prefix) else "uncached"

    def score_pairs(self, preps: list[PreparedPair]) -> list[list[EditScore]]:
        """Score every edit of every pair, sharing forwards across pairs.

        Pairs are grouped by prefix length. A group's prefixes run as one
        unpadded forward over the tokens all of them have (the shortest prefix
        in the group), and each edit continues from its own pair's row of that
        cache, re-running the few prefix tokens its pair has past the shared
        length. Rows of a batch are independent, so the values are those of
        :meth:`score_pair`. A pair that cannot split its prefix off (non-causal
        backbone, or a tokenizer that merges across the split) takes
        :meth:`score_pair`.
        """
        if self.path != "prefix_cached":
            return [self.score_pair(prep) for prep in preps]
        outs: list[list] = [[None] * len(prep.edits) for prep in preps]
        cfg = self.config
        pending: list[tuple[int, list[int], list[TokenizedEdit]]] = []  # pair, prefix ids, edits
        for p, prep in enumerate(preps):
            if not prep.edits:
                continue
            tok = PairTokenizer(prep, self.tokenizer, cfg.max_length)
            tes = tok(range(len(prep.edits)))
            if not tok.shared:
                outs[p] = self.score_pair(prep)
                continue
            scorable = []
            for te in tes:
                if te.status == "ok":
                    scorable.append(te)
                else:
                    outs[p][te.index] = EditScore(None, None, te.status, te.context)
            if scorable:
                pending.append((p, tok.prefix_ids, scorable))
        pending.sort(key=lambda x: len(x[1]))
        i = 0
        while i < len(pending):
            k = len(pending[i][1])  # the group's shared length: its shortest prefix
            j = i + 1
            while (
                j < len(pending)
                and j - i < cfg.max_prefill_pairs
                and len(pending[j][1]) - k <= cfg.prefix_slack
                and (j - i + 1) * k <= cfg.max_cache_tokens
            ):
                j += 1
            self._run_group(pending[i:j], k, outs)
            i = j
        return outs

    def _run_group(self, group, k: int, outs) -> None:
        """One prefill forward over ``group``'s first ``k`` prefix tokens, then its
        edits in batches drawn across the group's pairs."""
        cfg, dev = self.config, self.device
        b = len(group)
        with torch.inference_mode():
            cache = self.model.backbone(
                input_ids=torch.tensor([prefix[:k] for _, prefix, _ in group], dtype=torch.long, device=dev),
                attention_mask=torch.ones((b, k), dtype=torch.long, device=dev),
                position_ids=torch.arange(k, device=dev).unsqueeze(0).expand(b, k),
                use_cache=True,
            ).past_key_values
            self.stats.prefix_forwards += 1
            self.stats.prefill_rows += b
            self.stats.record_cache_forward(b * k)
            rows = []  # (group row, edit, suffix tokens from k on, the pair's output list)
            for g, (p, prefix, tes) in enumerate(group):
                extra = prefix[k:]
                self.stats.slack_tokens += len(extra) * len(tes)
                rows += [(g, te, extra + te.suffix, outs[p]) for te in tes]
            for batch in _micro_batches(
                rows, lambda r: len(r[2]), cfg.max_batch_edits, cfg.max_cache_tokens, offset=k
            ):
                fork = _fork_cache(cache, torch.tensor([r[0] for r in batch], device=dev), share=False)
                logits = self._continue(fork, k, [r[2] for r in batch], [r[1] for r in batch])
                self._finish([r[1] for r in batch], logits, [r[3] for r in batch])

    def score_pair(self, prep: PreparedPair) -> list[EditScore]:
        """Score every edit of ``prep``, in edit order."""
        n = len(prep.edits)
        if n == 0:
            return []
        out: list[EditScore | None] = [None] * n
        state = _PairState()
        step = max(1, self.config.chunk_edits)
        tok = PairTokenizer(prep, self.tokenizer, self.config.max_length)
        for start in range(0, n, step):
            self._score_chunk(prep, tok(range(start, min(n, start + step))), state, out)
        return out  # type: ignore[return-value]

    # -- routing --------------------------------------------------------------
    def _score_chunk(self, prep, chunk: list[TokenizedEdit], state: _PairState, out) -> None:
        cached: list[TokenizedEdit] = []
        uncached: list[TokenizedEdit] = []
        for te in chunk:
            if te.status != "ok":
                out[te.index] = EditScore(None, None, te.status, te.context)
            elif self.path != "prefix_cached":
                uncached.append(te)
            elif self._continues_prefix(prep, te, state):
                cached.append(te)
            else:
                self.stats.fallback_rows += 1
                uncached.append(te)
        if cached:
            self._run_cached(state, cached, [prep.edits[te.index].op for te in cached], out)
        if uncached:
            self._run_uncached(uncached, out)

    def _continues_prefix(self, prep, te: TokenizedEdit, state: _PairState) -> bool:
        """Whether ``te`` can continue from the pair's prefix cache (sets the split
        point on the pair's first scorable edit)."""
        if state.k is None:
            if te.split is not None:  # shared-prefix tokenization: the split is known
                state.prefix_ids, state.k = te.prefix, te.split
            else:
                state.prefix_ids = tokenize_prefix(prep, self.tokenizer, self.config.max_length)
                state.k = min(_lcp(state.prefix_ids, te.ids), _locus_start(te), len(te) - 1)
        k = state.k
        return bool(
            k
            and len(te) > k
            and _locus_start(te) >= k
            and (te.split == k if te.split is not None else te.ids[:k] == state.prefix_ids[:k])
        )

    # -- forwards -------------------------------------------------------------
    def _finish(self, rows: list[TokenizedEdit], logits: torch.Tensor, outs: list) -> None:
        """Write each row's score into its pair's output list (``outs``: one per row)."""
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise NonFiniteScoreError("non-finite model output; refusing to aggregate it")
        sev = distribution_severity(logits).tolist()
        probs = torch.softmax(logits, dim=-1).tolist()
        for te, s, p, o in zip(rows, sev, probs, outs):
            o[te.index] = EditScore(min(1.0, max(0.0, float(s))), p, "ok", te.context)

    def _run_uncached(self, rows: list[TokenizedEdit], out) -> None:
        cfg, dev = self.config, self.device
        for batch in _micro_batches(rows, len, cfg.max_batch_edits, cfg.max_batch_tokens):
            ids, mask, width = _padded([te.ids for te in batch])
            ref = torch.tensor([te.ref_span for te in batch], dtype=torch.long)
            hyp = torch.tensor([te.hyp_span for te in batch], dtype=torch.long)
            with torch.inference_mode():
                logits = self.model(ids.to(dev), mask.to(dev), ref.to(dev), hyp.to(dev))
            self.stats.record(len(batch), width, sum(len(te) for te in batch), cached=False)
            self._finish(batch, logits, [out] * len(batch))

    def _extend(self, cache, ids: list[int], start: int):
        """``cache`` (holding ``start`` tokens) extended by ``ids``, as a new cache."""
        dev = self.device
        return self.model.backbone(
            input_ids=torch.tensor([ids], dtype=torch.long, device=dev),
            attention_mask=torch.ones((1, start + len(ids)), dtype=torch.long, device=dev),
            position_ids=torch.arange(start, start + len(ids), device=dev).unsqueeze(0),
            past_key_values=copy.deepcopy(cache),
            use_cache=True,
        ).past_key_values

    def _header_node(self, state: _PairState, op: str, group: list[TokenizedEdit]):
        """``(header_ids, cache)`` for ``op``, or ``(None, None)``: the longest token
        prefix the group's suffixes share, before every locus and last token.
        Decided once per pair, from the first group of >= 2 edits."""
        if op in state.op_nodes:
            return state.op_nodes[op]
        if not self.config.cache_op_header or len(group) < 2:
            return None, None
        k = state.k
        sufs = [_suffix(te, k) for te in group]
        h = min(len(x) for x in sufs) - 1
        for x in sufs[1:]:
            h = min(h, _lcp(sufs[0], x))
        h = min(h, min(_locus_start(te) - k for te in group))
        node = (None, None)
        if h >= 2:
            node = (sufs[0][:h], self._extend(state.cache, sufs[0][:h], k))
            self.stats.header_forwards += 1
            self.stats.record_cache_forward(h)
        state.op_nodes[op] = node
        return node

    def _run_cached(self, state: _PairState, rows: list[TokenizedEdit], ops: list[str], out) -> None:
        k = state.k
        with torch.inference_mode():
            if state.cache is None:
                state.cache = self._extend(None, state.prefix_ids[:k], 0)
                self.stats.prefix_forwards += 1
                self.stats.record_cache_forward(k)
            groups: dict[str, list[TokenizedEdit]] = {}
            for te, op in zip(rows, ops):
                groups.setdefault(op, []).append(te)
            plain: list[TokenizedEdit] = []
            for op, group in groups.items():
                hdr, hcache = self._header_node(state, op, group)
                if hdr is None:
                    plain += group
                    continue
                start, on_header = k + len(hdr), []
                for te in group:
                    if len(te) > start and _locus_start(te) >= start and _suffix(te, k)[: len(hdr)] == hdr:
                        on_header.append(te)
                    else:
                        plain.append(te)
                self.stats.header_rows += len(on_header)
                self._run_from(hcache, start, on_header, out)
            self._run_from(state.cache, k, plain, out)

    def _run_from(self, base_cache, start: int, rows: list[TokenizedEdit], out) -> None:
        """Score ``rows`` as continuations of ``base_cache`` (which holds ``start`` tokens)."""
        if not rows:
            return
        cfg, dev = self.config, self.device
        for batch in _micro_batches(
            rows, lambda te: len(te) - start, cfg.max_batch_edits, cfg.max_batch_tokens
        ):
            cache = _fork_cache(base_cache, len(batch), share=str(dev).startswith("cpu"))
            logits = self._continue(cache, start, [_suffix(te, start) for te in batch], batch)
            self._finish(batch, logits, [out] * len(batch))

    def _continue(self, cache, start: int, sufs: list[list[int]], rows: list[TokenizedEdit]):
        """Digit logits for ``rows``, whose tokens from position ``start`` on are
        ``sufs``, run on top of ``cache`` (one row per edit, ``start`` tokens each)."""
        dev, b = self.device, len(rows)
        ids, smask, width = _padded(sufs)
        ref = torch.tensor([te.ref_span for te in rows], dtype=torch.long) - start
        hyp = torch.tensor([te.hyp_span for te in rows], dtype=torch.long) - start
        assert int(ref[:, 0].min()) >= 0 and int(hyp[:, 0].min()) >= 0  # loci are in the suffix
        ids, smask, ref, hyp = ids.to(dev), smask.to(dev), ref.to(dev), hyp.to(dev)
        full_mask = torch.cat([torch.ones((b, start), dtype=torch.long, device=dev), smask], dim=1)
        pos = (torch.arange(width, device=dev) + start).unsqueeze(0).expand(b, width)
        with discard_final_state():  # the fork's end state is never read
            hidden = self.model.backbone(
                input_ids=ids,
                attention_mask=full_mask,
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state
        self.stats.record(b, width, sum(len(x) for x in sufs), cached=True)
        return self.model.head_logits(hidden, ref, hyp, smask)


def pair_numbers(severities: list[float | None], n_ref: int, unscored: str) -> dict:
    """Aggregate one pair's severities under an unscored-edit policy.

    Sums run left to right in edit order (not ``sum()``, which compensates on
    Python >= 3.12) so a score does not depend on the interpreter.
    """
    n_edits = len(severities)
    n_unscored = sum(1 for s in severities if s is None)
    ssum = 0.0
    for s in severities:
        if s is not None:
            ssum += s
    semcer = effective = None
    if n_unscored == 0:
        semcer = ssum / n_ref if n_ref > 0 else float("nan")
    elif unscored == "impute_one":
        total = 0.0
        for s in severities:
            total += 1.0 if s is None else s
        semcer = total / n_ref if n_ref > 0 else float("nan")
        effective = ssum + float(n_unscored)
    return {
        "status": "ok" if n_unscored == 0 else "partial",
        "semcer": semcer,
        "severity_sum": ssum,
        "effective_severity_sum": effective,
        "n_unscored": n_unscored,
        "coverage": 1.0 if n_edits == 0 else (n_edits - n_unscored) / n_edits,
    }


__all__ = [
    "CAUSAL_MODEL_TYPES",
    "EditScore",
    "Engine",
    "EngineConfig",
    "EngineStats",
    "NonFiniteScoreError",
    "pair_numbers",
]
