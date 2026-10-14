"""Per-pair preparation, torch-free.

1. NFC both sides and split into extended grapheme clusters;
2. align (:mod:`semcer._align`), after the cell-budget check so a runaway pair
   never allocates its DP table;
3. render the pair layer once and each edit's context block on demand;
4. tokenize edits with offsets and resolve their REF/HYP token loci.

An edit's model input is the pair layer, ``"\\n\\n"``, then the edit-context
block, with character loci offset by ``len(pair) + 2``. It is tokenized with
offsets and right truncation at ``max_length``; a locus past the window is
``truncated``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semcer._align import GraphemeEdit, align_grapheme_lists, dp_cells
from semcer._render import render_edit_context_with_spans, render_pair_layer
from semcer.graphemes import graphemes, nfc

#: Version of the model-input construction above. A change here is a new metric.
INPUT_TEMPLATE_VERSION = "pair_layer+nn+edit_context_w10/1"
#: Version of the alignment convention (NFC, UAX #29 graphemes, unit-cost DP,
#: tie-break SUB > DEL > INS, reference DEL/INS index conventions).
ALIGNMENT_VERSION = "nfc_egc_wf_sub>del>ins/1"
#: Default cell budget: pairs whose DP table would exceed this many cells are not aligned.
DEFAULT_CELL_BUDGET = 25_000_000

_SEPARATOR = "\n\n"


class PairTooLarge(Exception):
    """The pair's DP table exceeds the cell budget."""

    def __init__(self, n_ref: int, n_hyp: int, budget: int):
        self.n_ref, self.n_hyp, self.budget = n_ref, n_hyp, budget
        super().__init__(
            f"pair too large to align: {n_ref} x {n_hyp} graphemes needs "
            f"{dp_cells(n_ref, n_hyp):,} DP cells > cell_budget {budget:,}"
        )


@dataclass
class PreparedPair:
    """A normalized, aligned pair with its shared pair layer rendered."""

    reference: str  # NFC
    hypothesis: str  # NFC
    r_graphs: list[str]
    h_graphs: list[str]
    edits: list[GraphemeEdit]
    pair_text: str = field(repr=False)

    @property
    def n_ref(self) -> int:
        return len(self.r_graphs)

    @property
    def prefix_text(self) -> str:
        """The text every edit input starts with: pair layer + ``"\\n\\n"``."""
        return self.pair_text + _SEPARATOR

    def edit_block(self, k: int):
        """Edit ``k``'s rendered context block (text and block-relative loci)."""
        return render_edit_context_with_spans(self.r_graphs, self.h_graphs, self.edits[k])

    def edit_input(self, k: int) -> tuple[str, tuple[int, int], tuple[int, int], str]:
        """``(input_text, ref_char_span, hyp_char_span, context)`` for edit ``k``."""
        ec = self.edit_block(k)
        off = len(self.prefix_text)
        (r0, r1), (h0, h1) = ec.ref_locus_span, ec.hyp_locus_span
        return self.prefix_text + ec.text, (off + r0, off + r1), (off + h0, off + h1), ec.text


def prepare_pair(
    reference: str,
    hypothesis: str,
    *,
    cell_budget: int | None = DEFAULT_CELL_BUDGET,
) -> PreparedPair:
    """Steps 1-3 for one pair. Raises :class:`PairTooLarge` over ``cell_budget``
    (``None`` disables the cap). Equal NFC strings skip the DP entirely."""
    gt, pred = nfc(reference), nfc(hypothesis)
    r_graphs, h_graphs = graphemes(gt), graphemes(pred)
    edits: list[GraphemeEdit] = []
    if gt != pred:
        if cell_budget is not None and dp_cells(len(r_graphs), len(h_graphs)) > cell_budget:
            raise PairTooLarge(len(r_graphs), len(h_graphs), cell_budget)
        edits = align_grapheme_lists(r_graphs, h_graphs)
    return PreparedPair(gt, pred, r_graphs, h_graphs, edits, render_pair_layer(gt, pred))


@dataclass
class TokenizedEdit:
    """One edit's model input: token ids and half-open token loci, or a drop status.

    The full input is ``prefix + suffix``. With shared-prefix tokenization
    ``prefix`` is the pair's prefix token list (one object shared by all its
    edits) and ``split == len(prefix)``; with full tokenization ``prefix`` is
    empty, ``suffix`` holds the whole input and ``split`` is ``None``. Loci are
    absolute positions.
    """

    index: int  # position in the pair's edit list
    prefix: list[int]
    suffix: list[int]
    ref_span: tuple[int, int] | None
    hyp_span: tuple[int, int] | None
    status: str  # "ok" | "truncated" | "zero_token"
    context: str
    split: int | None = None

    @property
    def ids(self) -> list[int]:
        return self.prefix + self.suffix if self.prefix else self.suffix

    def __len__(self) -> int:
        return len(self.prefix) + len(self.suffix)


def _token_span(locus: tuple[int, int], offsets) -> tuple[tuple[int, int] | None, str]:
    """Half-open span of the tokens overlapping the char ``locus``, and a status.

    Zero-width (special) tokens are ignored. A locus at or past the end of the
    last real token is ``"truncated"``; one no token overlaps is ``"zero_token"``.
    """
    ls, le = locus
    max_end, first, last = -1, -1, -1
    for k, (cs, ce) in enumerate(offsets):
        if ce <= cs:
            continue
        max_end = max(max_end, ce)
        if cs < le and ce > ls:
            if first < 0:
                first = k
            last = k
    if max_end >= 0 and ls >= max_end:
        return None, "truncated"
    if first < 0:
        return None, "zero_token"
    return (first, last + 1), "ok"


def _resolve(rspan, hspan, offsets, shift: int = 0):
    """``(status, ref_span, hyp_span)``: token loci shifted by ``shift``, or ``None``s."""
    r, r_status = _token_span(rspan, offsets)
    h, h_status = _token_span(hspan, offsets)
    status = r_status if r_status != "ok" else h_status
    if status != "ok":
        return status, None, None
    return status, (r[0] + shift, r[1] + shift), (h[0] + shift, h[1] + shift)


def tokenize_edits_full(
    prep: PreparedPair, tokenizer, max_length: int, indices: range | list[int]
) -> list[TokenizedEdit]:
    """Reference tokenization: every edit's full input text in one batched call
    (no padding, so each row equals a single-text call), then locus resolution."""
    indices = list(indices)
    if not indices:
        return []
    built = [prep.edit_input(k) for k in indices]
    enc = tokenizer(
        [b[0] for b in built], return_offsets_mapping=True, truncation=True, max_length=max_length
    )
    out: list[TokenizedEdit] = []
    for row, (k, (_, rspan, hspan, context)) in enumerate(zip(indices, built)):
        status, r, h = _resolve(rspan, hspan, enc["offset_mapping"][row])
        out.append(TokenizedEdit(k, [], list(enc["input_ids"][row]), r, h, status, context))
    return out


class PairTokenizer:
    """Tokenizes one pair's edits with the shared prefix tokenized **once**.

    Every edit input is ``prefix_text + block`` and every block starts with
    ``"Operation: "``. Under a byte-level BPE whose pre-tokenizer cannot join the
    ``"\\n"`` that ends the prefix with the following letter (Qwen's), the full
    tokenization is ``tok(prefix_text) + tok(block)``, so only the ~35-token
    block is tokenized per edit. Truncation at ``max_length`` keeps the first
    ``max_length - len(prefix)`` block tokens; loci always lie in the block.

    The split is **verified** on the pair's first edit against the full
    tokenization; if it does not hold, the pair uses :func:`tokenize_edits_full`.
    ``shared`` says which path the pair took.
    """

    def __init__(self, prep: PreparedPair, tokenizer, max_length: int):
        self.prep, self.tokenizer, self.max_length = prep, tokenizer, max_length
        self.shared: bool | None = None  # decided on the first call with edits
        self.prefix_ids: list[int] = []

    def _verify(self, block_text: str, block_ids: list[int]) -> bool:
        full = self.tokenizer(self.prep.prefix_text + block_text)["input_ids"]
        return list(full) == self.prefix_ids + list(block_ids)

    def __call__(self, indices: range | list[int]) -> list[TokenizedEdit]:
        indices = list(indices)
        if not indices:
            return []
        if self.shared is False:
            return tokenize_edits_full(self.prep, self.tokenizer, self.max_length, indices)
        ecs = [self.prep.edit_block(k) for k in indices]
        enc = self.tokenizer([ec.text for ec in ecs], return_offsets_mapping=True)
        if self.shared is None:
            self.prefix_ids = list(self.tokenizer(self.prep.prefix_text)["input_ids"])
            self.shared = self._verify(ecs[0].text, enc["input_ids"][0])
            if not self.shared:
                return tokenize_edits_full(self.prep, self.tokenizer, self.max_length, indices)
        P = len(self.prefix_ids)
        keep = self.max_length - P
        if keep <= 0:  # the pair layer alone fills the window
            prefix = self.prefix_ids[: self.max_length]
            return [TokenizedEdit(k, prefix, [], None, None, "truncated", ec.text, P) for k, ec in zip(indices, ecs)]
        out: list[TokenizedEdit] = []
        for row, (k, ec) in enumerate(zip(indices, ecs)):
            offsets = enc["offset_mapping"][row][:keep]
            status, r, h = _resolve(ec.ref_locus_span, ec.hyp_locus_span, offsets, P)
            ids = list(enc["input_ids"][row][:keep])
            out.append(TokenizedEdit(k, self.prefix_ids, ids, r, h, status, ec.text, P))
        return out


def tokenize_prefix(prep: PreparedPair, tokenizer, max_length: int) -> list[int]:
    """Token ids of the shared prefix (pair layer + ``"\\n\\n"``), truncated like the edits."""
    return list(tokenizer(prep.prefix_text, truncation=True, max_length=max_length)["input_ids"])


__all__ = [
    "ALIGNMENT_VERSION",
    "DEFAULT_CELL_BUDGET",
    "INPUT_TEMPLATE_VERSION",
    "PairTooLarge",
    "PreparedPair",
    "PairTokenizer",
    "TokenizedEdit",
    "prepare_pair",
    "tokenize_edits_full",
    "tokenize_prefix",
]
