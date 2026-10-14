"""Model-input renderers: the pair layer and each edit's context block.

The severity model reads ``render_pair_layer(ref, hyp) + "\n\n" +
render_edit_context_with_spans(...).text``, byte-for-byte what it was trained
on; any change here is a new metric.
"""

from __future__ import annotations

from typing import NamedTuple

from semcer._align import GraphemeEdit


def render_pair_layer(r: str, h: str) -> str:
    """The (r, h) layer shared by every edit of a pair."""
    return f"REFERENCE:\n{r}\n\nHYPOTHESIS:\n{h}"


class EditContext(NamedTuple):
    """An edit block and the half-open, bracket-inclusive char spans (into
    ``text``) of its REF and HYP loci."""

    text: str
    ref_locus_span: tuple[int, int]
    hyp_locus_span: tuple[int, int]


def _window(graphs: list[str], pos: int, consume: bool, window: int) -> tuple[str, str]:
    """``(before, after)`` the bracket at ``pos``, with ``...`` where cut off.

    ``consume`` drops ``graphs[pos]`` (it goes in the bracket); otherwise the
    bracket is inserted before ``pos``.
    """
    n = len(graphs)
    skip = 1 if consume else 0
    left_start = max(0, pos - window)
    right_end = min(n, pos + window + skip)
    before = ("..." if left_start > 0 else "") + "".join(graphs[left_start:pos])
    after = "".join(graphs[pos + skip : right_end]) + ("..." if right_end < n else "")
    return before, after


def render_edit_context_with_spans(
    r_graphs: list[str],
    h_graphs: list[str],
    edit: GraphemeEdit,
    *,
    window: int = 10,
) -> EditContext:
    """Render one edit's block with its locus spans::

        Operation: SUB|INS|DEL
        REF: ...{left}[{g_r or ''}]{right}...
        HYP: ...{left}[{g_h or ''}]{right}...

    Each side shows ``window`` graphemes around the locus. An insertion has an
    empty REF bracket before ``i``; a deletion an empty HYP bracket. The spans
    come from the construction, never a text search.
    """
    op = edit.op
    if op == "INS":
        r_before, r_after = _window(r_graphs, edit.i, False, window)
        r_inner = ""
    else:
        r_before, r_after = _window(r_graphs, edit.i, True, window)
        r_inner = edit.g_r or ""

    if op == "DEL":
        # The aligner gives a deletion j = the last matched hyp index before it
        # (clamped to 0): the bracket goes after h[j] if h[j] matches r[i-1],
        # else before it.
        bracket_pos = 0
        if edit.i > 0 and h_graphs:
            matched = 0 <= edit.j < len(h_graphs) and h_graphs[edit.j] == r_graphs[edit.i - 1]
            bracket_pos = edit.j + 1 if matched else edit.j
        h_before, h_after = _window(h_graphs, bracket_pos, False, window)
        h_inner = ""
    else:
        h_before, h_after = _window(h_graphs, edit.j, True, window)
        h_inner = edit.g_h or ""

    head = f"Operation: {op}\nREF: "
    ref_line = f"{r_before}[{r_inner}]{r_after}"
    hyp_line = f"{h_before}[{h_inner}]{h_after}"
    text = f"{head}{ref_line}\nHYP: {hyp_line}\n"

    ref_start = len(head) + len(r_before)
    hyp_start = len(head) + len(ref_line) + len("\nHYP: ") + len(h_before)
    return EditContext(
        text=text,
        ref_locus_span=(ref_start, ref_start + len(r_inner) + 2),
        hyp_locus_span=(hyp_start, hyp_start + len(h_inner) + 2),
    )
