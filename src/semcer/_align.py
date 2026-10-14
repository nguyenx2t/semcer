"""Grapheme alignment: Wagner-Fischer over NFC extended grapheme clusters.

Ties break SUB > DEL > INS (diagonal > up > left in the backtrace). A deletion's
``j`` is the last matched hypothesis index before it (clamped to 0) and an
insertion's ``i`` is the reference position it precedes: the conventions the
severity model's input rendering assumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from semcer.graphemes import graphemes

Op = Literal["SUB", "INS", "DEL"]


@dataclass(frozen=True)
class GraphemeEdit:
    op: Op
    i: int  # reference grapheme index (0-based)
    j: int  # hypothesis grapheme index (0-based)
    g_r: str | None  # None for INS
    g_h: str | None  # None for DEL


def dp_cells(n_ref: int, n_hyp: int) -> int:
    """Size of the ``(n+1) x (m+1)`` DP table, checked against the cell budget."""
    return (n_ref + 1) * (n_hyp + 1)


def align_graphemes(r: str, h: str) -> tuple[list[str], list[str], list[GraphemeEdit]]:
    """NFC + grapheme split + alignment: ``(r_graphs, h_graphs, edits)``."""
    r_graphs = graphemes(r)
    h_graphs = graphemes(h)
    return r_graphs, h_graphs, align_grapheme_lists(r_graphs, h_graphs)


# Backtrace pointers, in tie-break order: SUB (diagonal) > DEL (up) > INS (left).
_DIAG, _UP, _LEFT = 0, 1, 2


def align_grapheme_lists(r_graphs: list[str], h_graphs: list[str]) -> list[GraphemeEdit]:
    """Edit script over two grapheme lists, in ascending path order (quadratic DP)."""
    n, m = len(r_graphs), len(h_graphs)
    # dp[i][j] = min edit cost between r[:i] and h[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = _UP
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = _LEFT

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = r_graphs[i - 1] == h_graphs[j - 1]
            sub_cost = dp[i - 1][j - 1] + (0 if same else 1)
            del_cost = dp[i - 1][j] + 1
            ins_cost = dp[i][j - 1] + 1

            best = sub_cost
            ptr = _DIAG
            if del_cost < best:
                best, ptr = del_cost, _UP
            if ins_cost < best:
                best, ptr = ins_cost, _LEFT
            dp[i][j] = best
            bt[i][j] = ptr

    # Backtrace from (n, m).
    edits: list[GraphemeEdit] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            ptr = _LEFT
        elif j == 0:
            ptr = _UP
        else:
            ptr = bt[i][j]

        if ptr == _DIAG:
            if r_graphs[i - 1] != h_graphs[j - 1]:
                edits.append(GraphemeEdit("SUB", i - 1, j - 1, r_graphs[i - 1], h_graphs[j - 1]))
            i -= 1
            j -= 1
        elif ptr == _UP:
            edits.append(GraphemeEdit("DEL", i - 1, max(j - 1, 0), r_graphs[i - 1], None))
            i -= 1
        else:  # _LEFT
            edits.append(GraphemeEdit("INS", i, j - 1, None, h_graphs[j - 1]))
            j -= 1

    edits.reverse()
    return edits
