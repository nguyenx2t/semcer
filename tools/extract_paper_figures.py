"""Read the plotted values out of the paper's vector figures, for the project page.

    curl -L -o eprint.tar.gz https://arxiv.org/e-print/2610.20461
    mkdir src && tar -xzf eprint.tar.gz -C src
    python tools/extract_paper_figures.py --figures src/figures/pre/onecol --output site/static/data/figures.json

The figures are vector PDFs, so each marker and curve vertex sits at the exact
coordinate the plotting code gave it. Axis ticks map coordinates back to data
values; the page then redraws the same numbers as interactive charts.
"""

from __future__ import annotations

import argparse
import json
import os

import pymupdf

LANG_CODE = {
    "Chinese": "zh",
    "Telugu": "te",
    "Kannada": "kn",
    "Tamil": "ta",
    "Thai": "th",
    "Marathi": "mr",
    "Hindi": "hi",
    "Arabic": "ar",
    "Urdu": "ur",
    "Vietnamese": "vi",
    "German": "de",
    "Russian": "ru",
    "Portuguese": "pt",
    "French": "fr",
    "English": "en",
    "Spanish": "es",
}
SCRIPTS = ("logographic", "abugida", "abjad", "alphabet")


def page(path: str):
    return pymupdf.open(path)[0]


def lin(p0: float, v0: float, p1: float, v1: float):
    return lambda p: v0 + (p - p0) * (v1 - v0) / (p1 - p0)


def center(r) -> tuple[float, float]:
    return (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2


def rgb(c) -> tuple:
    return tuple(round(x, 2) for x in c) if c else ()


def words(p) -> list[tuple[float, float, str]]:
    return [((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, w[4]) for w in p.get_text("words")]


def row_labels(p, names) -> list[tuple[float, str]]:
    return sorted((y, t) for _, y, t in words(p) if t in names)


def nearest(rows, y: float, tol: float = 5.0) -> str | None:
    best = min(rows, key=lambda r: abs(r[0] - y))
    return best[1] if abs(best[0] - y) <= tol else None


def ticks(p, y_min: float = 179.0, y_max: float = 183.5) -> list[float]:
    """x of the short black tick marks under an x axis."""
    out = []
    for d in p.get_drawings():
        it = d["items"]
        if len(it) == 1 and it[0][0] == "l" and rgb(d["color"]) == (0.0, 0.0, 0.0):
            a, b = it[0][1], it[0][2]
            if abs(a.x - b.x) < 0.01 and y_min <= min(a.y, b.y) and max(a.y, b.y) <= y_max:
                out.append(round(a.x, 2))
    return sorted(set(out))


def script_of(p, y: float) -> str:
    seps = sorted((wy, t) for _, wy, t in words(p) if t in SCRIPTS)
    return [t for wy, t in seps if wy < y][-1]


def review(path: str) -> dict:
    p = page(path)
    names = {
        (0.0, 0.0, 0.0): "readers",
        (0.06, 0.3, 0.57): "severity",
        (0.26, 0.58, 0.62): "judge_free",
        (0.71, 0.26, 0.26): "cer",
        (0.46, 0.46, 0.46): "random",
    }
    tk = ticks(p, 146.0, 149.5)  # 6 ticks per panel: 0, 10, ..., 50 %
    panels = [tk[i : i + 6] for i in range(0, 18, 6)]
    y = lin(144.28, 0.0, 33.4, 0.9)
    out = {lang: {} for lang in ("en", "vi", "zh")}
    for d in p.get_drawings():
        if len(d["items"]) != 26:
            continue
        pts = [d["items"][0][1]] + [it[2] for it in d["items"]]
        k = next(i for i, t in enumerate(panels) if t[0] - 1 <= pts[0].x <= t[-1] + 1)
        x = lin(panels[k][0], 0.0, panels[k][-1], 50.0)
        lang = ("en", "vi", "zh")[k]
        out[lang][names[rgb(d["color"])]] = [[round(x(q.x), 2), round(y(q.y), 4)] for q in pts]
    return out


def load(path: str) -> list[dict]:
    p = page(path)
    t0, t5, t1 = ticks(p)
    x = lin(t0, 0.0, t1, 1.0)
    rows = row_labels(p, LANG_CODE)
    out: dict[str, dict] = {}
    for d in p.get_drawings():
        r = d["rect"]
        if not (4.0 < r.width < 5.0) or center(r)[1] > 190:
            continue
        cx, cy = center(r)
        name = nearest(rows, cy)
        if name is None:
            continue
        rec = out.setdefault(name, {"lang": LANG_CODE[name], "name": name, "script": script_of(p, cy)})
        if len(d["items"]) == 8:  # open circle
            rec["mark_only"] = round(x(cx), 4)
        elif rgb(d["fill"]) == (0.3, 0.3, 0.3):  # filled square
            rec["content_sub"] = round(x(cx), 4)
    return sorted(out.values(), key=lambda r: [y for y, n in rows].index(next(y for y, n in rows if n == r["name"])))


def system(path: str) -> list[dict]:
    p = page(path)
    tk = ticks(p)
    left = lin(tk[0], 0.0, tk[2], 100.0)
    right = lin(tk[3], 0.4, tk[5], 0.8)
    split = (tk[2] + tk[3]) / 2
    rows = row_labels(p, LANG_CODE)
    classes = {(0.3, 0.3, 0.3): "classical", (0.26, 0.58, 0.62): "doc_vlm", (0.06, 0.3, 0.57): "general_vlm"}
    out: dict[str, dict] = {}

    def slot(name, cls):
        rec = out.setdefault(name, {"lang": LANG_CODE[name], "name": name, "script": None})
        return rec.setdefault(cls, {})

    for d in p.get_drawings():
        r = d["rect"]
        cx, cy = center(r)
        if cy > 190 or cy < 12:
            continue
        it = d["items"]
        name = nearest(rows, cy, tol=6.0)
        if name is None:
            continue
        if len(it) == 1 and it[0][0] == "l" and (d.get("width") or 0) > 1.2:  # range bar across a class's systems
            cls = classes.get(rgb(d["color"]))
            a, b = it[0][1], it[0][2]
            side, f = ("real_word", left) if a.x < split else ("severity", right)
            s = slot(name, cls)
            s[side + "_range"] = sorted([round(f(a.x), 4), round(f(b.x), 4)])
        elif d["fill"] is not None and rgb(d["fill"]) in classes and r.width < 7:
            cls = classes[rgb(d["fill"])]
            side, f = ("real_word", left) if cx < split else ("severity", right)
            slot(name, cls)[side] = round(f(cx), 4)
            out[name]["script"] = script_of(p, cy)
    order = [n for _, n in rows]
    return sorted(out.values(), key=lambda r: order.index(r["name"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--figures", required=True, help="directory with fig_review.pdf, fig_load.pdf, fig_system.pdf")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    f = lambda n: os.path.join(args.figures, n)  # noqa: E731
    data = {
        "source": "arXiv:2610.20461v1, read from the vector figures",
        "review": review(f("fig_review.pdf")),
        "load": load(f("fig_load.pdf")),
        "system": system(f("fig_system.pdf")),
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
