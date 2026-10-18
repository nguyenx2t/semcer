"""Build the project page's gallery: lines where two systems tie on CER but not on damage.

    hf download nguyenx2t/semcer-judged-edits --repo-type dataset \
        --include "text/*" --include "pairs/*" --local-dir semcer-data
    python tools/build_site_gallery.py --data semcer-data --output site/static/data/gallery.json

Only the ``text`` config is read, so every line comes from a source whose license
allows redistribution. Candidates are drawn from the judge's severities in the
dataset; the page shows the released scorer's severities, recomputed here, so a
visitor who runs ``pip install semcer`` on a line gets the numbers shown.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import pandas as pd

import semcer

LICENSES = {
    "bressay": "CC BY 4.0",
    "coco_text": "CC BY 4.0",
    "german_hw": "AFL-3.0",
    "hiertext_lines": "CC BY-SA 4.0",
    "mozhi": "CC BY 4.0",
    "retas_lines": "Public domain (US)",
    "school_notebooks_ru": "MIT",
    "textocr": "CC BY 4.0",
    "thaiocrbench": "CC BY-SA 4.0",
}
LANGS = ["en", "de", "fr", "es", "pt", "ru", "ur"]


def candidates(data: str, per_lang: int, seed: int) -> list[dict]:
    text = pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(data, "text", "*.parquet")))
    pairs = pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(data, "pairs", "*.parquet")))
    m = text.merge(pairs, on=["lang", "dataset", "system", "id"])
    m = m[
        m.text_consistent
        & (m.n_unscored == 0)
        & m.n_edits.between(1, 2)
        & m.n_ref_graphemes.between(30, 80)
        & m.lang.isin(LANGS)
    ]
    out = []
    for (_, _), grp in m.groupby(["dataset", "id"]):
        for _, tie in grp.groupby("n_edits"):
            if len(tie) < 2 or tie.hypothesis.nunique() < 2:
                continue
            lo, hi = tie.loc[tie.severity_sum.idxmin()], tie.loc[tie.severity_sum.idxmax()]
            out.append({"gap": hi.severity_sum - lo.severity_sum, "a": lo, "b": hi})
    rng = random.Random(seed)
    picked = []
    for lang in LANGS:
        pool = sorted((c for c in out if c["a"].lang == lang), key=lambda c: -c["gap"])[: per_lang * 6]
        rng.shuffle(pool)
        picked.extend(pool[: per_lang * 3])  # scored below; the best per_lang survive
    return picked


def view(result: semcer.PairResult, system: str) -> dict:
    return {
        "system": system,
        "text": result.hypothesis,
        "semcer": round(result.semcer, 4),
        "edits": [
            {
                "op": e.op,
                "ref": e.ref_grapheme,
                "hyp": e.hyp_grapheme,
                "ref_span": list(e.ref_char_span),
                "hyp_span": list(e.hyp_char_span),
                "severity": round(e.severity, 3),
            }
            for e in result.edits
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="local copy of nguyenx2t/semcer-judged-edits")
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-lang", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scorer = semcer.SemCER()
    by_lang: dict[str, list] = {}
    for c in candidates(args.data, args.per_lang, args.seed):
        a, b = c["a"], c["b"]
        ra = scorer.score(reference=a.reference, hypothesis=a.hypothesis)
        rb = scorer.score(reference=b.reference, hypothesis=b.hypothesis)
        lo, hi = sorted([(ra, a.system), (rb, b.system)], key=lambda x: x[0].semcer)
        if hi[0].severity_sum - lo[0].severity_sum < 0.3:  # keep ties the released scorer also breaks
            continue
        by_lang.setdefault(a.lang, []).append(
            {
                "lang": a.lang,
                "dataset": a.dataset,
                "license": LICENSES[a.dataset],
                "id": a.id,
                "reference": lo[0].reference,
                "cer": round(lo[0].cer, 4),
                "outputs": [view(*lo), view(*hi)],
                "gap": hi[0].severity_sum - lo[0].severity_sum,
            }
        )
    items = []
    for lang in LANGS:
        best = sorted(by_lang.get(lang, []), key=lambda x: -x["gap"])[: args.per_lang]
        items.extend({k: v for k, v in x.items() if k != "gap"} for x in best)
    sig = scorer.signature
    meta = {k: sig[k] for k in ("semcer", "model", "revision", "weights", "precision")}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "items": items}, f, ensure_ascii=False, indent=1)
    print(f"{len(items)} items -> {args.output}")


if __name__ == "__main__":
    main()
