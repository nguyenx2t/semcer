"""Build the project page's accent examples: one typical word per language and error type.

    python tools/build_site_accent_examples.py --data semcer-data \
        --figures site/static/data/figures.json --output site/static/data/accent_examples.json

From single-error lines in the dataset's redistributable ``text`` config, pick a
dropped accent (é -> e) and a wrong letter whose judge severity is within 0.08
of that language's mean in the paper's figure, preferring the most common word.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import unicodedata

import pandas as pd
import regex


def base(g: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", g) if not unicodedata.combining(c))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="local copy of nguyenx2t/semcer-judged-edits (text/ and pairs/)")
    ap.add_argument("--figures", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    text = pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(args.data, "text", "*.parquet")))
    pairs = pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(args.data, "pairs", "*.parquet")))
    m = text.merge(pairs, on=["lang", "dataset", "system", "id"])
    freq = {
        lang: collections.Counter(w.lower() for r in d.reference.drop_duplicates() for w in regex.findall(r"\p{L}+", r))
        for lang, d in m.groupby("lang")
    }
    m = m[m.text_consistent & (m.n_edits == 1) & (m.n_unscored == 0) & m.n_ref_graphemes.between(20, 120)]
    means = {r["lang"]: r for r in json.load(open(args.figures))["load"]}

    out = {}
    for lang, d in m.groupby("lang"):
        rows = []
        for x in d.itertuples():
            ref, hyp = regex.findall(r"\X", x.reference), regex.findall(r"\X", x.hypothesis)
            if len(ref) != len(hyp):
                continue
            diff = [i for i in range(len(ref)) if ref[i] != hyp[i]]
            if len(diff) != 1:
                continue
            i = diff[0]
            a, b = ref[i], hyp[i]
            if not (a.isalpha() and b.isalpha()):
                continue
            if b == base(a) and a != b:
                kind = "mark"  # an accent dropped
            elif base(a).lower() != base(b).lower():
                kind = "letter"
            else:
                continue
            lo, hi = i, i
            while lo > 0 and regex.match(r"\p{L}|\p{M}", ref[lo - 1]):
                lo -= 1
            while hi < len(ref) - 1 and regex.match(r"\p{L}|\p{M}", ref[hi + 1]):
                hi += 1
            w_ref, w_hyp = "".join(ref[lo : hi + 1]), "".join(hyp[lo : hi + 1])
            if 3 <= len(w_ref) <= 12:
                rows.append((kind, w_ref, w_hyp, x.severity_sum, freq[lang][w_ref.lower()]))
        f, pick = means.get(lang, {}), {}
        for kind, target in (("mark", f.get("mark_only")), ("letter", f.get("content_sub"))):
            if target is None or (lang == "en" and kind == "mark"):  # English text has too few accents to be typical
                continue
            near = [r for r in rows if r[0] == kind and abs(r[3] - target) <= 0.08]
            if near:
                best = max(near, key=lambda r: r[4])
                pick[kind] = {"ref": best[1], "hyp": best[2], "severity": round(best[3], 2)}
        out[lang] = pick
        print(lang, pick)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
