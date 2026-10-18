"""Build the project page's review examples: the paper's single-edit reader items.

    python tools/build_site_review_items.py --bundle task1_v2_bundle.zip \
        --ratings semcer-data/ratings/meaning.parquet --output site/static/data/review_items.json

Each item is one line with one error, rated by readers. The page sorts the items
by the judge's severity and by CER's charge, as in the paper's review curves, and
shows the wrong word of the items checked so far. Only the word around the error
is kept, not the line.
"""

from __future__ import annotations

import argparse
import json
import zipfile

import pandas as pd

LANGS = ("en", "vi", "zh")


def wrong_word(ref: str, hyp: str, lang: str) -> tuple[str, str]:
    """The word around the one difference between ``ref`` and ``hyp``."""
    i = 0
    while i < min(len(ref), len(hyp)) and ref[i] == hyp[i]:
        i += 1
    j = 0
    while j < min(len(ref), len(hyp)) - i and ref[-1 - j] == hyp[-1 - j]:
        j += 1
    if lang == "zh":  # no spaces: one character of context on each side
        lo, hi = max(0, i - 1), j - 1 if j > 0 else 0
    else:
        lo = ref.rfind(" ", 0, i) + 1
        right = ref.find(" ", len(ref) - j)
        hi = len(ref) - right if right != -1 else 0
    r = ref[lo : len(ref) - hi]
    h = hyp[lo : len(hyp) - hi]
    punct = " ,.;:!?\"'()"
    sr, sh = r.strip(punct), h.strip(punct)
    if sr and sh and sr != sh:  # trim punctuation only when the error is not in it
        return sr, sh
    return r, h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True, help="the reader-study bundle with _vetted_v2/<lang>_single.jsonl")
    ap.add_argument("--ratings", required=True, help="ratings/meaning.parquet from nguyenx2t/semcer-judged-edits")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ratings = pd.read_parquet(args.ratings)
    ratings = ratings[ratings.study == "single-edit"]
    out = {}
    with zipfile.ZipFile(args.bundle) as z:
        for lang in LANGS:
            lines = [json.loads(x) for x in z.read(f"_vetted_v2/{lang}_single.jsonl").decode().splitlines()]
            items = pd.DataFrame(lines).drop_duplicates(["system", "id"])
            m = ratings[ratings.lang == lang].merge(items, on=["system", "id", "dataset"], how="inner")
            rows = []
            for _, x in m.iterrows():  # keep the ratings table's order: the paper's ties break on it
                r, h = wrong_word(x.ground_truth, x.prediction, lang)
                rows.append(
                    {
                        "ref": r,
                        "hyp": h,
                        "readers": round(float(x.mean_rating), 3),
                        "severity": round(float(x.per_edit_severity), 3),
                        "cer": round(1 / int(x.n_r), 5),
                    }
                )
            out[lang] = rows
            print(lang, len(rows))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
