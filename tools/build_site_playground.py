"""Build the project page's playground: every one-character OCR error in a few lines.

    python tools/build_site_playground.py --output site/static/data/playground.json

For each sentence, each grapheme gets a few OCR-style confusions and a deletion.
Every resulting one-edit line is scored with the released scorer, so the page
can show any of them without a server. With one edit per line, CER is the same
for every choice; only the severity changes.
"""

from __future__ import annotations

import argparse
import json
import unicodedata

import semcer

SENTENCES = [
    {
        "lang": "en",
        "name": "English",
        "reference": "Take 2 tablets at 8:30, not after meals.",
        "gloss": None,
    },
    {
        "lang": "vi",
        "name": "Vietnamese",
        "reference": "Bệnh nhân uống 2 viên thuốc lúc 8 giờ sáng.",
        "gloss": "The patient takes 2 pills at 8 in the morning.",
    },
    {
        "lang": "zh",
        "name": "Chinese",
        "reference": "本月收入已达8600元，比上月多一倍。",
        "gloss": "This month's income reached 8,600 yuan, double last month's.",
    },
]

# Look-alike substitutions an OCR system makes, per lowercase letter or digit.
CONFUSE = {
    "0": "O86",
    "1": "7lI",
    "2": "Z7",
    "3": "85",
    "4": "9A",
    "5": "S6",
    "6": "58b",
    "7": "1T",
    "8": "3B0",
    "9": "g0",
    "a": "oe",
    "b": "h6",
    "c": "eo",
    "d": "cl",
    "e": "co",
    "f": "t",
    "g": "9q",
    "h": "bn",
    "i": "l1",
    "j": "i",
    "k": "h",
    "l": "1I",
    "m": "n",
    "n": "mh",
    "o": "0a",
    "p": "q",
    "q": "g",
    "r": "n",
    "s": "5",
    "t": "fl",
    "u": "vn",
    "v": "y",
    "w": "v",
    "x": "k",
    "y": "v",
    "z": "2",
    ".": ",:",
    ",": ".;",
    ":": ";.",
    "，": "、。",
    "。": "，.",
}
# Look-alike Chinese characters.
CONFUSE_ZH = {
    "本": "木末",
    "月": "目日",
    "收": "牧",
    "入": "人八",
    "已": "己巳",
    "达": "这",
    "元": "无兀",
    "比": "此",
    "上": "土止",
    "多": "夕",
    "一": "二",
    "倍": "培",
}
TONES = ["̀", "́", "̃", "̉", "̣"]  # grave, acute, tilde, hook, dot below


def vi_variants(g: str) -> list[str]:
    """Wrong tone mark, no tone mark, or no marks at all, for a Vietnamese vowel."""
    d = unicodedata.normalize("NFD", g)
    base, marks = d[0], d[1:]
    tone = [m for m in marks if m in TONES]
    shape = "".join(m for m in marks if m not in TONES)
    out = []
    if tone:
        out.append(base + shape)  # tone dropped
        for t in TONES:
            if t != tone[0] and len(out) < 3:
                out.append(base + shape + t)
    elif shape:
        out.append(base + shape + TONES[1])  # a tone added where there was none
    if marks:
        out.append(base)  # every mark dropped
    return [unicodedata.normalize("NFC", v) for v in out]


def options(g: str, lang: str) -> list[str]:
    if g.isspace():
        return [""]
    out: list[str] = []
    if lang == "zh" and g in CONFUSE_ZH:
        out.extend(CONFUSE_ZH[g])
    if lang == "vi" and len(unicodedata.normalize("NFD", g)) > 1:
        out.extend(vi_variants(g))
    if g == "đ":
        out.append("d")
    low = unicodedata.normalize("NFD", g)[0].lower()
    for c in CONFUSE.get(g, CONFUSE.get(low, "")):
        out.append(c.upper() if g.isupper() and c.isalpha() else c)
    if g.isalpha() and g.lower() != g.upper() and len(out) < 4:
        out.append(g.swapcase())
    out = [o for i, o in enumerate(out) if o != g and o not in out[:i]][:4]
    out.append("")  # deletion
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    scorer = semcer.SemCER()
    out = []
    for s in SENTENCES:
        ref = semcer.nfc(s["reference"])
        graphs = semcer.graphemes(ref)
        jobs = []
        for i, g in enumerate(graphs):
            for o in options(g, s["lang"]):
                jobs.append((i, o, "".join(graphs[:i]) + o + "".join(graphs[i + 1 :])))
        results = scorer.score_batch(references=[ref] * len(jobs), hypotheses=[h for _, _, h in jobs])
        per = [[] for _ in graphs]
        for (i, o, _), r in zip(jobs, results):
            if r.n_edits != 1:  # an option that aligns to more than one edit is not a one-character error
                continue
            per[i].append({"to": o, "severity": round(r.edits[0].severity, 3)})
        out.append({**s, "reference": ref, "graphemes": graphs, "options": per, "cer": round(1 / len(graphs), 4)})
        print(s["lang"], len(jobs), "lines scored")
    sig = scorer.signature
    meta = {k: sig[k] for k in ("semcer", "model", "revision", "weights", "precision")}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "sentences": out}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
