"""Deterministic synthetic (reference, hypothesis) pairs in the 16 training languages.

Not a test module. Corruptions are grapheme substitutions, insertions,
deletions, word drops, case flips and punctuation noise, so every edit type,
repeated graphemes, combining sequences and gaps at both ends occur.
"""

from __future__ import annotations

import random

import regex

SENTENCES = {
    "en": ["The total is 15 dollars.", "Invoice No. 2024-117, due 3 March.", "He said: \"don't wait\"."],
    "de": ["Die Größe beträgt 12 Meter.", "Straße und Übung für Anfänger.", "Das Ergebnis ist nicht klar."],
    "fr": ["Le garçon a déjà mangé.", "Élève numéro 7, classe B.", "Où est la bibliothèque ?"],
    "es": ["El niño tiene 5 años.", "¿Dónde está la estación?", "Mañana llegará el tren."],
    "pt": ["A ação custa 30 reais.", "Não há informação disponível.", "O coração da cidade."],
    "ru": ["Итого: 15 рублей.", "Москва — столица России.", "Он не пришёл вчера."],
    "ar": ["المجموع ١٥ دينارًا.", "ذهب الولد إلى المدرسة.", "هذا ليس صحيحا."],
    "hi": ["कुल राशि 15 रुपये है।", "वह कल स्कूल नहीं गया।", "भारत की राजधानी दिल्ली है।"],
    "mr": ["एकूण रक्कम १५ रुपये.", "तो काल शाळेत गेला नाही.", "महाराष्ट्राची राजधानी मुंबई."],
    "kn": ["ಒಟ್ಟು ಮೊತ್ತ 15 ರೂಪಾಯಿ.", "ಅವನು ನಿನ್ನೆ ಬರಲಿಲ್ಲ.", "ಬೆಂಗಳೂರು ನಗರ."],
    "ta": ["மொத்தம் 15 ரூபாய்.", "அவன் நேற்று வரவில்லை.", "சென்னை நகரம்."],
    "te": ["మొత్తం 15 రూపాయలు.", "అతను నిన్న రాలేదు.", "హైదరాబాద్ నగరం."],
    "th": ["ยอดรวม 15 บาท", "เขาไม่ได้มาเมื่อวาน", "กรุงเทพมหานคร"],
    "ur": ["کل رقم ۱۵ روپے ہے۔", "وہ کل اسکول نہیں گیا۔", "لاہور ایک شہر ہے۔"],
    "vi": ["Tổng cộng là 15 đồng.", "Huống hồ anh ấy không đến.", "Người được giải thưởng."],
    "zh": ["总计十五元。", "我们今天去学校。", "他昨天没有来，数字是12345。"],
}
LANGS = tuple(SENTENCES)
_NOISE = list("aeiouxyz0123456789.,;:!?-'\" ")


def _corrupt(text: str, rng: random.Random, rate: float) -> str:
    gs = regex.findall(r"\X", text)
    pool = sorted(set(gs)) + _NOISE
    out: list[str] = []
    for g in gs:
        r = rng.random()
        if r < rate / 3:
            continue  # deletion
        if r < 2 * rate / 3:
            out.append(rng.choice(pool))  # substitution
            continue
        out.append(g)
        if r < rate:
            out.append(rng.choice(pool))  # insertion
    if rng.random() < 0.2 and out:
        out = [g.upper() if rng.random() < 0.3 else g for g in out]
    return "".join(out)


def make_pairs(n: int = 200, seed: int = 20260924) -> list[dict]:
    """``n`` records ``{"id", "language", "reference", "hypothesis"}`` round-robin over LANGS."""
    rng = random.Random(seed)
    out = []
    for k in range(n):
        lang = LANGS[k % len(LANGS)]
        sents = SENTENCES[lang]
        ref = " ".join(rng.sample(sents, rng.randint(1, len(sents))))
        kind = k % 25
        if kind == 0:
            hyp = ref  # exact match
        elif kind == 1:
            hyp = ""  # all deletions
        elif kind == 2:
            hyp = ref + " " + ref[: rng.randint(1, 10)]  # trailing insertions
        else:
            hyp = _corrupt(ref, rng, rate=rng.choice([0.03, 0.1, 0.25]))
        out.append({"id": f"{lang}-{k:03d}", "language": lang, "reference": ref, "hypothesis": hyp})
    return out


def dense_pair(n_edits_target: int, seed: int = 7) -> tuple[str, str]:
    """A long pair with roughly ``n_edits_target`` edits (a runaway-style output)."""
    rng = random.Random(seed)
    base = " ".join(s for v in SENTENCES.values() for s in v)
    ref = (base * (1 + n_edits_target // max(1, len(base) // 3)))[: n_edits_target * 3]
    hyp = "".join(c if rng.random() > 0.33 else rng.choice(_NOISE) for c in ref)
    return ref, hyp
