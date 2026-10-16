---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B-Base
library_name: semcer
tags:
- ocr
- evaluation
- metric
- cer
- semcer
language: [ar, de, en, es, fr, hi, kn, mr, pt, ru, ta, te, th, ur, vi, zh]
---

# semcer severity model

The scoring model of [semcer](https://github.com/nguyenx2t/semcer), a character
error rate for OCR that weights each edit by its severity, from the paper
[Equal CER, Unequal Damage: Severity-Weighted Character Error Rate for OCR](https://arxiv.org/abs/2610.20461).

Given a reference transcription and an OCR or VLM transcription, the model
rates every character edit from 0 (harmless) to 1 (meaning lost). A line's
semcer is the sum of those severities divided by the reference length.

## Use

```python
import semcer

result = semcer.score(reference="The total is 15 dollars.", hypothesis="The total is 75 dollars.")
print(result.explain())
```

The weights load through the `semcer` package, which downloads this repository
on first use.

## Model

| | |
|---|---|
| Backbone | `Qwen/Qwen3.5-0.8B-Base`, text model only |
| Input | reference and hypothesis, then the edit with ten graphemes of context on each side; up to 2,048 tokens |
| Output | probabilities over rubric levels 0 to 4; severity is the expected level divided by 4 |
| Files | `model.safetensors` (fp32, 3.0 GB), `model.bf16.safetensors` (1.5 GB) |

## Training

The model is distilled from an LLM judge, DeepSeek V4 Flash. The judge rated
each edit on a 0 to 4 rubric, and the model was trained for 100,000 steps to
predict the judge's distribution over the five levels. The training lines are
OCR and VLM transcriptions from 25 public datasets in 16 languages. No human
rating was used. The training labels are AI-generated.

## Agreement with the judge

On 1,623 held-out lines in 16 languages (75,082 edits):

| Measure | Value |
|---|---|
| Mean absolute error per edit | 0.038 |
| Clear edit pairs ordered as the judge orders them | 90.3% (95% interval 89.5 to 91.1) |
| The same, lowest and highest language | 87.3% and 95.1% |

A clear pair is two edits in one line whose judge severities differ by at least
0.125.

## Limitations

- A reference transcription is required.
- Severity differences below about 0.1 are not reliable.
- The judge was compared with human readers in English, Vietnamese and Chinese
  only.
- Edits past the 2,048-token window are not scored.
- The numbers above come from one training run.

## Citation

```bibtex
@misc{nguyen2026equalcer,
  title={Equal CER, Unequal Damage: Severity-Weighted Character Error Rate for OCR},
  author={Nguyen Nguyen and Yongyi Zang},
  year={2026},
  eprint={2610.20461},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2610.20461},
}
```

## License

Apache-2.0. The weights are a modified version of `Qwen/Qwen3.5-0.8B-Base`
(Copyright Alibaba Cloud, Apache-2.0). See `LICENSE` and `NOTICE`.
