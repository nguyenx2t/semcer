# semcer

**Equal CER, Unequal Damage: Severity-Weighted Character Error Rate for OCR**<br>
Nguyen Nguyen, Yongyi Zang

Paper (coming soon)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-semcer--qwen3.5--0.8b-yellow)](https://huggingface.co/nguyenx2t/semcer-qwen3.5-0.8b)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-61M%20judged%20edits-yellow)](https://huggingface.co/datasets/nguyenx2t/semcer-judged-edits)
[![Docs](https://img.shields.io/badge/docs-usage%20guide-blue)](https://github.com/nguyenx2t/semcer/blob/main/docs/USAGE.md)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenx2t/semcer/blob/main/notebooks/quickstart.ipynb)

![Six one-character OCR errors, each charged 1 by CER and a different severity by semcer](https://raw.githubusercontent.com/nguyenx2t/semcer/main/docs/assets/teaser.png)

CER counts every character error as 1. semcer uses the same alignment and
gives each error a severity from 0 (harmless) to 1 (meaning lost), predicted by
a 0.8B model that runs locally. The figure shows six one-character errors from
the paper, with the mean rating of human readers on the same scale.

A line's score is the sum of its severities divided by the reference length, so
semcer is at most CER.

## Install

```bash
pip install semcer
```

## Usage

```python
import semcer

result = semcer.score(reference="The total is 15 dollars.", hypothesis="The total is 75 dollars.")
print(result.explain())
# semcer=0.0414  CER=0.0417  (1 edit(s) over 24 reference graphemes; lower is better)
#   SUB ref[13:14] '1' -> '7'  severity 0.99 (meaning lost)
```

For many lines, load the scorer once. The model downloads on first use.

```python
scorer = semcer.SemCER()
results = scorer.score_batch(references=refs, hypotheses=hyps)
```

Or from the shell, with two text files that hold one line each per row:

```bash
semcer score --references refs.txt --hypotheses hyps.txt --output scores.jsonl
```

## Speed

| Device | Precision | Edits/s | Lines/s |
|---|---|---|---|
| CPU (12-vCPU Xeon, 2.2 GHz) | fp32 | 5.2 | 0.63 |
| NVIDIA T4 16 GB (free Colab) | fp32 | 46 | 5.5 |
| NVIDIA A100 40 GB | fp32 | 174 | 20.9 |
| NVIDIA A100 40 GB | bf16 (`dtype="bf16"`) | 286 | 34.5 |

Measured on 201 lines containing 1,667 edits; a larger corpus runs faster per
line. A GPU is used when available. bf16 moves a severity by up to 0.006.

## Citation

To be added with the paper.

## License

Apache-2.0 for the code and the model weights.

The scan crops in the figure come from
[OmniDocBench](https://github.com/opendatalab/OmniDocBench) and
[VieBookRead](https://huggingface.co/datasets/thaodd11/VieBookRead). They are
shown for illustration, remain under those datasets' terms, and are not covered
by this repository's license.
