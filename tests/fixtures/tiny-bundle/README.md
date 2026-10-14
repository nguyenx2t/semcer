# semcer severity student — DEV BUNDLE

> **Random head, not a metric.** This bundle has the release architecture and
> format with a randomly initialized head. It exists to exercise the scoring
> engine; its scores mean nothing. Do not publish it.

Predicts, for each aligned grapheme edit between a reference and a hypothesis
transcription, the teacher judge's 5-way severity distribution; severity =
E[digit]/4 in [0, 1]. semcer = sum(severity) / reference graphemes.

- backbone: `Qwen/Qwen3.5-0.8B-Base` @ `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` (Apache-2.0)
- head / readout: `distribution` / `span2_last`
- max input tokens: 2048 (edits past the window are reported, not scored)
- input template: `pair_layer+nn+edit_context_w10/1`
- alignment: `nfc_egc_wf_sub>del>ins/1`
- source checkpoint sha256: `ae445ee55e39b67941693221f71d725fcb7b2a7b31290b8171b506deef819019`
- training commit: `None`

Scores are student scores distilled from an LLM judge, not judge output.
Human validation covers English, Vietnamese and Chinese only.
