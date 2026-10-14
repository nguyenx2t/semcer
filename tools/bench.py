"""Speed and parity of one scoring configuration (maintainers only).

    python tools/bench.py --model BUNDLE --device cuda --save cuda_fp32.json --compare cpu_fp32.json

Scores a fixed workload (``--input`` JSONL with reference/hypothesis, or
``--pairs N`` synthetic pairs in the 16 training languages from
``tests/_pairs.py``) and reports load time, preparation and model time,
edits/s and pairs/s, forwards, padding and peak device memory. ``--save`` writes
the report with every per-edit severity; ``--compare`` reads such a report (for
example CPU fp32) and gives the largest per-edit difference, the parity number
for a new device or precision.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _workload(args) -> list[dict]:
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    sys.path.insert(0, str(REPO / "tests"))
    from _pairs import dense_pair, make_pairs

    pairs = make_pairs(args.pairs)
    for n in args.dense:
        ref, hyp = dense_pair(n)
        pairs.append({"id": f"dense-{n}", "reference": ref, "hypothesis": hyp})
    return pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="bundle directory or Hub id (default: the release model)")
    ap.add_argument("--revision")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--input", help="JSONL workload; default is synthetic")
    ap.add_argument("--pairs", type=int, default=400, help="synthetic pairs")
    ap.add_argument("--dense", type=int, nargs="*", default=[300], help="edit counts of extra dense pairs")
    ap.add_argument("--batch-pairs", type=int)
    ap.add_argument("--max-batch-edits", type=int)
    ap.add_argument("--no-cross-pair", action="store_true", help="score pair by pair on any device")
    ap.add_argument("--repeat", type=int, default=1, help="timed passes after one warm-up pass")
    ap.add_argument("--warmup", type=int, default=16, help="pairs scored before timing")
    ap.add_argument("--save")
    ap.add_argument("--compare")
    args = ap.parse_args(argv)

    import torch

    from semcer import SemCER
    from semcer._prep import prepare_pair

    pairs = _workload(args)
    refs = [p["reference"] for p in pairs]
    hyps = [p["hypothesis"] for p in pairs]

    kwargs = {}
    if args.no_cross_pair:
        kwargs["batch_pairs"] = 1
    elif args.batch_pairs:
        kwargs["batch_pairs"] = args.batch_pairs
    if args.max_batch_edits:
        kwargs["max_batch_edits"] = args.max_batch_edits
    t0 = time.perf_counter()
    scorer = SemCER.from_pretrained(
        args.model, revision=args.revision, device=args.device, dtype=args.dtype, unscored="impute_one", **kwargs
    )
    load_s = time.perf_counter() - t0
    cuda = scorer.device.startswith("cuda")

    def sync():
        if cuda:
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    for r, h in zip(refs, hyps):
        prepare_pair(r, h)
    prep_s = time.perf_counter() - t0

    scorer.score_batch(refs[: args.warmup], hyps[: args.warmup], details="scores")
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    stats0 = vars(scorer.stats).copy()
    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        results = scorer.score_batch(refs, hyps)
        sync()
        times.append(time.perf_counter() - t0)
    total_s = min(times)
    stats = {k: (v - stats0[k]) // args.repeat for k, v in vars(scorer.stats).items() if not k.startswith("max_")}
    n_edits = sum(r.n_edits or 0 for r in results)
    n_scored = sum((r.n_edits or 0) - r.n_unscored for r in results)

    report = {
        "workload": args.input or f"synthetic: {args.pairs} pairs + dense {args.dense}",
        "n_pairs": len(pairs),
        "n_edits": n_edits,
        "n_scored_edits": n_scored,
        "signature": scorer.signature,
        "batch_pairs": scorer.batch_pairs,
        "max_batch_edits": scorer._engine.config.max_batch_edits,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
        },
        "load_s": round(load_s, 3),
        "prep_s": round(prep_s, 3),
        "total_s": round(total_s, 3),
        "total_s_all_passes": [round(t, 3) for t in times],
        "edits_per_s": round(n_scored / total_s, 2),
        "pairs_per_s": round(len(pairs) / total_s, 2),
        "prep_share": round(prep_s / total_s, 4),
        "stats": stats,
        "padding_fraction": round(scorer.stats.padding_fraction, 4),
        "peak_cuda_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
    }
    if args.compare:
        ref = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        worst, n, pair_worst = 0.0, 0, 0.0
        for mine, theirs, r, s in zip((res.severities for res in results), ref["severities"], results, ref["semcer"]):
            if len(mine) != len(theirs):
                raise SystemExit("the compared report has a different edit list; same workload?")
            for a, b in zip(mine, theirs):
                if (a is None) != (b is None):
                    raise SystemExit("the compared report scores a different set of edits")
                if a is not None:
                    worst, n = max(worst, abs(a - b)), n + 1
            if r.semcer is not None and s is not None:
                pair_worst = max(pair_worst, abs(r.semcer - s))
        report["compare"] = {
            "against": {k: ref["signature"][k] for k in ("device", "precision", "path", "gdn", "conv")},
            "n_edits": n,
            "max_abs_severity_diff": worst,
            "max_abs_pair_semcer_diff": pair_worst,
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.save:
        report["severities"] = [r.severities for r in results]
        report["semcer"] = [r.semcer for r in results]
        Path(args.save).write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
