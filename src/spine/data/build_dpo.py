"""Phase 2, step 1 -- mine preference pairs from the SFT model's own samples.

    python -m spine.data.build_dpo --adapter runs/sft/adapter \\
        --pool data/sft/dpo_pool.jsonl --k 6 --temperature 0.9

Why the pairs are built this way
--------------------------------
*On-policy.* Candidates are sampled from the SFT checkpoint that DPO will start
from. Preference pairs harvested from some other model teach the policy about a
distribution it never visits, and the implicit-reward gradient mostly pushes
probability mass around off-manifold text.

*Programmatic, not vibes.* The label already exists, so the "annotator" is
`metrics.reward` -- a dense score over schema validity, per-field correctness,
interface F1, span grounding and prose-freeness. Dense matters: a binary
correct/incorrect judge makes almost every pair a tie between two wrong
answers, and the preference signal collapses.

*Margin-gated.* A pair only survives if best - worst >= ``--min-margin``. Pairs
separated by a rounding error are noise, and DPO will happily fit noise.

*Difficulty-weighted.* Prompts the SFT model already nails k-for-k contribute
nothing (no contrast) and are skipped. What is left is exactly the residual
failure set -- which is what phase 2 is for.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from ..eval.metrics import reward, score_one
from ..prompts import sft_messages


def failure_signature(prediction: str, target: str, text: str) -> str:
    """Coarse label for *why* a candidate is bad -- used for the pair-mix report."""
    s = score_one(prediction, target, text)
    if s["parse_mode"] == "failed":
        return "unparseable"
    if s["parse_mode"] != "strict":
        return "prose_wrapped"
    if not s["schema_valid"]:
        if any("hallucinated fields" in e for e in s["errors"]):
            return "extra_fields"
        if any("vocabulary" in e for e in s["errors"]):
            return "out_of_vocab_enum"
        if any("started_at" in e for e in s["errors"]):
            return "bad_timestamp_format"
        return "schema_other"
    if not s["status_correct"]:
        return "wrong_abstain_decision"
    if s["grounded"] is False:
        return "ungrounded_evidence"
    wrong = [k for k, v in s["field_correct"].items() if not v]
    return f"wrong_field:{wrong[0]}" if wrong else "near_miss"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DPO preference pairs from on-policy samples.")
    ap.add_argument("--pool", type=Path, default=Path("data/sft/dpo_pool.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/dpo/pairs.jsonl"))
    ap.add_argument("--report", type=Path, default=Path("reports/dpo_pairs.json"))
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default="runs/sft/adapter")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-margin", type=float, default=0.15)
    ap.add_argument("--solved-threshold", type=float, default=1.0, dest="solved_threshold",
                    help="skip a prompt only when EVERY sample already scores at least this "
                         "(no contrast left to learn from)")
    ap.add_argument("--gold-fallback", action="store_true",
                    help="if every sample is poor, use the gold target as `chosen` (off-policy; see README)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--candidates-out", type=Path, default=Path("data/dpo/candidates.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with args.pool.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[dpo-mine] {len(rows)} prompts, k={args.k} @ T={args.temperature}")

    from ..infer.generate import GenConfig, build_backend

    backend = build_backend(args.backend, args.model, args.adapter)
    cfg = GenConfig(
        max_new_tokens=320,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.k,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    chats = [sft_messages(r["text"]) for r in rows]
    samples = backend.generate(chats, cfg)
    backend.close()

    args.candidates_out.parent.mkdir(parents=True, exist_ok=True)
    with args.candidates_out.open("w", encoding="utf-8") as fh:
        for r, cands in zip(rows, samples):
            fh.write(json.dumps({"id": r["id"], "candidates": cands}, ensure_ascii=False) + "\n")

    pairs, stats = [], Counter()
    reward_hist: list[float] = []
    for r, cands in zip(rows, samples):
        scored = [(reward(c, r["target"], r["text"]), c) for c in cands]
        scored.sort(key=lambda x: -x[0])
        best_r, best = scored[0]
        worst_r, worst = scored[-1]
        reward_hist.append(best_r)

        if worst_r >= args.solved_threshold:
            stats["skipped_all_samples_solved"] += 1
            continue
        if best_r - worst_r < args.min_margin:
            if args.gold_fallback and best_r < 0.9:
                best, best_r = r["target"], 1.0
                stats["gold_fallback_used"] += 1
            else:
                stats["skipped_low_margin"] += 1
                continue
        if best.strip() == worst.strip():
            stats["skipped_identical"] += 1
            continue

        pairs.append(
            {
                "prompt": sft_messages(r["text"]),
                "chosen": [{"role": "assistant", "content": best.strip()}],
                "rejected": [{"role": "assistant", "content": worst.strip()}],
                "meta": {
                    "id": r["id"],
                    "chosen_reward": round(best_r, 4),
                    "rejected_reward": round(worst_r, 4),
                    "margin": round(best_r - worst_r, 4),
                    "rejected_failure": failure_signature(worst, r["target"], r["text"]),
                    **r.get("meta", {}),
                },
            }
        )
        stats["kept"] += 1

    rng.shuffle(pairs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    failure_mix = Counter(p["meta"]["rejected_failure"] for p in pairs)
    margins = [p["meta"]["margin"] for p in pairs] or [0.0]
    report = {
        "n_prompts": len(rows),
        "n_pairs": len(pairs),
        "yield": round(len(pairs) / max(len(rows), 1), 4),
        "k": args.k,
        "temperature": args.temperature,
        "min_margin": args.min_margin,
        "solved_threshold": args.solved_threshold,
        "stats": dict(stats),
        "mean_margin": round(sum(margins) / len(margins), 4),
        "mean_best_of_k_reward": round(sum(reward_hist) / max(len(reward_hist), 1), 4),
        "rejected_failure_mix": dict(failure_mix.most_common()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"[dpo-mine] pairs -> {args.out}")


if __name__ == "__main__":
    main()
