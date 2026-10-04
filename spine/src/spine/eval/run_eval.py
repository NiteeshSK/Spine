"""Evaluation harness.

Usage patterns
--------------
Score a checkpoint end to end::

    python -m spine.eval.run_eval --split test --backend vllm \\
        --model Qwen/Qwen3-8B --adapter runs/sft/adapter --tag sft

Score the honest baseline (base weights + full schema prompt + 4 shots)::

    python -m spine.eval.run_eval --split test --model Qwen/Qwen3-8B \\
        --prompt baseline --tag base_prompted

Re-score saved generations without touching a GPU (this is what CI does)::

    python -m spine.eval.run_eval --split test --predictions runs/sft/preds_test.jsonl --tag sft

Every run writes ``reports/<tag>_<split>.json`` (aggregate) and
``runs/.../preds_<split>.jsonl`` (per-example, with the failure reason) so the
comparison table in the README is reproducible from artifacts alone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..prompts import baseline_messages, sft_messages
from .metrics import score_dataset


def load_rows(path: Path, limit: int = 0) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return rows[:limit] if limit else rows


def build_chats(rows: list[dict], prompt: str) -> list[list[dict]]:
    if prompt == "baseline":
        return [baseline_messages(r["text"], fewshot=True) for r in rows]
    if prompt == "baseline_zeroshot":
        return [baseline_messages(r["text"], fewshot=False) for r in rows]
    return [sft_messages(r["text"]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a model or a predictions file on the SPINE task.")
    ap.add_argument("--data", type=Path, default=Path("data/sft"))
    ap.add_argument("--split", default="test", choices=["test", "ood", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", type=Path, default=Path("reports"))

    ap.add_argument("--predictions", type=Path, default=None, help="skip generation, score this jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--prompt", default="sft", choices=["sft", "baseline", "baseline_zeroshot"])
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--save-preds", type=Path, default=None)
    args = ap.parse_args()

    rows = load_rows(args.data / f"{args.split}.jsonl", args.limit)
    print(f"[eval] {len(rows)} rows from {args.split}")

    if args.predictions:
        with args.predictions.open(encoding="utf-8") as fh:
            saved = [json.loads(line) for line in fh if line.strip()]
        by_id = {s["id"]: s["prediction"] for s in saved}
        preds = [by_id[r["id"]] for r in rows]
        elapsed = 0.0
    else:
        from ..infer.generate import GenConfig, build_backend

        backend = build_backend(
            args.backend, args.model, args.adapter, load_in_4bit=not args.no_4bit
        )
        chats = build_chats(rows, args.prompt)
        cfg = GenConfig(max_new_tokens=args.max_new_tokens, temperature=0.0, n=1, batch_size=args.batch_size)
        t0 = time.time()
        outs = backend.generate(chats, cfg)
        elapsed = time.time() - t0
        preds = [o[0] for o in outs]
        backend.close()

        dest = args.save_preds or Path(f"runs/{args.tag}/preds_{args.split}.jsonl")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            for r, p in zip(rows, preds):
                fh.write(json.dumps({"id": r["id"], "prediction": p}, ensure_ascii=False) + "\n")
        print(f"[eval] generations -> {dest}  ({elapsed:.1f}s, {len(rows) / max(elapsed, 1e-9):.2f} ex/s)")

    agg, per = score_dataset(preds, rows)
    agg.update(
        {
            "tag": args.tag,
            "split": args.split,
            "model": args.model,
            "adapter": args.adapter,
            "prompt_style": args.prompt,
            "seconds": round(elapsed, 2),
            "examples_per_second": round(len(rows) / elapsed, 3) if elapsed else None,
        }
    )

    # slice the headline metric by surface style and by event type: an average
    # of 0.9 that hides a 0.4 on `handoff` notes is a report, not an insight
    slices: dict[str, dict] = {}
    for key in ("style", "event_type"):
        buckets: dict[str, list] = {}
        for r, p in zip(rows, per):
            buckets.setdefault(r["meta"][key], []).append(p)
        slices[key] = {
            k: {
                "n": len(v),
                "exact_match": sum(x["exact_match"] for x in v) / len(v),
                "schema_valid": sum(x["schema_valid"] for x in v) / len(v),
            }
            for k, v in sorted(buckets.items())
        }
    agg["slices"] = slices

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / f"{args.tag}_{args.split}.json"
    report_path.write_text(json.dumps(agg, indent=2))

    per_path = args.out / f"{args.tag}_{args.split}_per_example.jsonl"
    with per_path.open("w", encoding="utf-8") as fh:
        for r, p, pred in zip(rows, per, preds):
            fh.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "meta": r["meta"],
                        "exact_match": p["exact_match"],
                        "schema_valid": p["schema_valid"],
                        "parse_mode": p["parse_mode"],
                        "errors": p["errors"][:3],
                        "wrong_fields": [k for k, v in p["field_correct"].items() if not v],
                        "prediction": pred[:800],
                        "target": r["target"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(json.dumps({k: v for k, v in agg.items() if k not in ("slices", "field_accuracy")}, indent=2))
    print("field accuracy:", json.dumps(agg["field_accuracy"], indent=2))
    print(f"[eval] report -> {report_path}\n[eval] per-example -> {per_path}")


if __name__ == "__main__":
    main()
