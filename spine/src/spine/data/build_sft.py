"""Turn the raw corpus into SFT-ready chat rows -- and refuse to do it if the
corpus is dirty.

The audit gate is the point. "Data quality beats data quantity" is easy to say
and easy to skip; this makes it a build step that fails loudly:

  * every target must parse as JSON and pass schema validation
  * every `evidence_span` must be a verbatim substring of its input
  * exact-duplicate inputs are dropped
  * near-duplicate inputs (token-shingle Jaccard > 0.9) are reported
  * train/val/test leakage is checked by text hash
  * the label distribution is printed so you notice a collapsed class before
    you spend four GPU-hours learning it

Emits `{"messages": [...]}` rows, which is what `trl.SFTTrainer` consumes
directly with `assistant_only_loss=True`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from ..prompts import sft_messages
from ..schema import validate_record

_TOK = re.compile(r"[a-z0-9]+")


def _shingles(text: str, k: int = 5) -> set[str]:
    toks = _TOK.findall(text.lower())
    return {" ".join(toks[i : i + k]) for i in range(max(1, len(toks) - k + 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def audit(rows: list[dict], name: str, *, near_dup_sample: int = 1500) -> tuple[dict, list[dict]]:
    report = {"split": name, "n_in": len(rows)}
    problems: list[str] = []
    seen: dict[str, int] = {}
    kept: list[dict] = []

    for i, r in enumerate(rows):
        try:
            rec = json.loads(r["target"])
        except json.JSONDecodeError:
            problems.append(f"[{i}] target is not JSON")
            continue
        v = validate_record(rec)
        if not v.ok:
            problems.append(f"[{i}] invalid target: {v.errors[:2]}")
            continue
        if rec["status"] == "incident":
            span = rec["incident"]["evidence_span"]
            if span not in r["text"]:
                problems.append(f"[{i}] evidence_span not verbatim in input")
                continue
        h = hashlib.sha1(r["text"].strip().encode()).hexdigest()
        if h in seen:
            problems.append(f"[{i}] exact duplicate of [{seen[h]}]")
            continue
        seen[h] = i
        r["_hash"] = h
        kept.append(r)

    # near-duplicate probe on a sample (O(n^2) on the full set is not worth it)
    sample = kept[:near_dup_sample]
    sh = [_shingles(r["text"]) for r in sample]
    near = 0
    for i in range(len(sample)):
        for j in range(i + 1, min(i + 60, len(sample))):
            if _jaccard(sh[i], sh[j]) > 0.9:
                near += 1
    report.update(
        {
            "n_kept": len(kept),
            "n_dropped": len(rows) - len(kept),
            "near_duplicate_pairs_in_sample": near,
            "label_distribution": dict(Counter(r["meta"]["label"] for r in kept)),
            "style_distribution": dict(Counter(r["meta"]["style"] for r in kept)),
            "event_distribution": dict(Counter(r["meta"]["event_type"] for r in kept)),
            "problems_sample": problems[:10],
            "n_problems": len(problems),
        }
    )
    return report, kept


def to_chat(rows: list[dict]) -> list[dict]:
    return [
        {"messages": sft_messages(r["text"], r["target"]), "id": r["id"], "meta": r["meta"]}
        for r in rows
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit the raw corpus and emit SFT chat rows.")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/sft"))
    ap.add_argument("--report", type=Path, default=Path("reports/data_audit.json"))
    ap.add_argument("--max-train", type=int, default=0, help="0 = keep all; used for the data-scaling ablation")
    ap.add_argument("--fail-on-problems", action="store_true")
    args = ap.parse_args()

    reports, hashes = [], {}
    args.out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test", "ood", "pool"):
        src = args.raw / f"{split}.jsonl"
        if not src.exists():
            continue
        rep, kept = audit(read_jsonl(src), split)
        # Short abstain notes can collide across splits by chance. Train wins;
        # every later split is deduped against it so the leakage check below is
        # satisfied by actually removing the overlap, not by tolerating it.
        if split != "train" and "train" in hashes:
            before = len(kept)
            kept = [r for r in kept if r["_hash"] not in hashes["train"]]
            rep["dropped_overlapping_train"] = before - len(kept)
            if before - len(kept):
                print(f"       de-leaked {before - len(kept)} rows shared with train")
        if split == "train" and args.max_train:
            kept = kept[: args.max_train]
            rep["n_kept_after_cap"] = len(kept)
        hashes[split] = {r["_hash"] for r in kept}
        reports.append(rep)
        for r in kept:
            r.pop("_hash", None)

        if split in ("train", "val"):
            dst = args.out / f"{split}.jsonl"
            with dst.open("w", encoding="utf-8") as fh:
                for row in to_chat(kept):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:  # eval splits stay in raw form; the harness applies prompts itself
            dst = args.out / f"{split}.jsonl"
            with dst.open("w", encoding="utf-8") as fh:
                for r in kept:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split:<6} kept {rep['n_kept']:>5} / {rep['n_in']:<5} dropped {rep['n_dropped']}  -> {dst}")

    leak = {}
    if "train" in hashes:
        for other in ("val", "test", "ood", "pool"):
            if other in hashes:
                leak[f"train_vs_{other}"] = len(hashes["train"] & hashes[other])
    print("leakage (shared input hashes):", leak)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"splits": reports, "leakage": leak}, indent=2))
    print(f"audit report -> {args.report}")

    total_problems = sum(r["n_problems"] for r in reports)
    if total_problems and args.fail_on_problems:
        raise SystemExit(f"data audit found {total_problems} problems; refusing to build")
    if any(leak.values()):
        raise SystemExit(f"train/eval leakage survived de-duplication: {leak}")


if __name__ == "__main__":
    main()
