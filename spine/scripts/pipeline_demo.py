"""End-to-end dry run of the whole harness with SIMULATED model outputs.

    python scripts/pipeline_demo.py

There is no model here. Three fake "models" corrupt the gold labels in
characteristic ways (a prompted base model that wraps output in prose and
invents fields; an SFT model that mostly nails the schema but fumbles
timestamps; a DPO model that fixes some of the residual). The point is to prove
the *plumbing* -- metrics, slicing, reports, comparison table, figures, DPO pair
mining -- runs correctly before you spend a single GPU-hour.

Every artifact it writes is stamped `demo_` and every figure is watermarked.
These are NOT results. Real numbers come from `make sft eval-sft`.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spine.eval.metrics import reward, score_dataset  # noqa: E402
from spine.schema import ABSTAIN_REASONS, SEVERITIES  # noqa: E402
from spine.viz import plots  # noqa: E402

DEMO_DIR = ROOT / "reports" / "demo"
ASSET_DIR = ROOT / "assets" / "pipeline_demo"


def corrupt(target: str, rng: random.Random, profile: str) -> str:
    """Turn a gold label into a plausible model output for `profile`."""
    rec = json.loads(target)
    p = {
        "base": dict(prose=0.45, extra=0.30, enum=0.35, ts=0.55, drop=0.25, abstain_flip=0.40, span=0.45, iface=0.40),
        "sft": dict(prose=0.02, extra=0.03, enum=0.06, ts=0.18, drop=0.04, abstain_flip=0.08, span=0.09, iface=0.10),
        "dpo": dict(prose=0.00, extra=0.01, enum=0.04, ts=0.10, drop=0.02, abstain_flip=0.04, span=0.03, iface=0.05),
    }[profile]

    if rec["status"] == "no_incident":
        if rng.random() < p["abstain_flip"]:
            return json.dumps(
                {
                    "status": "incident",
                    "incident": {
                        "device_hostname": None, "vendor": "unknown", "site": None, "interfaces": [],
                        "event_type": "unknown", "severity": "info", "started_at": None,
                        "ticket_id": None, "affected_clients": None, "action_required": "monitor",
                        "evidence_span": "n/a",
                    },
                }
            )
        if rng.random() < p["enum"]:
            rec["reason"] = rng.choice(ABSTAIN_REASONS)
        out = json.dumps(rec)
        return f"```json\n{out}\n```" if rng.random() < p["prose"] else out

    inc = rec["incident"]
    if rng.random() < p["abstain_flip"] * 0.35:
        return json.dumps({"status": "no_incident", "reason": "insufficient_detail"})
    if rng.random() < p["enum"]:
        inc["severity"] = rng.choice(SEVERITIES)
    if rng.random() < p["ts"] and inc["started_at"]:
        inc["started_at"] = inc["started_at"].replace("Z", "+05:30") if rng.random() < 0.5 else None
    if inc["interfaces"] and rng.random() < p["iface"]:
        inc["interfaces"] = ([] if rng.random() < 0.4
                             else [i.replace("GigE", "GigabitEthernet ") for i in inc["interfaces"]])
    if rng.random() < p["span"]:
        inc["evidence_span"] = "the interface reported a fault"
    if rng.random() < p["drop"]:
        inc.pop(rng.choice(["site", "ticket_id", "affected_clients"]), None)
    if rng.random() < p["extra"]:
        inc["root_cause"] = "suspected fibre issue"
    out = json.dumps(rec)
    if rng.random() < p["prose"]:
        out = f"Here is the extracted incident:\n\n```json\n{out}\n```\n\nLet me know if you need anything else."
    return out


def fake_train_log(path: Path, steps: int, kind: str) -> None:
    rng = random.Random(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    import math

    if kind == "sft":
        cols = ["step", "epoch", "loss", "eval_loss", "probe_json_valid_strict",
                "probe_schema_valid", "probe_exact_match", "probe_refusal_f1"]
        rows = []
        for s in range(10, steps + 1, 10):
            f = s / steps
            rows.append({
                "step": s, "epoch": round(3 * f, 3),
                "loss": round(1.62 * math.exp(-3.1 * f) + 0.11 + rng.uniform(-0.02, 0.02), 4),
                "eval_loss": round(1.55 * math.exp(-2.8 * f) + 0.13 + rng.uniform(-0.01, 0.01), 4) if s % 50 == 0 else "",
            })
            if s % 100 == 0:
                rows[-1].update({
                    "probe_json_valid_strict": round(min(0.995, 0.42 + 0.62 * f), 4),
                    "probe_schema_valid": round(min(0.99, 0.30 + 0.70 * f), 4),
                    "probe_exact_match": round(min(0.93, 0.05 + 0.90 * f ** 0.7), 4),
                    "probe_refusal_f1": round(min(0.97, 0.20 + 0.80 * f ** 0.6), 4),
                })
    else:
        cols = ["step", "epoch", "loss", "rewards/chosen", "rewards/rejected",
                "rewards/margins", "rewards/accuracies"]
        rows = []
        for s in range(5, steps + 1, 5):
            f = s / steps
            c = 0.42 * f + rng.uniform(-0.03, 0.03)
            r = -0.95 * f + rng.uniform(-0.04, 0.04)
            rows.append({
                "step": s, "epoch": round(f, 3),
                "loss": round(0.693 * math.exp(-2.4 * f) + 0.16 + rng.uniform(-0.01, 0.01), 4),
                "rewards/chosen": round(c, 4), "rewards/rejected": round(r, 4),
                "rewards/margins": round(c - r, 4),
                "rewards/accuracies": round(min(0.97, 0.52 + 0.42 * f), 4),
            })
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    data_dir = ROOT / "data" / "sft"
    split_file = data_dir / "test.jsonl"
    if not split_file.exists():
        raise SystemExit("run `make data` first (python -m spine.data.synth && python -m spine.data.build_sft)")

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    with split_file.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    for tag, profile, seed in [("demo_base", "base", 1), ("demo_sft", "sft", 2), ("demo_dpo", "dpo", 3)]:
        rng = random.Random(seed)
        preds = [corrupt(r["target"], rng, profile) for r in rows]
        agg, per = score_dataset(preds, rows)
        agg.update({"tag": tag, "split": "test", "model": f"SIMULATED::{profile}", "examples_per_second": None})
        slices: dict = {}
        for key in ("style", "event_type"):
            buckets: dict = {}
            for r, p in zip(rows, per):
                buckets.setdefault(r["meta"][key], []).append(p)
            slices[key] = {
                k: {"n": len(v), "exact_match": sum(x["exact_match"] for x in v) / len(v),
                    "schema_valid": sum(x["schema_valid"] for x in v) / len(v)}
                for k, v in sorted(buckets.items())
            }
        agg["slices"] = slices
        (DEMO_DIR / f"{tag}_test.json").write_text(json.dumps(agg, indent=2))
        print(f"{tag:<10} exact_match={agg['exact_match']:.3f} schema_valid={agg['schema_valid']:.3f} "
              f"json_strict={agg['json_valid_strict']:.3f} refusal_f1={agg['refusal_f1']:.3f}")

    # DPO pair mining, simulated: 6 samples per prompt from the "sft" model
    rng = random.Random(11)
    pool = rows[:200]
    kept, margins = 0, []
    for r in pool:
        cands = [corrupt(r["target"], rng, "sft") for _ in range(6)]
        scored = sorted(((reward(c, r["target"], r["text"]), c) for c in cands), key=lambda x: -x[0])
        if scored[-1][0] >= 1.0 or scored[0][0] - scored[-1][0] < 0.15:
            continue
        kept += 1
        margins.append(scored[0][0] - scored[-1][0])
    print(f"demo pair mining: {kept}/{len(pool)} prompts yielded a pair "
          f"(mean margin {sum(margins) / max(len(margins), 1):.3f})")

    fake_train_log(ROOT / "runs" / "demo_sft" / "train_log.csv", 900, "sft")
    fake_train_log(ROOT / "runs" / "demo_dpo" / "train_log.csv", 300, "dpo")

    plots.set_watermark("SIMULATED - PIPELINE DEMO")
    plots.plot_training_curves(ROOT / "runs/demo_sft/train_log.csv", ASSET_DIR / "sft_curves.png")
    plots.plot_dpo_curves(ROOT / "runs/demo_dpo/train_log.csv", ASSET_DIR / "dpo_curves.png")
    reps = [(t, json.loads((DEMO_DIR / f"{t}_test.json").read_text())) for t in ("demo_base", "demo_sft", "demo_dpo")]
    plots.plot_comparison(reps, ASSET_DIR / "comparison_test.png", title="test split (simulated)")
    plots.plot_field_accuracy(reps, ASSET_DIR / "field_accuracy_test.png")
    print(f"\nfigures -> {ASSET_DIR}\nreports -> {DEMO_DIR}")
    print("These are plumbing checks, not model results.")


if __name__ == "__main__":
    main()
