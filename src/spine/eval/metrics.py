"""Metric suite.

The headline number is `strict_exact_match`, but a single number hides where a
model is actually failing, so the suite reports five families:

1. **Parseability**  -- can the output be loaded as JSON at all, and did it need
   repair (fenced / substring recovery) to get there? A base model that only
   passes after repair is not production-ready; a fine-tune that needs no repair
   is a real deployment win.
2. **Schema validity** -- right keys, right types, closed vocabularies obeyed.
3. **Content accuracy** -- exact match on the whole record, plus per-field
   accuracy so you can see that (say) `started_at` is dragging the average down.
4. **Refusal behaviour** -- precision/recall/F1 on the abstain branch, plus the
   two asymmetric error rates that matter operationally:
   *missed abstain* (invented an incident from a lunch-menu email) and
   *over-abstain* (dropped a real P1 on the floor).
5. **Grounding** -- is `evidence_span` a verbatim substring of the input?
   This is the cheap, judge-free hallucination probe.

Pure stdlib so it runs in CI and on a CPU-only laptop.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable

from ..schema import SCORED_FIELDS, extract_json, normalize_record, validate_record


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def interfaces_f1(pred: list[str] | None, gold: list[str] | None) -> tuple[int, int, int]:
    ps = set(pred or [])
    gs = set(gold or [])
    return len(ps & gs), len(ps - gs), len(gs - ps)


def score_one(prediction: str, gold_target: str, source_text: str) -> dict[str, Any]:
    """Score a single completion. Returns a flat dict of per-example signals."""
    gold = json.loads(gold_target) if isinstance(gold_target, str) else gold_target
    gold_n = normalize_record(gold)

    pred_raw, mode = extract_json(prediction)
    out: dict[str, Any] = {
        "parse_mode": mode,
        "json_valid_strict": mode == "strict",
        "json_valid_lenient": mode != "failed",
        "schema_valid": False,
        "exact_match": False,
        "status_correct": False,
        "grounded": None,
        "gold_status": gold_n["status"],
        "pred_status": None,
        "field_correct": {},
        "iface_counts": (0, 0, 0),
        "errors": [],
    }
    if pred_raw is None:
        out["errors"].append("unparseable")
        return out

    v = validate_record(pred_raw)
    out["schema_valid"] = v.ok
    out["errors"].extend(v.errors)

    pred_n = normalize_record(pred_raw)
    out["pred_status"] = pred_n.get("status")
    out["status_correct"] = pred_n.get("status") == gold_n["status"]
    out["exact_match"] = out["schema_valid"] and pred_n == gold_n

    if gold_n["status"] == "incident" and pred_n.get("status") == "incident":
        gi, pi = gold_n["incident"], pred_n["incident"]
        for f in SCORED_FIELDS:
            if f == "interfaces":
                continue
            out["field_correct"][f] = pi.get(f) == gi.get(f)
        out["iface_counts"] = interfaces_f1(pi.get("interfaces"), gi.get("interfaces"))
        out["field_correct"]["interfaces"] = pi.get("interfaces") == gi.get("interfaces")
        span = (pred_raw.get("incident") or {}).get("evidence_span")
        out["grounded"] = bool(span) and isinstance(span, str) and span.strip() in source_text
    return out


def aggregate(per_example: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(per_example)
    n = len(rows)
    if n == 0:
        return {}

    agg: dict[str, Any] = {
        "n": n,
        "json_valid_strict": sum(r["json_valid_strict"] for r in rows) / n,
        "json_valid_lenient": sum(r["json_valid_lenient"] for r in rows) / n,
        "needed_repair": sum(r["parse_mode"] in ("fenced", "substring") for r in rows) / n,
        "schema_valid": sum(r["schema_valid"] for r in rows) / n,
        "exact_match": sum(r["exact_match"] for r in rows) / n,
        "status_accuracy": sum(r["status_correct"] for r in rows) / n,
    }

    # ---- refusal behaviour (positive class = "no_incident") ----
    tp = sum(r["gold_status"] == "no_incident" and r["pred_status"] == "no_incident" for r in rows)
    fp = sum(r["gold_status"] == "incident" and r["pred_status"] == "no_incident" for r in rows)
    fn = sum(r["gold_status"] == "no_incident" and r["pred_status"] != "no_incident" for r in rows)
    p, rc, f1 = _f1(tp, fp, fn)
    n_abstain_gold = sum(r["gold_status"] == "no_incident" for r in rows)
    n_incident_gold = n - n_abstain_gold
    agg.update(
        {
            "refusal_precision": p,
            "refusal_recall": rc,
            "refusal_f1": f1,
            # invented an incident out of noise
            "missed_abstain_rate": fn / n_abstain_gold if n_abstain_gold else 0.0,
            # bailed on a real incident
            "over_abstain_rate": fp / n_incident_gold if n_incident_gold else 0.0,
        }
    )
    # abstain reason correctness, among correctly-abstained cases
    agg["refusal_correctness"] = (
        sum(r["exact_match"] for r in rows if r["gold_status"] == "no_incident") / n_abstain_gold
        if n_abstain_gold
        else 0.0
    )

    # ---- per-field accuracy over the incident subset both sides agreed on ----
    hits: dict[str, int] = defaultdict(int)
    tot: dict[str, int] = defaultdict(int)
    for r in rows:
        for f, ok in r["field_correct"].items():
            tot[f] += 1
            hits[f] += bool(ok)
    agg["field_accuracy"] = {f: hits[f] / tot[f] for f in tot}
    agg["macro_field_accuracy"] = (
        sum(agg["field_accuracy"].values()) / len(agg["field_accuracy"]) if agg["field_accuracy"] else 0.0
    )
    # a field is only scored when both sides produced an incident; report coverage
    agg["field_scored_coverage"] = (max(tot.values()) / n_incident_gold) if n_incident_gold and tot else 0.0

    itp = sum(r["iface_counts"][0] for r in rows)
    ifp = sum(r["iface_counts"][1] for r in rows)
    ifn = sum(r["iface_counts"][2] for r in rows)
    ip, ir, iff = _f1(itp, ifp, ifn)
    agg.update({"interface_precision": ip, "interface_recall": ir, "interface_f1": iff})

    grounded = [r["grounded"] for r in rows if r["grounded"] is not None]
    agg["grounding_rate"] = sum(grounded) / len(grounded) if grounded else 0.0

    agg["parse_modes"] = dict(Counter(r["parse_mode"] for r in rows))
    agg["top_schema_errors"] = [
        e for e, _ in Counter(err.split(":")[0] for r in rows for err in r["errors"]).most_common(6)
    ]
    return agg


def score_dataset(predictions: list[str], rows: list[dict]) -> tuple[dict, list[dict]]:
    """`rows` are raw corpus rows with `text` and `target`."""
    if len(predictions) != len(rows):
        raise ValueError(f"{len(predictions)} predictions vs {len(rows)} rows")
    per = [score_one(p, r["target"], r["text"]) for p, r in zip(predictions, rows)]
    return aggregate(per), per


# --------------------------------------------------------------------------
# Programmatic reward used to build DPO preference pairs (phase 2)
# --------------------------------------------------------------------------

# NOTE: an earlier version also carried a separate "no_prose" term. It fired on
# exactly the same condition as "strict_json", so prose-wrapped-but-correct
# candidates were penalised twice and sometimes ranked below schema-invalid
# ones. One signal, one weight.
REWARD_WEIGHTS = {
    "strict_json": 1.0,
    "schema": 2.0,
    "status": 2.0,
    "fields": 4.0,
    "interfaces": 1.0,
    "grounding": 1.0,
}


def reward(prediction: str, gold_target: str, source_text: str) -> float:
    """Scalar quality in [0, 1] for one candidate completion.

    Deliberately dense rather than binary: a candidate that gets 9 of 11 fields
    right must outrank one that gets 4 right, otherwise almost every DPO pair
    would be a tie among valid-but-wrong outputs and the preference signal
    collapses.
    """
    s = score_one(prediction, gold_target, source_text)
    w = REWARD_WEIGHTS
    total = sum(w.values())
    got = 0.0
    got += w["strict_json"] * float(s["json_valid_strict"])
    got += w["schema"] * float(s["schema_valid"])
    got += w["status"] * float(s["status_correct"])
    if s["field_correct"]:
        got += w["fields"] * (sum(bool(v) for v in s["field_correct"].values()) / len(s["field_correct"]))
    elif s["gold_status"] == "no_incident" and s["exact_match"]:
        got += w["fields"]  # abstain cases have no fields; full credit when exact
    itp, ifp, ifn = s["iface_counts"]
    if itp + ifp + ifn:
        got += w["interfaces"] * _f1(itp, ifp, ifn)[2]
    else:
        got += w["interfaces"]
    if s["grounded"] is not None:
        got += w["grounding"] * float(s["grounded"])
    elif s["gold_status"] == "no_incident":
        got += w["grounding"]
    return got / total
