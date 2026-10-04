import json

from spine.data.build_sft import audit
from spine.data.synth import generate
from spine.schema import validate_record


def test_every_target_is_schema_valid():
    for row in generate(400, seed=3):
        assert validate_record(json.loads(row["target"])).ok


def test_labels_are_faithful_to_the_text():
    """No field may be labelled unless it is recoverable from the input."""
    for row in generate(600, seed=4):
        rec = json.loads(row["target"])
        if rec["status"] != "incident":
            continue
        inc, low = rec["incident"], row["text"].lower()
        assert inc["evidence_span"] in row["text"]
        if inc["device_hostname"]:
            assert inc["device_hostname"] in low
        if inc["site"]:
            assert inc["site"] in low
        if inc["ticket_id"]:
            assert inc["ticket_id"].lower() in low
        if inc["affected_clients"] is not None:
            assert str(inc["affected_clients"]) in row["text"]


def test_splits_do_not_share_examples():
    train = {r["text"] for r in generate(800, seed=5, split="train")}
    test = {r["text"] for r in generate(400, seed=7, split="test")}
    assert not train & test


def test_ood_split_uses_held_out_surface_styles():
    styles = {r["meta"]["style"] for r in generate(200, seed=9, split="ood")}
    assert styles <= {"cli", "handoff", "abstain"}


def test_abstain_rate_is_in_range():
    rows = generate(1000, seed=11)
    rate = sum(r["meta"]["label"] == "no_incident" for r in rows) / len(rows)
    assert 0.12 < rate < 0.26


def test_audit_catches_a_poisoned_row():
    rows = generate(50, seed=13)
    bad = json.loads(rows[0]["target"])
    if bad["status"] == "incident":
        bad["incident"]["severity"] = "apocalyptic"
    else:
        bad["reason"] = "made_up"
    rows[0]["target"] = json.dumps(bad)
    report, kept = audit(rows, "train")
    assert report["n_problems"] >= 1 and len(kept) == len(rows) - 1


def test_audit_drops_exact_duplicates():
    rows = generate(30, seed=17)
    rows.append(dict(rows[0], id="dup"))
    report, kept = audit(rows, "train")
    assert len(kept) == len(rows) - 1
