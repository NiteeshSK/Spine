import json

from spine.eval.metrics import aggregate, reward, score_dataset, score_one

TEXT = "sw-core-02 GE1/0/24 went down at 2025-03-11 08:42:10 UTC, INC0044219, 180 users, need someone onsite"
GOLD = {
    "status": "incident",
    "incident": {
        "device_hostname": "sw-core-02",
        "vendor": "hpe_comware",
        "site": None,
        "interfaces": ["GigE1/0/24"],
        "event_type": "link_down",
        "severity": "critical",
        "started_at": "2025-03-11T08:42:10Z",
        "ticket_id": "INC0044219",
        "affected_clients": 180,
        "action_required": "dispatch_field_engineer",
        "evidence_span": "GE1/0/24 went down",
    },
}
TARGET = json.dumps(GOLD)


def test_perfect_prediction():
    s = score_one(TARGET, TARGET, TEXT)
    assert s["exact_match"] and s["schema_valid"] and s["json_valid_strict"] and s["grounded"]
    assert reward(TARGET, TARGET, TEXT) == 1.0


def test_prose_wrapper_costs_strictness_but_not_content():
    s = score_one(f"Here is the JSON:\n{TARGET}", TARGET, TEXT)
    assert s["exact_match"] and not s["json_valid_strict"] and s["parse_mode"] == "substring"
    assert 0.88 < reward(f"Here is the JSON:\n{TARGET}", TARGET, TEXT) < 1.0


def test_one_wrong_field_beats_five_wrong_fields():
    near = json.loads(TARGET)
    near["incident"]["severity"] = "major"
    far = json.loads(TARGET)
    far["incident"].update(severity="info", event_type="cpu_high", vendor="cisco_ios",
                           affected_clients=None, ticket_id=None)
    assert reward(json.dumps(near), TARGET, TEXT) > reward(json.dumps(far), TARGET, TEXT)


def test_ungrounded_evidence_is_penalised():
    bad = json.loads(TARGET)
    bad["incident"]["evidence_span"] = "fibre was cut by a backhoe"
    s = score_one(json.dumps(bad), TARGET, TEXT)
    assert s["grounded"] is False
    assert reward(json.dumps(bad), TARGET, TEXT) < 1.0


def test_unparseable_output():
    s = score_one("I'm sorry, I can't do that.", TARGET, TEXT)
    assert not s["json_valid_lenient"] and not s["exact_match"] and s["parse_mode"] == "failed"


def test_refusal_metrics_split_the_two_error_types():
    rows = [
        {"text": "lunch is at 1", "target": json.dumps({"status": "no_incident", "reason": "not_network_related"})},
        {"text": TEXT, "target": TARGET},
    ]
    # model invented an incident from the lunch note, and abstained on the real one
    preds = [TARGET, json.dumps({"status": "no_incident", "reason": "insufficient_detail"})]
    agg, _ = score_dataset(preds, rows)
    assert agg["missed_abstain_rate"] == 1.0
    assert agg["over_abstain_rate"] == 1.0
    assert agg["exact_match"] == 0.0


def test_aggregate_reports_field_accuracy():
    wrong = json.loads(TARGET)
    wrong["incident"]["severity"] = "minor"
    agg, _ = score_dataset([TARGET, json.dumps(wrong)], [{"text": TEXT, "target": TARGET}] * 2)
    assert agg["field_accuracy"]["severity"] == 0.5
    assert agg["field_accuracy"]["event_type"] == 1.0
    assert agg["exact_match"] == 0.5


def test_empty_aggregate_is_safe():
    assert aggregate([]) == {}
