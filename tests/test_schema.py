import json

from spine.schema import (
    dumps,
    extract_json,
    normalize_interface,
    normalize_record,
    validate_record,
)

GOLD = {
    "status": "incident",
    "incident": {
        "device_hostname": "sw-core-02",
        "vendor": "hpe_comware",
        "site": "chn-dc1",
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


def test_interface_dialects_collapse():
    forms = ["GigabitEthernet1/0/24", "Gi1/0/24", "GE1/0/24", "ge-1/0/24", "gigabitethernet1/0/24"]
    assert len({normalize_interface(f) for f in forms}) == 1


def test_interface_families_stay_distinct():
    assert normalize_interface("Te1/0/2") != normalize_interface("Gi1/0/2")
    assert normalize_interface("Vlan400") == normalize_interface("vl400")
    assert normalize_interface("Port-Channel12") == normalize_interface("po12")


def test_valid_record_passes():
    assert validate_record(GOLD).ok


def test_missing_field_fails():
    bad = json.loads(json.dumps(GOLD))
    del bad["incident"]["severity"]
    assert not validate_record(bad).ok


def test_extra_field_is_flagged_as_hallucination():
    bad = json.loads(json.dumps(GOLD))
    bad["incident"]["root_cause"] = "fibre cut"
    res = validate_record(bad)
    assert not res.ok and any("hallucinated" in e for e in res.errors)


def test_out_of_vocabulary_enum_fails():
    bad = json.loads(json.dumps(GOLD))
    bad["incident"]["severity"] = "SUPER URGENT"
    assert not validate_record(bad).ok


def test_bad_timestamp_shape_fails():
    bad = json.loads(json.dumps(GOLD))
    bad["incident"]["started_at"] = "11/03/2025 08:42"
    assert not validate_record(bad).ok


def test_abstain_branch():
    assert validate_record({"status": "no_incident", "reason": "not_network_related"}).ok
    assert not validate_record({"status": "no_incident", "reason": "because"}).ok
    assert not validate_record({"status": "no_incident", "reason": "not_network_related", "incident": {}}).ok


def test_normalization_makes_equivalent_records_equal():
    other = json.loads(json.dumps(GOLD))
    other["incident"]["device_hostname"] = "SW-Core-02"
    other["incident"]["interfaces"] = ["gi1/0/24"]
    other["incident"]["ticket_id"] = "inc0044219"
    assert normalize_record(other) == normalize_record(GOLD)


def test_parse_modes():
    payload = dumps(GOLD)
    assert extract_json(payload)[1] == "strict"
    assert extract_json(f"```json\n{payload}\n```")[1] == "fenced"
    assert extract_json(f"Sure! Here you go:\n{payload}\nHope that helps.")[1] == "substring"
    assert extract_json("I cannot help with that.")[1] == "failed"


def test_substring_recovery_handles_braces_in_strings():
    rec = json.loads(json.dumps(GOLD))
    rec["incident"]["evidence_span"] = 'log said {down} on port'
    text = "here: " + json.dumps(rec)
    got, mode = extract_json(text)
    assert mode == "substring" and got["incident"]["evidence_span"] == 'log said {down} on port'
