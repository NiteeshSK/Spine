"""Task contract for SPINE.

The whole project lives or dies on this file. Everything downstream -- the
synthetic data generator, the SFT targets, the DPO reward, the eval metrics --
imports its notion of "correct" from here. If the schema is loose, the model
learns to be loose.

Design notes
------------
* Closed vocabularies (enums) wherever possible. A free-text `severity` field
  would make exact-match meaningless and would teach the model that
  "Critical", "critical" and "CRIT" are all fine. They are not.
* Exactly one nullable-string escape hatch per field that genuinely can be
  absent from the source text. Absent != hallucinate.
* `evidence_span` must be a VERBATIM substring of the input. This is the
  cheapest grounding signal you can get without a judge model, and it is what
  makes the DPO reward in phase 2 non-trivial.
* Two top-level branches: `incident` and `no_incident`. The second is the
  refusal path. A model that never abstains scores 100% on the happy path and
  is useless in a real NOC.

Pure stdlib on purpose: this module is imported by CPU-only smoke tests.
"""

from __future__ import annotations

import json
import re
from typing import Any

SCHEMA_VERSION = "spine-incident-1.0"

# --------------------------------------------------------------------------
# Closed vocabularies
# --------------------------------------------------------------------------

VENDORS = (
    "hpe_comware",
    "aruba_aoscx",
    "cisco_ios",
    "juniper_junos",
    "fortinet",
    "unknown",
)

EVENT_TYPES = (
    "link_down",
    "link_flap",
    "cpu_high",
    "memory_high",
    "psu_failure",
    "fan_failure",
    "bgp_peer_down",
    "ospf_adjacency_down",
    "stp_loop",
    "config_change",
    "auth_failure",
    "packet_loss",
    "temperature_high",
    "stack_split",
    "unknown",
)

SEVERITIES = ("critical", "major", "minor", "warning", "info")

ACTIONS = (
    "dispatch_field_engineer",
    "remote_remediation",
    "escalate_vendor_tac",
    "monitor",
    "none",
)

ABSTAIN_REASONS = (
    "not_network_related",
    "insufficient_detail",
    "informational_only",
    "duplicate_notice",
)

INCIDENT_FIELDS = (
    "device_hostname",
    "vendor",
    "site",
    "interfaces",
    "event_type",
    "severity",
    "started_at",
    "ticket_id",
    "affected_clients",
    "action_required",
    "evidence_span",
)

# Fields compared during per-field accuracy. `evidence_span` is scored
# separately (groundedness) because many spans are equally valid.
SCORED_FIELDS = tuple(f for f in INCIDENT_FIELDS if f != "evidence_span")

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

_IFACE_PREFIXES = [
    (re.compile(r"^(?:hundredgige|hu)", re.I), "HundredGigE"),
    (re.compile(r"^(?:fortygige|fo|fgig)", re.I), "FortyGigE"),
    (re.compile(r"^(?:twentyfivegige|twe)", re.I), "TwentyFiveGigE"),
    (re.compile(r"^(?:tengige|te|tengigabitethernet|xe)", re.I), "TenGigE"),
    (re.compile(r"^(?:gigabitethernet|gigabiteth|gige|gi|ge)", re.I), "GigE"),
    (re.compile(r"^(?:port-channel|po|bagg|ae|lag)", re.I), "Port-Channel"),
    (re.compile(r"^(?:vlan|vl)", re.I), "Vlan"),
    (re.compile(r"^(?:mgmt|management)", re.I), "Mgmt"),
]

_IFACE_SPLIT = re.compile(r"^([A-Za-z\-]+)\s*([\d/\.:]+)$")


def normalize_interface(raw: str) -> str:
    """Map every vendor spelling of an interface onto one canonical form.

    ``GigabitEthernet1/0/24``, ``Gi1/0/24``, ``ge-1/0/24`` and ``GE1/0/24`` are
    the same physical port. Without this, per-field accuracy would be measuring
    the model's ability to guess a house style rather than to read a log line.
    """
    s = raw.strip().rstrip(",.;")
    s = s.replace("_", "-")
    m = _IFACE_SPLIT.match(s)
    if not m:
        return s
    prefix, number = m.group(1), m.group(2)
    prefix = prefix.strip("-")
    for pattern, canon in _IFACE_PREFIXES:
        if pattern.match(prefix):
            return f"{canon}{number}"
    return f"{prefix}{number}"


def normalize_hostname(raw: str) -> str:
    return raw.strip().lower().rstrip(".")


def normalize_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field == "interfaces":
        if not isinstance(value, list):
            return value
        return sorted({normalize_interface(str(v)) for v in value})
    if field == "device_hostname":
        return normalize_hostname(str(value))
    if field in ("vendor", "event_type", "severity", "action_required"):
        return str(value).strip().lower()
    if field == "ticket_id":
        return str(value).strip().upper().replace(" ", "")
    if field == "site":
        return str(value).strip().lower()
    if field == "affected_clients":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return value


def normalize_record(record: dict) -> dict:
    """Canonicalize a full record so two semantically identical outputs compare equal."""
    if not isinstance(record, dict):
        return record
    out: dict[str, Any] = {"status": str(record.get("status", "")).strip().lower()}
    if out["status"] == "no_incident":
        out["reason"] = str(record.get("reason", "")).strip().lower()
        return out
    inc = record.get("incident") or {}
    if not isinstance(inc, dict):
        inc = {}
    out["incident"] = {f: normalize_value(f, inc.get(f)) for f in INCIDENT_FIELDS}
    return out


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


class ValidationResult:
    __slots__ = ("ok", "errors")

    def __init__(self, ok: bool, errors: list[str]):
        self.ok = ok
        self.errors = errors

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.ok

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ValidationResult(ok={self.ok}, errors={self.errors})"


def validate_record(record: Any) -> ValidationResult:
    """Structural + vocabulary validation. Does not check factual correctness."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ValidationResult(False, ["top level is not an object"])

    status = record.get("status")
    if status not in ("incident", "no_incident"):
        return ValidationResult(False, [f"bad status: {status!r}"])

    if status == "no_incident":
        extra = set(record) - {"status", "reason"}
        if extra:
            errors.append(f"unexpected keys for no_incident: {sorted(extra)}")
        if record.get("reason") not in ABSTAIN_REASONS:
            errors.append(f"bad reason: {record.get('reason')!r}")
        return ValidationResult(not errors, errors)

    extra = set(record) - {"status", "incident"}
    if extra:
        errors.append(f"unexpected top-level keys: {sorted(extra)}")

    inc = record.get("incident")
    if not isinstance(inc, dict):
        return ValidationResult(False, errors + ["incident is not an object"])

    missing = [f for f in INCIDENT_FIELDS if f not in inc]
    if missing:
        errors.append(f"missing fields: {missing}")
    unknown = sorted(set(inc) - set(INCIDENT_FIELDS))
    if unknown:
        errors.append(f"hallucinated fields: {unknown}")

    def _enum(name: str, allowed: tuple[str, ...], nullable: bool = False) -> None:
        v = inc.get(name)
        if v is None and nullable:
            return
        if not isinstance(v, str) or v.lower() not in allowed:
            errors.append(f"{name} not in vocabulary: {v!r}")

    _enum("vendor", VENDORS)
    _enum("event_type", EVENT_TYPES)
    _enum("severity", SEVERITIES)
    _enum("action_required", ACTIONS)

    for name in ("device_hostname", "site", "ticket_id", "evidence_span"):
        v = inc.get(name)
        if v is not None and not isinstance(v, str):
            errors.append(f"{name} must be string or null, got {type(v).__name__}")
    if not inc.get("evidence_span"):
        errors.append("evidence_span is required and must be non-empty")

    ifaces = inc.get("interfaces")
    if not isinstance(ifaces, list) or any(not isinstance(i, str) for i in ifaces):
        errors.append("interfaces must be a list of strings (possibly empty)")

    clients = inc.get("affected_clients")
    if clients is not None and (isinstance(clients, bool) or not isinstance(clients, int)):
        errors.append("affected_clients must be an integer or null")
    elif isinstance(clients, int) and not isinstance(clients, bool) and clients < 0:
        errors.append("affected_clients must be non-negative")

    started = inc.get("started_at")
    if started is not None and (not isinstance(started, str) or not ISO_RE.match(started)):
        errors.append(f"started_at must match YYYY-MM-DDTHH:MM:SSZ or be null, got {started!r}")

    return ValidationResult(not errors, errors)


# --------------------------------------------------------------------------
# Robust parsing of model output
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str, *, lenient: bool = True) -> tuple[dict | None, str]:
    """Parse a model completion into a dict.

    Returns ``(record, mode)`` where mode is one of ``strict`` (the completion
    was exactly one JSON object), ``fenced`` / ``substring`` (recovered), or
    ``failed``.

    We track the mode instead of silently repairing, because "the fine-tune
    stopped needing repair" is one of the headline results of phase 1.
    """
    if text is None:
        return None, "failed"
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj, "strict"
    except json.JSONDecodeError:
        pass
    if not lenient:
        return None, "failed"

    m = _FENCE_RE.search(stripped)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj, "fenced"
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(stripped[start : i + 1])
                        if isinstance(obj, dict):
                            return obj, "substring"
                    except json.JSONDecodeError:
                        pass
                    break
        start = stripped.find("{", start + 1)
    return None, "failed"


def dumps(record: dict) -> str:
    """Canonical serialization used for SFT targets: stable key order, no spaces waste."""
    return json.dumps(record, ensure_ascii=False, separators=(", ", ": "))


JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SPINE incident record",
    "version": SCHEMA_VERSION,
    "oneOf": [
        {
            "type": "object",
            "required": ["status", "incident"],
            "additionalProperties": False,
            "properties": {
                "status": {"const": "incident"},
                "incident": {
                    "type": "object",
                    "required": list(INCIDENT_FIELDS),
                    "additionalProperties": False,
                    "properties": {
                        "device_hostname": {"type": ["string", "null"]},
                        "vendor": {"enum": list(VENDORS)},
                        "site": {"type": ["string", "null"]},
                        "interfaces": {"type": "array", "items": {"type": "string"}},
                        "event_type": {"enum": list(EVENT_TYPES)},
                        "severity": {"enum": list(SEVERITIES)},
                        "started_at": {"type": ["string", "null"], "pattern": ISO_RE.pattern},
                        "ticket_id": {"type": ["string", "null"]},
                        "affected_clients": {"type": ["integer", "null"], "minimum": 0},
                        "action_required": {"enum": list(ACTIONS)},
                        "evidence_span": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
        {
            "type": "object",
            "required": ["status", "reason"],
            "additionalProperties": False,
            "properties": {
                "status": {"const": "no_incident"},
                "reason": {"enum": list(ABSTAIN_REASONS)},
            },
        },
    ],
}
