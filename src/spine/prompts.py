"""Prompt templates.

Two of them, and the difference is the point of the whole project:

* ``BASELINE_SYSTEM`` is the "best prompted base model" -- the full schema, the
  rules, and few-shot exemplars stuffed into context. This is the honest
  comparison point. Comparing a fine-tune against a lazy one-line prompt is how
  people fake fine-tuning wins.
* ``SFT_SYSTEM`` is one short line. After fine-tuning, the schema lives in the
  weights, not in the context window. Prompt tokens per request drop by ~95%,
  which is a latency and cost result you can put in the README next to the
  accuracy result.
"""

from __future__ import annotations

import json

from .schema import ABSTAIN_REASONS, ACTIONS, EVENT_TYPES, INCIDENT_FIELDS, SEVERITIES, VENDORS

SFT_SYSTEM = "Extract the network incident as JSON matching the SPINE incident schema. Output JSON only."

BASELINE_SYSTEM = f"""You are a network operations parser. Convert a messy free-text network \
report (alert email, chat message, ticket note, CLI paste) into a single JSON object.

Output ONLY the JSON object. No prose, no markdown fences, no explanation.

Two possible shapes.

(1) A real, actionable network incident:
{{
  "status": "incident",
  "incident": {{
    "device_hostname": string or null,
    "vendor": one of {list(VENDORS)},
    "site": string or null,
    "interfaces": array of interface names (may be empty),
    "event_type": one of {list(EVENT_TYPES)},
    "severity": one of {list(SEVERITIES)},
    "started_at": "YYYY-MM-DDTHH:MM:SSZ" or null,
    "ticket_id": string or null,
    "affected_clients": non-negative integer or null,
    "action_required": one of {list(ACTIONS)},
    "evidence_span": a VERBATIM substring copied from the input that justifies event_type
  }}
}}

(2) Not an actionable incident:
{{"status": "no_incident", "reason": one of {list(ABSTAIN_REASONS)}}}

Rules:
- Never invent a value. If the text does not state it, use null (or [] for interfaces).
- Normalize interface names to canonical form: GigE1/0/24, TenGigE1/0/2, Port-Channel12, Vlan400.
- Hostnames lowercase. Ticket IDs uppercase, no spaces.
- All timestamps in UTC. If the text gives a local time with an offset, convert it.
- evidence_span must appear character-for-character in the input.
- Use "no_incident" for maintenance announcements, marketing, small talk, resolved-notice
  duplicates, or text too vague to identify any event.
"""


def _ex(user: str, record: dict) -> list[dict]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(record, ensure_ascii=False)},
    ]


#: Few-shot exemplars for the prompted baseline. Kept to four (one abstain, one
#: multi-interface, one sparse, one CLI paste) so the baseline is strong but the
#: context stays under a realistic budget.
FEWSHOT: list[dict] = [
    *_ex(
        "URGENT [SITE: CHN-DC1] sw-core-02 (Comware 7) -- GE1/0/24 went down at "
        "2025-03-11 08:42:10 UTC. INC0044219. ~180 users offline. Need someone onsite.",
        {
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
        },
    ),
    *_ex(
        "hey team, reminder the cafeteria will be closed friday for deep cleaning. "
        "thanks!",
        {"status": "no_incident", "reason": "not_network_related"},
    ),
    *_ex(
        "aruba 6300 acc-blr-11: %LOG: ports 1/1/7 and 1/1/8 flapping since 14:05 IST, "
        "3rd time today. monitoring for now.",
        {
            "status": "incident",
            "incident": {
                "device_hostname": "acc-blr-11",
                "vendor": "aruba_aoscx",
                "site": None,
                "interfaces": ["1/1/7", "1/1/8"],
                "event_type": "link_flap",
                "severity": "minor",
                "started_at": "2025-01-01T08:35:00Z",
                "ticket_id": None,
                "affected_clients": None,
                "action_required": "monitor",
                "evidence_span": "ports 1/1/7 and 1/1/8 flapping",
            },
        },
    ),
    *_ex(
        "<134>Mar 4 22:18:03 rtr-edge-04 %%10OSPF/3/OSPF_NBR_CHG: Neighbour 10.20.0.9 "
        "changed from FULL to DOWN on Vlan-interface400. TAC case open: CS9911204.",
        {
            "status": "incident",
            "incident": {
                "device_hostname": "rtr-edge-04",
                "vendor": "hpe_comware",
                "site": None,
                "interfaces": ["Vlan400"],
                "event_type": "ospf_adjacency_down",
                "severity": "major",
                "started_at": None,
                "ticket_id": "CS9911204",
                "affected_clients": None,
                "action_required": "escalate_vendor_tac",
                "evidence_span": "changed from FULL to DOWN",
            },
        },
    ),
]


def baseline_messages(text: str, *, fewshot: bool = True) -> list[dict]:
    """Messages for the prompted (non-fine-tuned) baseline."""
    msgs = [{"role": "system", "content": BASELINE_SYSTEM}]
    if fewshot:
        msgs.extend(FEWSHOT)
    msgs.append({"role": "user", "content": text})
    return msgs


def sft_messages(text: str, target: str | None = None) -> list[dict]:
    """Messages for the fine-tuned model (and for building SFT training rows)."""
    msgs = [
        {"role": "system", "content": SFT_SYSTEM},
        {"role": "user", "content": text},
    ]
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs
