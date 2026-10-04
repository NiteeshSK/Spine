"""Synthetic corpus generator: messy network-operations text -> gold SPINE records.

Why synthetic
-------------
There is no public dataset of "NOC free text + strict structured label", and
hand-labelling 6k examples is not a weekend. So the corpus is generated from a
*facts-first* pipeline:

    1. sample the ground-truth facts (device, event, severity, action, ...)
    2. render those facts into ONE of several messy surface forms
    3. the label is the facts -- it is never parsed back out of the text

Facts-first is what keeps the labels clean. If you generate text and then
regex the label out of it, your labels inherit every bug in your regex and the
model dutifully learns the bugs. (See docs/TROUBLESHOOTING.md, issue #1.)

Realism controls
----------------
* six surface styles (alert email, syslog, chat, ticket note, CLI paste,
  handoff note), two of them held out for the OOD split
* injected noise: typos, forwarded headers, signature blocks, HTML remnants,
  irrelevant chatter, inconsistent casing
* local timestamps with real offsets that must be converted to UTC
* four vendor dialects of the same interface (GE1/0/24 / Gi1/0/24 / ge-1/0/24)
* ~18% abstain cases, because a NOC inbox is mostly not incidents

Pure stdlib: runs on the CPU-only laptop in about a second for 8k rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
from pathlib import Path
from typing import Any

from ..schema import dumps, normalize_interface, validate_record

# --------------------------------------------------------------------------
# Entity pools. Train and test pools are DISJOINT so that "exact match" cannot
# be won by memorizing hostnames.
# --------------------------------------------------------------------------

SITES = {
    "train": ["chn-dc1", "blr-dc2", "hyd-cmp3", "pune-hub", "noida-br7", "ncr-edge2"],
    "test": ["kochi-dc4", "ahm-br2", "jaipur-hub", "vzg-cmp1"],
}
ROLES = ["sw-core", "sw-dist", "sw-acc", "rtr-edge", "fw-perim", "wlc"]

VENDOR_CUES = {
    "hpe_comware": ["Comware 7", "HPE 5940", "H3C/Comware", "HPE FlexNetwork", "comware v7"],
    "aruba_aoscx": ["Aruba CX 6300", "AOS-CX 10.10", "aruba 6405", "ArubaOS-CX"],
    "cisco_ios": ["Cisco C9300", "IOS-XE 17.6", "cat9k", "Cisco 4331 ISR"],
    "juniper_junos": ["Juniper EX4300", "Junos 21.4R3", "MX204", "junos"],
    "fortinet": ["FortiGate 200F", "FortiOS 7.2", "fortigate-hub"],
}

IFACE_DIALECT = {
    "hpe_comware": ["GE{a}/{b}/{c}", "GigabitEthernet{a}/{b}/{c}", "Ten-GigabitEthernet{a}/{b}/{c}"],
    "aruba_aoscx": ["{a}/{b}/{c}", "interface {a}/{b}/{c}"],
    "cisco_ios": ["Gi{a}/{b}/{c}", "GigabitEthernet{a}/{b}/{c}", "Te{a}/{b}/{c}"],
    "juniper_junos": ["ge-{a}/{b}/{c}", "xe-{a}/{b}/{c}"],
    "fortinet": ["port{c}", "wan{b}"],
}

TZ_OFFSETS = [("IST", 5.5), ("UTC", 0.0), ("GMT+8", 8.0), ("CET", 1.0), ("EST", -5.0)]

# --------------------------------------------------------------------------
# Event catalogue: body template + evidence template + default severity/action
# --------------------------------------------------------------------------

EVENTS: dict[str, list[dict[str, str]]] = {
    "link_down": [
        {"body": "{iface} on {host} went down at {time}", "ev": "{iface} on {host} went down"},
        {"body": "%%10IFNET/3/PHY_UPDOWN: {iface} changed to down state", "ev": "{iface} changed to down state"},
        {"body": "link failure detected on {iface} ({host}) since {time}", "ev": "link failure detected on {iface}"},
    ],
    "link_flap": [
        {"body": "{iface} on {host} is flapping, 6 transitions since {time}", "ev": "{iface} on {host} is flapping"},
        {"body": "repeated up/down on {iface}, seen 4x in the last hour", "ev": "repeated up/down on {iface}"},
    ],
    "cpu_high": [
        {"body": "{host} control-plane CPU pegged at 97% since {time}", "ev": "CPU pegged at 97%"},
        {"body": "CPU utilization on {host} sustained above 90 percent", "ev": "CPU utilization on {host} sustained above 90 percent"},
    ],
    "memory_high": [
        {"body": "{host} memory usage 94%, buffer allocation failures logged", "ev": "memory usage 94%"},
    ],
    "psu_failure": [
        {"body": "PSU 2 on {host} reporting failed status at {time}", "ev": "PSU 2 on {host} reporting failed status"},
        {"body": "power supply unit 1 absent/failed, unit running on single feed", "ev": "power supply unit 1 absent/failed"},
    ],
    "fan_failure": [
        {"body": "fan tray 3 in {host} stopped, chassis temp climbing", "ev": "fan tray 3 in {host} stopped"},
    ],
    "bgp_peer_down": [
        {"body": "BGP neighbour 10.{n}.0.1 on {host} down (Hold Timer Expired) at {time}", "ev": "BGP neighbour 10.{n}.0.1 on {host} down"},
        {"body": "peer AS6510{n} session dropped on {host}, prefixes withdrawn", "ev": "peer AS6510{n} session dropped"},
    ],
    "ospf_adjacency_down": [
        {"body": "OSPF neighbour 10.{n}.0.9 changed from FULL to DOWN on {iface}", "ev": "changed from FULL to DOWN"},
    ],
    "stp_loop": [
        {"body": "possible L2 loop, MAC flapping between {iface} and {iface2} on {host}", "ev": "MAC flapping between {iface} and {iface2}"},
        {"body": "BPDU storm on {host}, root bridge changed twice", "ev": "BPDU storm on {host}"},
    ],
    "config_change": [
        {"body": "unauthorised running-config change on {host} at {time} by user netops2", "ev": "unauthorised running-config change on {host}"},
    ],
    "auth_failure": [
        {"body": "18 failed SSH logins on {host} from 192.168.{n}.44 in 5 minutes", "ev": "18 failed SSH logins on {host}"},
    ],
    "packet_loss": [
        {"body": "12% packet loss across {iface} on {host}, latency 340ms", "ev": "12% packet loss across {iface}"},
        {"body": "users report drops; ping to {host} shows 9% loss sustained", "ev": "ping to {host} shows 9% loss sustained"},
    ],
    "temperature_high": [
        {"body": "{host} inlet temperature 61C, above major threshold", "ev": "inlet temperature 61C"},
    ],
    "stack_split": [
        {"body": "IRF stack on {host} split, member 2 now standalone master", "ev": "IRF stack on {host} split"},
    ],
}

SEV_PHRASES = {
    "critical": ["P1", "SEV1", "CRITICAL", "Priority: 1 (critical)", "sev-1"],
    "major": ["P2", "SEV2", "Major", "Priority: 2"],
    "minor": ["P3", "minor", "Priority: 3"],
    "warning": ["P4", "warning", "low priority"],
    "info": ["FYI", "informational", "P5"],
}

ACTION_PHRASES = {
    "dispatch_field_engineer": ["need someone onsite", "rolling a truck", "field engineer required", "smart hands requested"],
    "remote_remediation": ["will bounce the port remotely", "fixing via SSH now", "remote reload scheduled"],
    "escalate_vendor_tac": ["TAC case opened", "escalating to vendor support", "raised with HPE TAC"],
    "monitor": ["monitoring for now", "keeping an eye on it", "no action yet, watching"],
    "none": ["already recovered on its own", "self-healed, logging only"],
}

# Deterministic labelling policy. Documented here so a reviewer can audit it.
DEFAULT_SEVERITY = {
    "link_down": "critical", "psu_failure": "major", "fan_failure": "major",
    "bgp_peer_down": "critical", "ospf_adjacency_down": "major", "stack_split": "critical",
    "stp_loop": "critical", "cpu_high": "major", "memory_high": "major",
    "temperature_high": "major", "packet_loss": "major", "link_flap": "minor",
    "config_change": "warning", "auth_failure": "warning",
}
DEFAULT_ACTION = {
    "link_down": "dispatch_field_engineer", "psu_failure": "dispatch_field_engineer",
    "fan_failure": "dispatch_field_engineer", "temperature_high": "dispatch_field_engineer",
    "bgp_peer_down": "escalate_vendor_tac", "ospf_adjacency_down": "escalate_vendor_tac",
    "stack_split": "escalate_vendor_tac", "stp_loop": "remote_remediation",
    "cpu_high": "remote_remediation", "memory_high": "remote_remediation",
    "link_flap": "monitor", "packet_loss": "monitor",
    "config_change": "remote_remediation", "auth_failure": "monitor",
}

NEEDS_IFACE = {"link_down", "link_flap", "stp_loop", "packet_loss", "ospf_adjacency_down"}

# --------------------------------------------------------------------------
# Abstain corpus
# --------------------------------------------------------------------------

# (reason, template, group). Groups are disjoint across train and eval so that
# the abstain branch is tested on phrasings the model has never seen -- the
# whole point of the refusal metric is generalisation, not recall.
ABSTAIN_POOL: list[tuple[str, str, str]] = [
    ("not_network_related", "Reminder: timesheets are due friday 6pm. Please submit before EOD, chasing people down is nobody's idea of fun.", "train"),
    ("not_network_related", "the printer on 3rd floor is out of toner again. raised a facilities ticket FAC-8891.", "train"),
    ("not_network_related", "Team lunch moved to 1:30 because the meeting room got double booked.", "train"),
    ("not_network_related", "Payroll portal will be down saturday for the annual audit. HR will send a note.", "train"),
    ("not_network_related", "anyone have the parking pass form? new security guy wont let me in", "eval"),
    ("not_network_related", "Congrats to Priya on 5 years with the team! Cake in the pantry at 4.", "eval"),
    ("insufficient_detail", "network seems slow today?", "train"),
    ("insufficient_detail", "something is wrong, please check", "train"),
    ("insufficient_detail", "hey can you look at the thing we discussed yesterday, its acting up again", "train"),
    ("insufficient_detail", "issue reported by user. no further details provided at this time.", "train"),
    ("insufficient_detail", "escalated. see attached screenshot.", "eval"),
    ("insufficient_detail", "customer called about connectivity. call dropped before details captured. will follow up.", "eval"),
    ("informational_only", "PLANNED MAINTENANCE NOTICE: code upgrade window on the {site} core stack, Saturday 02:00-05:00 IST. No action required, this is a scheduled change under CHG0042118.", "train"),
    ("informational_only", "Weekly capacity report attached. All links below 60% utilisation. Nothing anomalous this week.", "train"),
    ("informational_only", "FYI new firmware 10.11.1030 is now on the approved list for the access layer. Rollout planning starts next quarter.", "eval"),
    ("informational_only", "Reminder: quarterly DR drill for {site} is on the 28th. Runbook link in the wiki.", "eval"),
    ("duplicate_notice", "RESOLVED: earlier alert on {host} has cleared. Link is stable, closing INC{tid}. No further action.", "train"),
    ("duplicate_notice", "*** AUTO-CLOSE *** Alarm on {host} recovered after 4 minutes and has been auto-cleared by the platform. Duplicate of the notification sent earlier.", "train"),
    ("duplicate_notice", "This is a repeat of the notification already sent at 09:14. Original ticket INC{tid} is being worked. Please disregard.", "eval"),
    ("not_network_related", "Badge access to the 4th floor lab is being re-provisioned. Contact security if yours stops working.", "train"),
    ("not_network_related", "The espresso machine on floor {num} is out of order. Vendor visit booked for next week.", "train"),
    ("not_network_related", "Laptop refresh cycle starts next month. IT will reach out individually to schedule swaps.", "train"),
    ("not_network_related", "Fire drill on the {num}th at 11am. Everyone assembles in the east car park.", "eval"),
    ("insufficient_detail", "there's an issue on one of the boxes, will update once I know more", "train"),
    ("insufficient_detail", "getting complaints. checking. no device identified yet.", "train"),
    ("insufficient_detail", "user says it broke again. same as before. no logs attached.", "train"),
    ("insufficient_detail", "can someone take a look when free, seems off since morning", "eval"),
    ("informational_only", "CHANGE APPROVED: CHG{num}00 scheduled for the {site} distribution layer next quarter. Informational only, no impact expected.", "train"),
    ("informational_only", "Monthly SLA report published to the portal. All circuits met the 99.9% target.", "train"),
    ("informational_only", "New NOC escalation matrix is live on the wiki. Please bookmark it, the old page is retired.", "eval"),
    ("duplicate_notice", "CLEARED: the alarm raised on {host} at 09:02 has cleared by itself. This is the auto-generated clear notification for INC{tid}.", "train"),
    ("duplicate_notice", "Duplicate: same condition on {host} already tracked under INC{tid}. Merging this notification into the parent ticket.", "train"),
    ("duplicate_notice", "Re-notification (attempt 3 of 3) for an event on {host} that has already been acknowledged. No new action needed.", "eval"),
]

# --------------------------------------------------------------------------
# Noise injection
# --------------------------------------------------------------------------

FORWARD_HEADERS = [
    "---------- Forwarded message ----------\nFrom: NMS Alerting <noreply@nms.corp>\nTo: noc-l1@corp\n",
    "> On Tue, 11 Mar at 08:44, monitoring wrote:\n> ",
    "FW: FW: RE: ",
]
SIGNATURES = [
    "\n\n--\nSent from my iPhone",
    "\n\nRegards,\nL1 NOC\nExt 4412",
    "\n\n[This message is confidential. If you are not the intended recipient, delete it.]",
    "\n<div dir=\"ltr\"><br clear=\"all\"></div>",
]
CHATTER = [
    "btw the coffee machine is broken again",
    "ignore my last message, wrong channel",
    "cc @rahul for visibility",
    "(third time this month btw)",
    "sorry for the wall of text",
]
TYPO_MAP = {"the": "teh", "down": "dwon", "interface": "inteface", "network": "netowrk", "and": "adn"}


def _typo(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    for i, w in enumerate(words):
        key = w.lower()
        if key in TYPO_MAP and rng.random() < 0.35:
            words[i] = TYPO_MAP[key]
    return " ".join(words)


# --------------------------------------------------------------------------
# Surface styles
# --------------------------------------------------------------------------


def _style_email(f: dict, rng: random.Random) -> str:
    subject = rng.choice([
        f"[{f['sev_phrase']}] {f['event_human']} - {f['host']}",
        f"[{f['sev_phrase']}] alarm on {f['host']}",
        "Automated NMS notification",
        f"{f['host']} - please check",
    ])
    head = f"Subject: {subject}\n"
    if f["site"]:
        head += f"Site: {f['site'].upper()}\n"
    if f["has_time"]:
        head += f"Alert time: {f['time']}\n"
    body = f"{f['vendor_cue']}\n\n{f['body']}.\n"
    if f["ticket"]:
        body += f"Ticket: {f['ticket']}\n"
    if f["clients"] is not None:
        body += f"Approx {f['clients']} users affected.\n"
    body += f"{f['action_phrase']}.\n"
    out = head + body
    if rng.random() < 0.4:
        out = rng.choice(FORWARD_HEADERS) + out
    if rng.random() < 0.5:
        out += rng.choice(SIGNATURES)
    return out


def _style_syslog(f: dict, rng: random.Random) -> str:
    pri = rng.choice([134, 187, 27, 165])
    # RFC5424-ish stamp with an explicit offset: without the offset the UTC
    # conversion in the gold label would be underivable from the text.
    stamp = f"{f['local_dt']:%Y-%m-%dT%H:%M:%S}{f['tz_iso']}"
    line = f"<{pri}>{stamp} {f['host']} {f['body']}"
    tail = f"  # {f['vendor_cue']}"
    if f["ticket"]:
        tail += f" | {f['ticket']}"
    tail += f" | {f['sev_phrase']} | {f['action_phrase']}"
    return line + tail


def _style_chat(f: dict, rng: random.Random) -> str:
    lead = rng.choice(["hey ", "guys ", "", "@here ", "ok so "])
    txt = f"{lead}{f['body'].lower()}"
    if f["host"] not in f["body"]:
        txt += f" on {f['host']}"
    if f["has_time"] and "{time}" not in f["body_tpl"]:
        txt += f" (started {f['time']})"
    if f["site"]:
        txt += f" -- this is at {f['site']}"
    if f["ticket"]:
        txt += f", ticket {f['ticket']}"
    if f["clients"] is not None:
        txt += f", roughly {f['clients']} users complaining"
    txt += f". {f['action_phrase']}. {f['sev_phrase'].lower()}"
    txt += f" ({f['vendor_cue']})"
    if rng.random() < 0.35:
        txt += " " + rng.choice(CHATTER)
    return _typo(txt, rng) if rng.random() < 0.5 else txt


def _style_ticket(f: dict, rng: random.Random) -> str:
    rows = [
        f"Ticket ID   : {f['ticket'] or 'not yet assigned'}",
        f"Priority    : {f['sev_phrase']}",
        f"CI          : {f['host']} ({f['vendor_cue']})",
        f"Location    : {f['site'] or 'n/a'}",
        f"Opened      : {f['time']}",
        f"Description : {f['body']}.",
        f"Impact      : {f['clients'] if f['clients'] is not None else 'not quantified'} users",
        f"Next step   : {f['action_phrase']}",
    ]
    rng.shuffle(rows) if rng.random() < 0.15 else None
    return "\n".join(rows)


def _style_cli(f: dict, rng: random.Random) -> str:
    """Held out from training in the OOD split."""
    return (
        f"<{f['host']}> display logbuffer | include ERROR\n"
        f"{f['time']} {f['body']}\n"
        f"<{f['host']}> display version\n"
        f"{f['vendor_cue']}, uptime 214 days\n"
        f"# operator note: {f['sev_phrase']}, {f['action_phrase']}"
        + (f", ref {f['ticket']}" if f["ticket"] else "")
    )


def _style_handoff(f: dict, rng: random.Random) -> str:
    """Held out from training in the OOD split. Deliberately the messiest."""
    txt = (
        f"shift handover note >> {f['body']} .. started around {f['time']} .. "
        f"box is {f['host']} {f['vendor_cue']} "
        + (f"at {f['site']} " if f["site"] else "")
        + (f"tkt {f['ticket']} " if f["ticket"] else "no tkt raised yet ")
        + (f"~{f['clients']} usrs " if f["clients"] is not None else "")
        + f".. {f['action_phrase']} .. flagged {f['sev_phrase']}"
    )
    return _typo(txt, rng)


STYLES = {
    "email": _style_email,
    "syslog": _style_syslog,
    "chat": _style_chat,
    "ticket": _style_ticket,
    "cli": _style_cli,
    "handoff": _style_handoff,
}
IN_DIST_STYLES = ["email", "syslog", "chat", "ticket"]
OOD_STYLES = ["cli", "handoff"]

EVENT_HUMAN = {k: k.replace("_", " ") for k in EVENTS}

# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def _iface(vendor: str, rng: random.Random) -> tuple[str, str]:
    """Return (surface form, canonical form)."""
    tpl = rng.choice(IFACE_DIALECT[vendor])
    a, b, c = rng.randint(1, 2), rng.randint(0, 1), rng.randint(1, 48)
    surface = tpl.format(a=a, b=b, c=c)
    canon = normalize_interface(surface.replace("interface ", ""))
    return surface, canon


def _sample_incident(rng: random.Random, split: str) -> dict[str, Any]:
    vendor = rng.choice(list(VENDOR_CUES))
    site_pool = SITES["test"] if split in ("test", "ood") else SITES["train"]
    site = rng.choice(site_pool) if rng.random() < 0.65 else None
    host = f"{rng.choice(ROLES)}-{site.split('-')[0] if site else rng.choice(['hq', 'lab', 'dmz', 'wan'])}-{rng.randint(1, 40):02d}"
    event = rng.choice(list(EVENTS))
    tpl = rng.choice(EVENTS[event])

    iface_s, iface_c = _iface(vendor, rng)
    iface2_s, iface2_c = _iface(vendor, rng)
    interfaces: list[str] = []
    if event in NEEDS_IFACE:
        interfaces = [iface_c]
        if "{iface2}" in tpl["body"]:
            interfaces.append(iface2_c)

    tz_name, tz_off = rng.choice(TZ_OFFSETS)
    base = dt.datetime(2025, rng.randint(1, 12), rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59))
    utc = base - dt.timedelta(hours=tz_off)
    has_time = rng.random() < 0.72
    tz_iso = f"{'+' if tz_off >= 0 else '-'}{int(abs(tz_off)):02d}:{int(round((abs(tz_off) % 1) * 60)):02d}"
    time_str = (
        f"{base:%Y-%m-%d %H:%M:%S} {tz_name}"
        if has_time
        else rng.choice(["an unknown time", "some point this morning", "earlier today"])
    )

    severity = rng.choice(list(SEV_PHRASES)) if rng.random() < 0.25 else DEFAULT_SEVERITY[event]
    action = rng.choice(list(ACTION_PHRASES)) if rng.random() < 0.25 else DEFAULT_ACTION[event]
    ticket = f"INC{rng.randint(1000000, 9999999)}" if rng.random() < 0.6 else None
    clients = rng.choice([12, 45, 90, 180, 400, 1200]) if rng.random() < 0.45 else None
    n = rng.randint(1, 250)

    subs = {
        "host": host, "iface": iface_s, "iface2": iface2_s,
        "time": time_str, "n": n, "site": site or "",
    }
    body = tpl["body"].format(**subs)
    evidence = tpl["ev"].format(**subs)

    facts = {
        "host": host, "site": site, "vendor": vendor,
        "vendor_cue": rng.choice(VENDOR_CUES[vendor]),
        "body": body, "event_human": EVENT_HUMAN[event],
        "ticket": ticket, "clients": clients, "time": time_str,
        "local_dt": base, "tz_iso": tz_iso, "has_time": has_time,
        "body_tpl": tpl["body"],
        "sev_phrase": rng.choice(SEV_PHRASES[severity]),
        "action_phrase": rng.choice(ACTION_PHRASES[action]),
    }

    style_pool = OOD_STYLES if split == "ood" else IN_DIST_STYLES
    style = rng.choice(style_pool)
    text = STYLES[style](facts, rng)

    # --- faithfulness guards -------------------------------------------------
    # A field may only be labelled if a reader could actually recover it from
    # the rendered text. Every guard below removed a distinct class of
    # hallucination from the trained model (docs/TROUBLESHOOTING.md, issue #2).
    low = text.lower()
    if host.lower() not in low:
        host = None
    if site and site.lower() not in low:
        site = None
    if ticket and ticket.lower() not in low:
        ticket = None
    if clients is not None and str(clients) not in text:
        clients = None
    if has_time and f"{base:%H:%M}" not in text:
        has_time = False

    record = {
        "status": "incident",
        "incident": {
            "device_hostname": host,
            "vendor": vendor,
            "site": site,
            "interfaces": sorted(set(interfaces)),
            "event_type": event,
            "severity": severity,
            "started_at": f"{utc:%Y-%m-%dT%H:%M:%S}Z" if has_time else None,
            "ticket_id": ticket,
            "affected_clients": clients,
            "action_required": action,
            "evidence_span": evidence,
        },
    }
    return {"text": text, "record": record, "style": style, "event_type": event, "label": "incident"}


def _sample_abstain(rng: random.Random, split: str) -> dict[str, Any]:
    group = "eval" if split in ("test", "ood") else "train"
    pool = [x for x in ABSTAIN_POOL if x[2] == group]
    reason, tpl, _ = rng.choice(pool)
    site_pool = SITES["test"] if split in ("test", "ood") else SITES["train"]
    text = tpl.format(
        site=rng.choice(site_pool),
        host=f"{rng.choice(ROLES)}-{rng.randint(1, 40):02d}",
        tid=rng.randint(1000000, 9999999),
        num=rng.randint(2, 9),
    )
    # Abstain examples come from a fixed template pool, so without extra surface
    # noise the unique-text dedup caps how many the generator can emit and the
    # abstain share silently collapses to ~8%. These variations restore it.
    prefixes = ["", "hi all, ", "quick one - ", "FYI: ", "[NOTICE] ", "@channel ", "Note: "]
    text = rng.choice(prefixes) + text
    if rng.random() < 0.35:
        text += " " + rng.choice(CHATTER)
    if rng.random() < 0.35:
        text += f" (ref #{rng.randint(100, 9999)})"
    if rng.random() < 0.3:
        text = rng.choice(FORWARD_HEADERS) + text
    if rng.random() < 0.3:
        text += rng.choice(SIGNATURES)
    return {
        "text": text,
        "record": {"status": "no_incident", "reason": reason},
        "style": "abstain",
        "event_type": "none",
        "label": "no_incident",
    }


def generate(n: int, seed: int = 0, split: str = "train", abstain_rate: float = 0.18) -> list[dict]:
    rng = random.Random(seed)
    rows, seen = [], set()
    guard = 0
    while len(rows) < n and guard < n * 40:
        guard += 1
        item = _sample_abstain(rng, split) if rng.random() < abstain_rate else _sample_incident(rng, split)
        key = item["text"].strip()
        if key in seen:
            continue
        seen.add(key)
        rec = item["record"]
        v = validate_record(rec)
        if not v.ok:  # pragma: no cover - generator bug guard
            raise AssertionError(f"generator emitted invalid record: {v.errors}\n{rec}")
        if rec["status"] == "incident":
            span = rec["incident"]["evidence_span"]
            if span not in item["text"]:
                # Surface styles lowercase or typo the body; recover the true span.
                lowered = span.lower()
                if lowered in item["text"].lower():
                    idx = item["text"].lower().index(lowered)
                    rec["incident"]["evidence_span"] = item["text"][idx : idx + len(span)]
                else:
                    continue  # typo mangled the span; drop rather than teach a bad grounding
        rows.append(
            {
                "id": f"{split}-{len(rows):06d}",
                "text": item["text"],
                "target": dumps(rec),
                "meta": {"style": item["style"], "event_type": item["event_type"], "label": item["label"]},
            }
        )
    if len(rows) < n:  # pragma: no cover
        raise RuntimeError(f"could only generate {len(rows)}/{n} unique rows; widen the templates")
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the SPINE corpus.")
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--train", type=int, default=6000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--test", type=int, default=800)
    ap.add_argument("--ood", type=int, default=300)
    ap.add_argument("--pool", type=int, default=1500,
                    help="unseen train-distribution prompts used to mine DPO pairs on-policy")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    specs = [
        ("train", args.train, args.seed),
        ("val", args.val, args.seed + 1),
        ("test", args.test, args.seed + 2),
        ("ood", args.ood, args.seed + 3),
        ("pool", args.pool, args.seed + 4),
    ]
    for split, count, seed in specs:
        if count <= 0:
            continue
        rows = generate(count, seed=seed, split=split)
        write_jsonl(rows, args.out / f"{split}.jsonl")
        n_abstain = sum(1 for r in rows if r["meta"]["label"] == "no_incident")
        print(f"{split:<6} {len(rows):>5} rows  ({n_abstain} abstain, {len(rows) - n_abstain} incident)  -> {args.out / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
