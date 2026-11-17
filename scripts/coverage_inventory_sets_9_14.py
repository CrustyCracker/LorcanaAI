#!/usr/bin/env python3
"""Inventory Sets 9–14 effect coverage: empty clauses, text-hack statics, mechanic buckets."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lorcana_ai.data.card_db import CardDatabase  # noqa: E402
from lorcana_ai.data.effect_dsl import KeywordFlag, TriggerType  # noqa: E402

# Patterns that currently live as full_text.lower() hacks in game_state / simulator.
TEXT_HACK_PATTERNS = [
    ("under_card_static", r"while there's a card under this character"),
    ("hand_empty_resist", r"while you have no cards in your hand.*resist"),
    ("detective_lead_aura", r"your detective characters gain alert and resist"),
    ("villain_in_play_static", r"while you have a villain character in play"),
    ("discard_10_location", r"while you have 10 or more cards in your discard"),
    ("support_lore_aura", r"your characters with support get \+1"),
    ("lore_behind_static", r"while an opponent has more lore than you"),
    ("karnage_debuff", r"each opposing damaged character gets -1"),
    ("amber_champ_aura", r"your other amber characters get \+2"),
    ("amber_champ_singer", r"gains singer 8"),
    ("cant_challenge_static", r"this character can't challenge"),
]

BUCKET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aura_ally_buff", re.compile(r"your (?:other )?(?:\w+\s+)?characters? get \+", re.I)),
    ("aura_gain_keyword", re.compile(r"your (?:other )?(?:\w+\s+)?characters? gain", re.I)),
    ("self_static_while", re.compile(r"while .{5,80}this (?:character|location) gets? \+", re.I)),
    ("support_keyword", re.compile(r"\bSupport\b")),
    ("ward_keyword", re.compile(r"\bWard\b")),
    ("activated", re.compile(r"⟳|exert\.|, \d+\s*⬡")),
    ("on_quest", re.compile(r"whenever (?:this character|she|he) quests", re.I)),
    ("on_play", re.compile(r"when you play", re.I)),
    ("on_banish_leave", re.compile(r"when this character is banished|leaves play|would be banished", re.I)),
    ("choice_you_may", re.compile(r"\byou may\b", re.I)),
    ("choice_chosen", re.compile(r"\bchosen\b", re.I)),
    ("reminder_only", re.compile(r"^\s*\([^)]+\)\s*$")),
    ("ready_allies", re.compile(r"ready (?:chosen|all)", re.I)),
    ("heal", re.compile(r"remove .+ damage", re.I)),
    ("look_at_top", re.compile(r"look at the top", re.I)),
    ("classification_gate", re.compile(r"princess|villain|detective|floodborn|madrigal", re.I)),
]


def strip_reminder(text: str) -> str:
    return " ".join(re.sub(r"\([^)]*\)", " ", text or "").split())


def is_reminder_only(full_text: str) -> bool:
    raw = (full_text or "").strip()
    if not raw:
        return True
    no_rem = strip_reminder(raw)
    # Keyword-only with reminder paren bodies
    if not no_rem:
        return True
    # Pure keyword lines like "Rush (This character can challenge ready characters.)"
    # Also bare keywords without reminder text: "Support", "Singer 7", "Challenger +3"
    kw_only = re.sub(
        r"\b(Rush|Ward|Evasive|Bodyguard|Support|Vanish|Reckless|Alert|Adventurous|"
        r"Resist\s*\+\d+|Challenger\s*\+\d+|Singer\s*\d+|Sing Together\s*\d+|"
        r"Shift\s*\d+|Boost\s*\d+|Universal Shift\s*\d+|Duo Shift\s*\d+|"
        r"Combo Shift\s*\d+|Temporary(?:\s+\w+)?\s+Shift\s*\d+|Potato Shift\s*\d+|"
        r"(?:Madrigal|Floodborn|Red Panda|Puppy|Pirate|Hero|Villain|Princess|Prince|"
        r"Knight|Alien|Robot|Whisper|Inventor|Detective|Dragon|Fairy|Hyena|Broom|"
        r"Musketeer|Racer|Hunny)\s+Shift\s*\d+)\b",
        "",
        no_rem,
        flags=re.I,
    )
    kw_only = re.sub(r"[,.\s⬡]+", "", kw_only)
    return kw_only == ""


def classify_card(card) -> dict:
    text = card.full_text or ""
    lower = text.lower()
    no_rem = strip_reminder(text).lower()
    clauses = list(card.abilities.clauses)
    has_clauses = len(clauses) > 0
    has_aura = bool(getattr(card.abilities, "aura_statics", None))
    has_grants = bool(getattr(card.abilities, "conditional_grants", None))
    has_compiled = has_clauses or has_aura or has_grants
    reminder = is_reminder_only(text)
    long_text = len(text) > 50
    empty = not has_compiled
    empty_meaningful = empty and not reminder and bool(no_rem)

    buckets = []
    for name, pat in BUCKET_PATTERNS:
        if pat.search(text) or pat.search(no_rem):
            buckets.append(name)
    if not buckets and reminder:
        buckets.append("reminder_only")
    if not buckets and empty_meaningful:
        buckets.append("unclassified_empty")
    if has_clauses:
        buckets.append("has_dsl_clause")
    if has_aura:
        buckets.append("has_aura_static")
    if has_grants:
        buckets.append("has_conditional_grant")
    if has_compiled:
        buckets.append("has_compiled_ir")

    text_hacks = []
    for name, pat in TEXT_HACK_PATTERNS:
        if re.search(pat, lower):
            text_hacks.append(name)

    triggers = Counter(c.trigger.value for c in clauses)
    ops = Counter(c.op.value for c in clauses)
    kws = [k.name for k in KeywordFlag if k and (card.abilities.keywords & k)]

    coverage = "compiled"
    if reminder and not has_compiled:
        coverage = "reminder_only"
    elif empty_meaningful:
        coverage = "empty_meaningful"
    elif empty:
        coverage = "empty_short"
    elif text_hacks and has_compiled:
        coverage = "compiled_plus_text_hack"

    return {
        "card_id": card.card_id,
        "full_name": card.full_name,
        "set_code": card.set_code,
        "card_type": card.card_type,
        "text_len": len(text),
        "clause_count": len(clauses),
        "aura_count": len(getattr(card.abilities, "aura_statics", []) or []),
        "grant_count": len(getattr(card.abilities, "conditional_grants", []) or []),
        "triggers": dict(triggers),
        "ops": dict(ops),
        "keywords": kws,
        "buckets": buckets,
        "text_hacks": text_hacks,
        "coverage": coverage,
        "reminder_only": reminder,
        "empty_meaningful": empty_meaningful,
        "has_static_aura_flag": has_aura,
        "has_choice_flag": any(
            getattr(c, "optional", False) or getattr(c, "choice_kind", None) not in (None, "NONE")
            for c in clauses
        ),
        "unsupported_notes": list(getattr(card.abilities, "unsupported_notes", []) or []),
    }


def main() -> None:
    db = CardDatabase()
    cards = list(db.by_id.values())
    rows = [classify_card(c) for c in cards]

    coverage_counts = Counter(r["coverage"] for r in rows)
    bucket_counts = Counter()
    for r in rows:
        for b in r["buckets"]:
            bucket_counts[b] += 1
    text_hack_counts = Counter()
    for r in rows:
        for h in r["text_hacks"]:
            text_hack_counts[h] += 1
    empty_by_set = Counter(r["set_code"] for r in rows if r["empty_meaningful"])
    empty_by_type = Counter(r["card_type"] for r in rows if r["empty_meaningful"])
    trigger_counts = Counter()
    op_counts = Counter()
    for r in rows:
        for t, n in r["triggers"].items():
            trigger_counts[t] += n
        for o, n in r["ops"].items():
            op_counts[o] += n

    total = len(rows)
    with_clauses = sum(1 for r in rows if r["clause_count"] > 0)
    with_compiled = sum(1 for r in rows if r["clause_count"] > 0 or r.get("aura_count", 0) > 0 or r.get("grant_count", 0) > 0)
    empty_meaningful = sum(1 for r in rows if r["empty_meaningful"])
    reminder = sum(
        1
        for r in rows
        if r["reminder_only"] and r["clause_count"] == 0 and r.get("aura_count", 0) == 0 and r.get("grant_count", 0) == 0
    )
    text_hack_cards = sum(1 for r in rows if r["text_hacks"])

    summary = {
        "total_cards": total,
        "with_clauses": with_clauses,
        "clause_pct": round(100.0 * with_clauses / max(1, total), 2),
        "with_compiled_ir": with_compiled,
        "compiled_ir_pct": round(100.0 * with_compiled / max(1, total), 2),
        "empty_meaningful": empty_meaningful,
        "empty_meaningful_pct": round(100.0 * empty_meaningful / max(1, total), 2),
        "reminder_only_no_clause": reminder,
        "text_hack_cards": text_hack_cards,
        "coverage_counts": dict(coverage_counts),
        "bucket_counts": dict(bucket_counts.most_common()),
        "text_hack_counts": dict(text_hack_counts.most_common()),
        "empty_by_set": dict(sorted(empty_by_set.items())),
        "empty_by_type": dict(empty_by_type),
        "trigger_counts": dict(trigger_counts.most_common()),
        "op_counts": dict(op_counts.most_common()),
    }

    out_dir = Path(
        r"C:\Users\hrzon\AppData\Local\Cursor\AgentStores\cursor_agent_stores"
        r"\bc-c5410d23-3396-4c76-a143-0aa341bdf619\files\internal"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sets-9-14-coverage.json"
    payload = {"summary": summary, "cards": rows}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Also dump a slim repo-local copy for CI/scripts
    local = ROOT / "results" / "sets_9_14_coverage.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {json_path}")
    print(f"Wrote {local}")


if __name__ == "__main__":
    main()
