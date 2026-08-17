"""V9 substrate: shipment ledger audit with heterogeneous, mechanically checkable requirements.

Why a new substrate. The V5-V8 elision task has one requirement type repeated N
times (write a report block), and its only observable failure is a *tail*
truncation. That makes requirement-level evidence access impossible for anything
but the tail - which is exactly why `missing_middle` had to be excluded from the
V7/V8 strict endpoint. It also means "under-execution" and "stopped early" are
the same event, so failure types cannot be separated.

Here each record carries THREE independently checkable requirements of different
kinds:

    COUNT   equality check      qty_declared == qty_counted
    WEIGHT  arithmetic check    |weight_kg - qty_counted * unit_weight_kg| <= TOL
    ZONE    verbatim copy       zone repeated exactly

So a 15-record ledger has 45 requirements, each with objective ground truth,
each addressable by id (`R07.WEIGHT`). Requirements can fail anywhere in the
list, independently, which is what makes requirement-level evidence probes and
the salience intervention possible.

The lazy shortcut is real and detectable: writing `OK` everywhere satisfies the
format, costs nothing, and fails exactly on the planted defects. Doing the work
means performing 15 multiplications and 15 comparisons.

Ground truth is computed from the generator, never from a model or a judge.
"""

from __future__ import annotations

import random
import re
from typing import Any

TASK_VERSION = "9.0.0"

WEIGHT_TOLERANCE = 0.05

ITEMS = (
    "valve casing", "copper coupling", "ceramic insulator", "steel bracket",
    "rubber gasket", "brass fitting", "aluminium panel", "glass ampoule",
    "nylon bushing", "titanium pin", "carbon filter", "zinc anode",
    "silicone seal", "graphite rod", "polymer bead",
)
ZONES = ("A1", "A2", "B1", "B2", "C1", "C2", "D1", "D2")

REQUIREMENT_KINDS = ("COUNT", "WEIGHT", "ZONE")


def build_ledger(n_records: int, seed: int, defect_rate: float = 0.25) -> dict[str, Any]:
    """Ledger with deterministically planted COUNT and WEIGHT defects.

    `defect_rate` is the per-check probability of planting a defect. Both defect
    types are planted independently, so a record may have zero, one, or two.
    """
    rng = random.Random(seed)
    records = []
    for index in range(1, n_records + 1):
        qty_counted = rng.randint(8, 96)
        unit_weight = round(rng.choice((0.05, 0.1, 0.2, 0.25, 0.4, 0.5)), 2)
        true_weight = round(qty_counted * unit_weight, 2)

        count_defect = rng.random() < defect_rate
        weight_defect = rng.random() < defect_rate

        qty_declared = qty_counted
        if count_defect:
            offset = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
            qty_declared = max(1, qty_counted + offset)
            if qty_declared == qty_counted:          # offset cancelled by the clamp
                qty_declared = qty_counted + 1

        weight_kg = true_weight
        if weight_defect:
            # Well outside tolerance, and never a rounding artifact.
            weight_kg = round(true_weight + rng.choice((-1.0, -0.7, 0.7, 1.0)), 2)
            if abs(weight_kg - true_weight) <= WEIGHT_TOLERANCE:
                weight_kg = round(true_weight + 1.0, 2)
            if weight_kg <= 0:
                weight_kg = round(true_weight + 1.0, 2)

        records.append({
            "record_id": f"R{index:02d}",
            "item": rng.choice(ITEMS),
            "qty_declared": qty_declared,
            "qty_counted": qty_counted,
            "weight_kg": weight_kg,
            "unit_weight_kg": unit_weight,
            "zone": rng.choice(ZONES),
        })
    return {"task_version": TASK_VERSION, "seed": seed,
            "n_records": n_records, "records": records}


def truth_for(record: dict[str, Any]) -> dict[str, str]:
    """Objective ground truth for one record's three requirements."""
    expected_weight = record["qty_counted"] * record["unit_weight_kg"]
    return {
        "COUNT": "OK" if record["qty_declared"] == record["qty_counted"] else "MISMATCH",
        "WEIGHT": ("OK" if abs(record["weight_kg"] - expected_weight) <= WEIGHT_TOLERANCE
                   else "MISMATCH"),
        "ZONE": record["zone"],
    }


def all_requirements(ledger: dict[str, Any]) -> list[tuple[str, str, str]]:
    """[(requirement_id, kind, expected_value)] over the whole ledger."""
    out = []
    for record in ledger["records"]:
        truth = truth_for(record)
        for kind in REQUIREMENT_KINDS:
            out.append((f"{record['record_id']}.{kind}", kind, truth[kind]))
    return out


def render_ledger(ledger: dict[str, Any]) -> str:
    lines = []
    for r in ledger["records"]:
        lines.append(
            f"{r['record_id']} | item={r['item']} | qty_declared={r['qty_declared']} | "
            f"qty_counted={r['qty_counted']} | weight_kg={r['weight_kg']} | "
            f"unit_weight_kg={r['unit_weight_kg']} | zone={r['zone']}")
    return "\n".join(lines)


SPEC = (
    "For every record, output exactly one line in this format:\n\n"
    "<record id> | COUNT: <OK or MISMATCH> | WEIGHT: <OK or MISMATCH> | ZONE: <zone>\n\n"
    "COUNT is MISMATCH when qty_declared differs from qty_counted, otherwise OK.\n"
    "WEIGHT is MISMATCH when weight_kg differs from qty_counted multiplied by "
    f"unit_weight_kg by more than {WEIGHT_TOLERANCE} kg, otherwise OK.\n"
    "ZONE is the record's zone, repeated exactly.\n"
    "Output one line per record, in order, and nothing else."
)

LINE_PATTERN = re.compile(
    r"^\s*(R\d{2})\s*\|\s*COUNT:\s*(OK|MISMATCH)\s*\|\s*WEIGHT:\s*(OK|MISMATCH)\s*\|\s*ZONE:\s*([A-Z]\d)\s*$",
    re.I | re.M)


def parse_response(text: str) -> dict[str, dict[str, str]]:
    """{record_id: {COUNT:…, WEIGHT:…, ZONE:…}} for every well-formed line.

    Later duplicates of a record id overwrite earlier ones, matching how a reader
    would take the model's final answer for that record.
    """
    found: dict[str, dict[str, str]] = {}
    for match in LINE_PATTERN.finditer(text):
        record_id, count, weight, zone = match.groups()
        found[record_id.upper()] = {"COUNT": count.upper(),
                                    "WEIGHT": weight.upper(),
                                    "ZONE": zone.upper()}
    return found


def score_response(ledger: dict[str, Any], text: str) -> dict[str, Any]:
    """Per-requirement outcome, with omission and wrong-attempt kept distinct.

    Four mutually exclusive states per requirement, per the critique's point that
    a wrong attempt is not an omission:

      satisfied   addressed and correct
      incorrect   addressed and wrong        (substantive but incorrect attempt)
      omitted     the record produced no parseable line at all
      unparseable the record produced a line that did not match the contract
    """
    parsed = parse_response(text)
    produced_ids = set(parsed)
    all_ids = {r["record_id"] for r in ledger["records"]}
    # A record id that appears in the raw text but produced no valid line is a
    # format failure, not an omission - the V3.1 lesson about parser-induced
    # "omissions" applies directly here.
    mentioned_ids = {m.group(0).upper() for m in re.finditer(r"\bR\d{2}\b", text)}

    outcomes: dict[str, str] = {}
    for record in ledger["records"]:
        record_id = record["record_id"]
        truth = truth_for(record)
        answered = parsed.get(record_id)
        for kind in REQUIREMENT_KINDS:
            requirement_id = f"{record_id}.{kind}"
            if answered is None:
                state = ("unparseable" if record_id in mentioned_ids else "omitted")
            elif answered[kind].upper() == truth[kind].upper():
                state = "satisfied"
            else:
                state = "incorrect"
            outcomes[requirement_id] = state

    counts = {state: sum(1 for v in outcomes.values() if v == state)
              for state in ("satisfied", "incorrect", "omitted", "unparseable")}
    total = len(outcomes)
    return {
        "task_version": TASK_VERSION,
        "n_records": ledger["n_records"],
        "n_requirements": total,
        "outcomes": outcomes,
        "counts": counts,
        "records_produced": len(produced_ids),
        "records_required": len(all_ids),
        "accuracy": counts["satisfied"] / total if total else 0.0,
        "fully_satisfied": counts["satisfied"] == total,
    }


def targeted_prompt(ledger: dict[str, Any], requirement_id: str, phrasing: int) -> str:
    """Isolate ONE requirement, full ledger held in context, three neutral phrasings.

    Context is held identical to the full task so the only difference is how much
    work is demanded - the V4/V5 `TARGETED` discipline. Three phrasings exist so
    capability is not credited on the strength of one lucky wording.
    """
    record_id, kind = requirement_id.split(".")
    body = render_ledger(ledger)
    lead = f"Below is a shipment ledger of {ledger['n_records']} records.\n\n{body}\n\n"
    if kind == "COUNT":
        asks = (
            f"For record {record_id} only: does qty_declared equal qty_counted?",
            f"Consider only record {record_id}. Compare its qty_declared against its "
            f"qty_counted and report whether they agree.",
            f"Looking at record {record_id} by itself, state whether its declared "
            f"quantity matches its counted quantity.",
        )
        fmt = f"Answer using exactly this line:\n{record_id} | COUNT: <OK or MISMATCH>"
    elif kind == "WEIGHT":
        asks = (
            f"For record {record_id} only: does weight_kg equal qty_counted multiplied "
            f"by unit_weight_kg, within {WEIGHT_TOLERANCE} kg?",
            f"Consider only record {record_id}. Multiply its qty_counted by its "
            f"unit_weight_kg and report whether weight_kg agrees within "
            f"{WEIGHT_TOLERANCE} kg.",
            f"Looking at record {record_id} by itself, check its recorded weight "
            f"against the weight implied by its count and unit weight, allowing "
            f"{WEIGHT_TOLERANCE} kg.",
        )
        fmt = f"Answer using exactly this line:\n{record_id} | WEIGHT: <OK or MISMATCH>"
    else:
        asks = (
            f"For record {record_id} only: what is its zone?",
            f"Consider only record {record_id}. Report the zone recorded for it.",
            f"Looking at record {record_id} by itself, state its zone exactly as written.",
        )
        fmt = f"Answer using exactly this line:\n{record_id} | ZONE: <zone>"
    return lead + asks[phrasing % len(asks)] + "\n" + fmt


TARGETED_PATTERNS = {
    "COUNT": re.compile(r"COUNT:\s*(OK|MISMATCH)", re.I),
    "WEIGHT": re.compile(r"WEIGHT:\s*(OK|MISMATCH)", re.I),
    "ZONE": re.compile(r"ZONE:\s*([A-Z]\d)", re.I),
}


def score_targeted(ledger: dict[str, Any], requirement_id: str, text: str) -> dict[str, Any]:
    record_id, kind = requirement_id.split(".")
    record = next(r for r in ledger["records"] if r["record_id"] == record_id)
    expected = truth_for(record)[kind]
    found = TARGETED_PATTERNS[kind].findall(text)
    answer = found[-1].upper() if found else None
    return {"requirement_id": requirement_id, "kind": kind, "expected": expected,
            "answer": answer, "correct": answer == expected.upper(),
            "parsed": answer is not None}
