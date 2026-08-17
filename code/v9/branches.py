"""V9 execution branches: NORMAL, two PLACEBO, two CAREFUL.

The point of this module is the V3.2 lesson. There, a held-out "careful"
elicitor beat baseline at p=0.035 on the qualified subset - and the same
contrast computed ungated over the full population reversed sign (p=0.096).
The neutral placebo prompt itself improved performance. A carefulness effect
measured against NORMAL alone is therefore uninterpretable: any added text
changes behaviour.

So the utilization gap is defined against PLACEBO, not against NORMAL:

    CAREFUL > NORMAL     necessary, not sufficient
    CAREFUL > PLACEBO    the claim that actually survives review

Two of each wording exist so a single phrasing cannot carry the result. The
placebos are length-matched to the careful prompts (within a tight token band)
and contain no instruction to verify, check, re-read, or be thorough - they add
context of comparable size and no methodological content.

Every branch is generated independently from the identical frozen task state.
No branch ever sees another branch's output.
"""

from __future__ import annotations

from v9.task import SPEC, render_ledger

BASE = ("Below is a shipment ledger of {n} records.\n\n{body}\n\n" + SPEC)

# Neutral framing, no verification language. This is the reference condition.
NORMAL_SUFFIX = ""

# Length-matched, methodologically empty. Adds background, not diligence.
PLACEBO_SUFFIXES = {
    "PLACEBO_A": (
        "\n\nThis ledger was exported from the warehouse management system on the "
        "first business day of the quarter. Records are stored in the order they "
        "were scanned at the receiving dock. The zone codes follow the site's "
        "standard grid labelling scheme, which has been in use since the facility "
        "opened."
    ),
    "PLACEBO_B": (
        "\n\nShipment ledgers of this kind are produced routinely by the inventory "
        "team and archived for the standard retention period. The item names are "
        "drawn from the site's parts catalogue. Unit weights are recorded in "
        "kilograms, consistent with the rest of the facility's documentation."
    ),
}

# Verification language, no hint about which requirement or which record matters.
CAREFUL_SUFFIXES = {
    "CAREFUL_A": (
        "\n\nBefore you finalise your answer, carefully verify every requirement "
        "stated above for every record. Do not rely on assumption or pattern - "
        "actually perform each comparison and each calculation. Check your work "
        "before you finish."
    ),
    "CAREFUL_B": (
        "\n\nWork through this thoroughly. For each record, deliberately carry out "
        "the comparison and the arithmetic the instructions require rather than "
        "estimating. Review what you have written for correctness before "
        "presenting it as complete."
    ),
}

CONDITIONS = ("NORMAL",) + tuple(PLACEBO_SUFFIXES) + tuple(CAREFUL_SUFFIXES)
CAREFUL_CONDITIONS = tuple(CAREFUL_SUFFIXES)
PLACEBO_CONDITIONS = tuple(PLACEBO_SUFFIXES)


def build_prompt(ledger: dict, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    suffix = (NORMAL_SUFFIX if condition == "NORMAL"
              else PLACEBO_SUFFIXES.get(condition) or CAREFUL_SUFFIXES[condition])
    return BASE.format(n=ledger["n_records"], body=render_ledger(ledger)) + suffix


def suffix_lengths() -> dict[str, int]:
    """Character lengths, so length-matching of placebo vs careful is auditable."""
    lengths = {"NORMAL": 0}
    for name, text in {**PLACEBO_SUFFIXES, **CAREFUL_SUFFIXES}.items():
        lengths[name] = len(text)
    return lengths
