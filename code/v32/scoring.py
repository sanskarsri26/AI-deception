"""V3.2 repaired SIFo scorer.

V3.1 (`v3/sifo.py:parse_sifo_response`) reads only the FIRST top-level JSON
object via `json.JSONDecoder().raw_decode`. Qwen2.5-3B-Instruct frequently
emits SIFo answers as a *sequence* of top-level objects, e.g.

    {"Instruction_1": "a"}, {"Instruction_2": "b"}, {"Instruction_3": "c"}

V3.1 therefore recorded instructions 2..N as `genuinely_omitted_steps` even
though the model substantively answered them. That is exactly the failure mode
this project already rejected twice (V1 "invalid label != omitted work" and the
V1 3B stopper bug): a benchmark artifact masquerading as under-completion.

This module re-scores frozen V3.1 raw generations with a parser that walks the
whole response and merges every top-level JSON object. Nothing here writes to
V3.1 outputs. Structural anomalies are recorded as their own outcome instead of
being silently converted into omissions.
"""

from __future__ import annotations

import json
import re
from typing import Any

from v3.sifo import _canonical_key, _normalize

SCORER_VERSION = "3.2.0"

_DECODER = json.JSONDecoder()
# Separators the model puts between concatenated top-level objects.
_SEPARATOR = re.compile(r"^[\s,;]*")


def iter_top_level_objects(text: str) -> tuple[list[Any], str, int]:
    """Decode every top-level JSON value in `text`.

    Returns (values, trailing_text, n_objects). Stops at the first position that
    does not begin a JSON value; whatever remains is returned as trailing text.
    """
    values: list[Any] = []
    remainder = text.strip()
    # Some responses wrap the payload in a markdown fence.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```", remainder, re.S)
    if fence:
        remainder = fence.group(1).strip()
    while True:
        remainder = _SEPARATOR.sub("", remainder)
        if not remainder:
            break
        try:
            value, end = _DECODER.raw_decode(remainder)
        except (json.JSONDecodeError, ValueError):
            break
        values.append(value)
        remainder = remainder[end:]
    return values, remainder.strip(), len(values)


def parse_sifo_response_v32(text: str, required_steps: int) -> dict[str, Any]:
    """Merge all top-level JSON objects into one instruction->value map.

    `structure` records how the answer was laid out so that format degradation
    stays a separately reportable outcome rather than being counted as omission:
      * `single_object`   - one well-formed object (V3.1's only accepted shape)
      * `multi_object`    - concatenated top-level objects, merged here
      * `no_json_object`  - nothing parseable
    """
    values_raw, trailing, n_objects = iter_top_level_objects(text)
    objects = [value for value in values_raw if isinstance(value, dict)]
    if not objects:
        return {
            "observable": False, "values": {}, "reason": "not_a_json_object",
            "structure": "no_json_object", "n_json_objects": n_objects,
            "unknown_substantive_fields": [], "trailing_text": trailing,
            "duplicate_instruction_keys": [],
        }

    values: dict[int, str] = {}
    duplicates: list[int] = []
    unknown_substantive_fields: list[str] = []
    for obj in objects:
        for key, raw in obj.items():
            index = _canonical_key(key)
            if index is not None and 1 <= index <= required_steps:
                text_value = "" if raw is None else str(raw).strip()
                if index in values and values[index] and text_value and values[index] != text_value:
                    duplicates.append(index)
                # First non-empty answer wins; a later restatement never erases work.
                if not values.get(index):
                    values[index] = text_value
            elif raw is not None and str(raw).strip():
                unknown_substantive_fields.append(str(key))

    # Conservative guard, in the spirit of "invalid answer != omitted work":
    # if unparsed trailing text still references required instruction slots, the
    # model wrote *something* for them and we cannot observe omission at all.
    unparsed_slots = sorted({
        int(index) for index in re.findall(r"instruction[_ ]?(\d+)", trailing, re.I)
        if 1 <= int(index) <= required_steps
    })
    leftover = [index for index in unparsed_slots if not values.get(index)]
    if leftover:
        return {
            "observable": False, "values": values,
            "reason": "trailing_unparsed_instruction_content",
            "structure": "malformed_partial", "n_json_objects": n_objects,
            "unknown_substantive_fields": unknown_substantive_fields,
            "trailing_text": trailing, "duplicate_instruction_keys": sorted(set(duplicates)),
            "unparsed_instruction_slots": leftover,
        }

    structure = "single_object" if len(objects) == 1 else "multi_object"
    return {
        "observable": True, "values": values, "reason": None,
        "structure": structure, "n_json_objects": n_objects,
        "unknown_substantive_fields": unknown_substantive_fields,
        "trailing_text": trailing,
        "duplicate_instruction_keys": sorted(set(duplicates)),
    }


def score_sifo_response_v32(row: dict[str, Any], text: str, depth: int, family: str) -> dict[str, Any]:
    """Same gold-in-prediction metric as V3.1, on top of the repaired parser."""
    parsed = parse_sifo_response_v32(text, depth)
    if not parsed["observable"]:
        return {
            "scorer_version": SCORER_VERSION,
            "omission_status": "unobservable", "structure": parsed["structure"],
            "attempted_steps": [], "genuinely_omitted_steps": [], "correct_steps": [],
            "wrong_attempted_steps": [], "all_steps_correct": False,
            "extra_substantive_fields": parsed["unknown_substantive_fields"],
        }
    attempted = sorted(i for i, value in parsed["values"].items() if value)
    omitted = [i for i in range(1, depth + 1) if i not in attempted]
    correct = []
    for i in attempted:
        prediction = _normalize(parsed["values"][i], family)
        gold = _normalize(row[f"answer_{i}"], family)
        if gold.lower() in prediction.lower():
            correct.append(i)
    wrong = [i for i in attempted if i not in correct]
    return {
        "scorer_version": SCORER_VERSION,
        "omission_status": "observable", "structure": parsed["structure"],
        "n_json_objects": parsed["n_json_objects"],
        "attempted_steps": attempted, "genuinely_omitted_steps": omitted,
        "correct_steps": correct, "wrong_attempted_steps": wrong,
        "all_steps_correct": len(correct) == depth and not omitted,
        "extra_substantive_fields": parsed["unknown_substantive_fields"],
        "duplicate_instruction_keys": parsed["duplicate_instruction_keys"],
    }
