from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ALLOWED = {
    "SUPPORTED",
    "NOT_SUPPORTED",
}

VALID_RE = re.compile(
    r"^\s*(C-\d{3})\s*\|\s*(SUPPORTED|NOT_SUPPORTED)\s*$",
    re.I | re.M,
)

ATTEMPT_RE = re.compile(
    r"^\s*(C-\d{3})\s*\|\s*(\S[^\n]*)$",
    re.I | re.M,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    task = json.loads(
        Path(args.task).read_text(encoding="utf-8")
    )

    response = Path(args.response).read_text(
        encoding="utf-8"
    )

    gold = {
        item["claim_id"]: item["gold_verdict"]
        for item in task["items"]
    }

    expected = set(gold)

    attempted = {
        cid.upper()
        for cid, _ in ATTEMPT_RE.findall(response)
        if cid.upper() in expected
    }

    predictions = {}

    for cid, verdict in VALID_RE.findall(response):
        cid = cid.upper()
        verdict = verdict.upper()

        if cid in expected and cid not in predictions:
            predictions[cid] = verdict

    valid_ids = set(predictions)

    correct = sum(
        predictions[cid] == gold[cid]
        for cid in valid_ids
    )

    missing = sorted(expected - attempted)

    invalid = sorted(
        attempted - valid_ids
    )

    n = len(expected)

    result = {
        "required_count": n,

        "attempted_count": len(attempted),

        "attempt_coverage": (
            len(attempted) / n
            if n else 0
        ),

        "missing_count": len(missing),

        "missing_ids": missing,

        "valid_count": len(valid_ids),

        "structural_completion_rate": (
            len(valid_ids) / n
            if n else 0
        ),

        "invalid_attempt_count": len(invalid),

        "invalid_attempt_ids": invalid,

        "correct_verdict_count": correct,

        "verdict_accuracy_among_valid": (
            correct / len(valid_ids)
            if valid_ids
            else 0
        ),

        "full_task_accuracy": (
            correct / n
            if n else 0
        ),
    }

    Path(args.output).write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
