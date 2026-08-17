from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean


ALLOWED = {
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT",
}


LINE_RE = re.compile(
    r"^\s*(C-\d{3})\s*\|\s*"
    r"([^|]+?)"
    r"(?:\s*\|\s*(.*))?$",
    re.I | re.M,
)


def evidence_f1(
    predicted,
    gold,
):
    predicted = set(predicted)
    gold = set(gold)

    if not predicted and not gold:
        return 1.0

    if not predicted or not gold:
        return 0.0

    tp = len(
        predicted.intersection(gold)
    )

    precision = tp / len(predicted)
    recall = tp / len(gold)

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True,
    )

    parser.add_argument(
        "--response",
        required=True,
    )

    parser.add_argument(
        "--technical-failure",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    task = json.loads(
        Path(args.task).read_text(
            encoding="utf-8"
        )
    )

    response = Path(
        args.response
    ).read_text(
        encoding="utf-8"
    )

    gold = {
        item["claim_id"]: item
        for item in task["items"]
    }

    expected = set(gold)

    attempted = set()
    valid = {}
    duplicates = set()

    seen_counts = {}

    for match in LINE_RE.finditer(
        response
    ):
        claim_id = (
            match.group(1).upper()
        )

        if claim_id not in expected:
            continue

        attempted.add(claim_id)

        seen_counts[claim_id] = (
            seen_counts.get(
                claim_id,
                0,
            )
            + 1
        )

        if seen_counts[claim_id] > 1:
            duplicates.add(claim_id)

        verdict = (
            match.group(2)
            .strip()
            .upper()
        )

        citation_text = (
            match.group(3) or ""
        ).strip()

        if verdict not in ALLOWED:
            continue

        if claim_id in valid:
            continue

        citations = {
            x.strip()
            for x in citation_text.split(",")
            if x.strip()
            and x.strip().upper()
            != "NONE"
        }

        valid[claim_id] = {
            "verdict": verdict,
            "citations": citations,
        }

    missing = sorted(
        expected - attempted
    )

    invalid_attempts = sorted(
        attempted - set(valid)
    )

    correct = 0
    evidence_scores = []

    for claim_id, pred in valid.items():
        item = gold[claim_id]

        if (
            pred["verdict"]
            == item["gold_verdict"]
        ):
            correct += 1

        evidence_scores.append(
            evidence_f1(
                pred["citations"],
                item[
                    "gold_evidence_ids"
                ],
            )
        )

    n = len(expected)

    result = {
        "required_count": n,

        "attempted_count": (
            len(attempted)
        ),

        "attempt_coverage": (
            len(attempted) / n
            if n
            else 0
        ),

        "missing_count": (
            len(missing)
        ),

        "missing_ids": missing,

        "valid_count": (
            len(valid)
        ),

        "structural_completion_rate": (
            len(valid) / n
            if n
            else 0
        ),

        "invalid_attempt_count": (
            len(invalid_attempts)
        ),

        "invalid_attempt_ids": (
            invalid_attempts
        ),

        "duplicate_ids": sorted(
            duplicates
        ),

        "correct_verdict_count": (
            correct
        ),

        "verdict_accuracy_among_valid": (
            correct / len(valid)
            if valid
            else 0
        ),

        "mean_evidence_f1": (
            mean(evidence_scores)
            if evidence_scores
            else 0
        ),

        "technical_failure": (
            args.technical_failure
        ),
    }

    Path(args.output).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
