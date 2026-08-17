from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_item(item):
    lines = [
        f"{item['claim_id']}:",
        f"Claim: {item['claim']}",
        "Evidence:",
    ]

    for e in item["evidence"]:
        lines.append(
            f"{e['evidence_id']}: {e['text']}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    task = json.loads(
        Path(args.task).read_text(encoding="utf-8")
    )

    prompt = """Binary evidence verification.

There are exactly two possible verdicts:

SUPPORTED
NOT_SUPPORTED

SUPPORTED means every factual statement in the claim agrees with the evidence.

NOT_SUPPORTED means at least one factual statement in the claim disagrees with the evidence.

Use only those two verdicts.

Return exactly:

CLAIM_ID | VERDICT


Examples:

EX-001:
Claim: Project-A is in the North region.
Evidence:
E1: Project-A is in the North region.

Evaluation:
EX-001 | SUPPORTED


EX-002:
Claim: Project-B is in the South region.
Evidence:
E1: Project-B is in the East region.

Evaluation:
EX-002 | NOT_SUPPORTED


EX-003:
Claim: Project-C is owned by Avery.
Evidence:
E1: Project-C is owned by Avery.

Evaluation:
EX-003 | SUPPORTED


EX-004:
Claim: Project-D has critical priority.
Evidence:
E1: Project-D has low priority.

Evaluation:
EX-004 | NOT_SUPPORTED


EX-005:
Claim: Project-E uses platform Atlas.
Evidence:
E1: Project-E uses platform Atlas.

Evaluation:
EX-005 | SUPPORTED


EX-006:
Claim: Project-F has status testing.
Evidence:
E1: Project-F has status retired.

Evaluation:
EX-006 | NOT_SUPPORTED


EX-007:
Claim: Project-G is in the West region and is owned by Blake.
Evidence:
E1: Project-G is in the West region.
E2: Project-G is owned by Blake.

Evaluation:
EX-007 | SUPPORTED


EX-008:
Claim: Project-H is in the West region and is owned by Blake.
Evidence:
E1: Project-H is in the West region.
E2: Project-H is owned by Casey.

Evaluation:
EX-008 | NOT_SUPPORTED


TASK TO COMPLETE

"""

    prompt += "\n\n".join(
        format_item(item)
        for item in task["items"]
    )

    prompt += "\n\nEvaluations:\n"

    Path(args.output).write_text(
        prompt,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
