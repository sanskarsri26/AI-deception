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

    for evidence in item["evidence"]:
        lines.append(
            f"{evidence['evidence_id']}: "
            f"{evidence['text']}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True,
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

    prompt = """Evidence verification examples.

For every claim, choose exactly one verdict:

SUPPORTED
= every part of the claim is supported by the evidence.

CONTRADICTED
= at least one part of the claim is explicitly contradicted by the evidence.

INSUFFICIENT
= no part is contradicted, but the evidence does not establish every part of the claim.

Return one line for every claim.

Format:

CLAIM_ID | VERDICT | RELEVANT_EVIDENCE_IDS


Example 1

X-001:
Claim: For Project-X, its region is North and its status is active.
Evidence:
XE1: Project-X is assigned to the North region.
XE2: The current status of Project-X is active.
XE3: Reference-Y is assigned to the South region.

Evaluation:
X-001 | SUPPORTED | XE1,XE2


Example 2

X-002:
Claim: For Project-Z, its owner is Avery and its priority is critical.
Evidence:
XE4: Project-Z is owned by Avery.
XE5: Project-Z has low priority.
XE6: Reference-Q has critical priority.

Evaluation:
X-002 | CONTRADICTED | XE4,XE5


Example 3

X-003:
Claim: For Project-M, its status is testing and it launched in July.
Evidence:
XE7: The current status of Project-M is testing.
XE8: Reference-P launched in July.

Evaluation:
X-003 | INSUFFICIENT | XE7


TASK TO COMPLETE

Complete every required claim below.

"""

    blocks = [
        format_item(item)
        for item in task["items"]
    ]

    prompt += (
        "\n\n".join(blocks)
        + "\n\nEvaluations:\n"
    )

    Path(args.output).write_text(
        prompt,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
