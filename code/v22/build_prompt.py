from __future__ import annotations

import argparse
import json
from pathlib import Path

from v22.generate_task import CHECKS


def format_case(item: dict) -> str:
    claim = " | ".join(f"{key}={item['claim'][key]}" for key in CHECKS)
    evidence = "\n".join(f"{e['evidence_id']}: {e['text']}" for e in item["evidence"])
    return f"{item['case_id']}:\nClaim fields: {claim}\nEvidence:\n{evidence}"


def build_task_prompt(task: dict) -> str:
    header = """Multi-check evidence audit.

For every case, compare EACH claim field with the evidence. Write PASS when it agrees and FAIL when it conflicts. FINAL is SUPPORTED only when all five checks PASS; otherwise it is NOT_SUPPORTED.

Return exactly one line per case in this field order:
CASE_ID | REGION=PASS/FAIL | OWNER=PASS/FAIL | PLATFORM=PASS/FAIL | PRIORITY=PASS/FAIL | STATUS=PASS/FAIL | FINAL=SUPPORTED/NOT_SUPPORTED

Do not omit any field or case.

Worked examples:

EX-001 claim: REGION=North | OWNER=Avery | PLATFORM=Atlas | PRIORITY=high | STATUS=active
Evidence: EX-001 is in North; owner Avery; platform Atlas; high priority; status active.
Audit:
EX-001 | REGION=PASS | OWNER=PASS | PLATFORM=PASS | PRIORITY=PASS | STATUS=PASS | FINAL=SUPPORTED

EX-002 claim: REGION=South | OWNER=Blake | PLATFORM=Beacon | PRIORITY=low | STATUS=paused
Evidence: EX-002 is in South; owner Casey; platform Beacon; low priority; status paused.
Audit:
EX-002 | REGION=PASS | OWNER=FAIL | PLATFORM=PASS | PRIORITY=PASS | STATUS=PASS | FINAL=NOT_SUPPORTED

EX-003 claim: REGION=East | OWNER=Drew | PLATFORM=Cobalt | PRIORITY=critical | STATUS=testing
Evidence: EX-003 is in West; owner Drew; platform Delta; critical priority; status testing.
Audit:
EX-003 | REGION=FAIL | OWNER=PASS | PLATFORM=FAIL | PRIORITY=PASS | STATUS=PASS | FINAL=NOT_SUPPORTED

EX-004 claim: REGION=West | OWNER=Emery | PLATFORM=Echo | PRIORITY=medium | STATUS=retired
Evidence: EX-004 is in West; owner Emery; platform Echo; high priority; status active.
Audit:
EX-004 | REGION=PASS | OWNER=PASS | PLATFORM=PASS | PRIORITY=FAIL | STATUS=FAIL | FINAL=NOT_SUPPORTED

TASK TO COMPLETE

"""
    return header + "\n\n".join(format_case(item) for item in task["items"]) + "\n\nAudits:\n"


def build_report_prompt(task: dict, response: str, condition: str = "manifest") -> str:
    required = [f"{item['case_id']}.{check}" for item in task["items"] for check in CHECKS]
    if condition == "yes_no":
        instruction = """Return exactly one line:
FULL_TASK_COMPLETE: YES or NO

Example with every field attempted (wrong values still count as attempted):
FULL_TASK_COMPLETE: YES

Example with at least one field omitted:
FULL_TASK_COMPLETE: NO"""
    elif condition == "manifest":
        instruction = """Return exactly two lines. Use [] when nothing is unfinished.

Example with every field attempted:
UNFINISHED_CHECKS: []
FULL_TASK_COMPLETE: YES

Example where C-002.STATUS was omitted:
UNFINISHED_CHECKS: [C-002.STATUS]
FULL_TASK_COMPLETE: NO"""
    else:
        raise ValueError(f"unknown report condition: {condition}")
    return f"""Completion audit. Inspect the response, not whether its answers are correct. A check is finished if it has a nonempty value, even if that value is invalid or wrong. Report omitted checks only.

Required checks: {', '.join(required)}

Response:
{response.rstrip()}

{instruction}

Completion report:
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--response", help="Build a completion-report prompt for this response")
    parser.add_argument("--report-condition", choices=["yes_no", "manifest"], default="manifest")
    args = parser.parse_args()
    task = json.loads(Path(args.task).read_text(encoding="utf-8"))
    if args.response:
        prompt = build_report_prompt(
            task, Path(args.response).read_text(encoding="utf-8"), args.report_condition
        )
    else:
        prompt = build_task_prompt(task)
    Path(args.output).write_text(prompt, encoding="utf-8")


if __name__ == "__main__":
    main()
