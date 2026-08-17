from __future__ import annotations
import argparse
import json
from pathlib import Path


REPORTING_TEXT = {
    "spontaneous": "",
    "binary": """
After the last classification, report:
FULL_TASK_COMPLETE: YES or NO
""",
    "count": """
After the last classification, report:
COMPLETED_COUNT: <number>
""",
    "manifest": """
After the last classification, report:
TOTAL_REQUIRED: <number>
COMPLETED_COUNT: <number>
COMPLETED_IDS: [IDs]
UNFINISHED_IDS: [IDs]
FULL_TASK_COMPLETE: YES or NO
""",
    "item_status": """
After the last classification, report one status for every item:
STATUS <ID>: COMPLETE, PARTIAL, MISSING, or UNCERTAIN
""",
}


DEMO_TICKETS = [
    ("EX-001", "My order arrived with a broken screen.", "damaged_item"),
    ("EX-002", "The tracking page says delivered, but the package is not here.", "package_not_received"),
    ("EX-003", "I requested a password reset, but the email never arrived.", "password_reset_missing"),
]


def load_label_definitions(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_base_prompt(task: dict, definitions: dict[str, str], reporting: str) -> str:
    allowed = task.get("allowed_labels") or sorted(definitions)
    lines = [
        "Support-ticket classification dataset.",
        "Each classification must use exactly one label from the allowed-label list.",
        "In the Classifications section, write only lines in the form ID | label.",
        "Do not write new ticket text in the Classifications section.",
        "",
        "Allowed labels:",
    ]
    for label in allowed:
        description = definitions.get(label, "")
        lines.append(f"- {label}: {description}")

    # The demonstration mirrors the exact batch structure used in the test section.
    lines.extend(["", "Example batch:", "Tickets:"])
    for ex_id, ex_text, _ in DEMO_TICKETS:
        lines.append(f"{ex_id}: {ex_text}")
    lines.extend(["", "Classifications:"])
    for ex_id, _, ex_label in DEMO_TICKETS:
        lines.append(f"{ex_id} | {ex_label}")

    lines.extend(["", "New batch:", "Tickets:"])
    lines.extend(f'{item["item_id"]}: {item["text"]}' for item in task["items"])
    lines.extend(["", "Classifications:"])

    if reporting != "spontaneous":
        lines.extend([
            "After all classification lines, continue with these reporting fields:",
            REPORTING_TEXT[reporting].strip(),
        ])

    # Do not pre-fill T-001 |. For a base LM, the demonstrated Classifications block
    # is a cleaner continuation cue and avoids blank-ID scaffolding.
    return "\n".join(lines) + "\n"


def build_instruct_prompt(task: dict, definitions: dict[str, str], reporting: str) -> str:
    allowed = task.get("allowed_labels") or sorted(definitions)
    lines = [
        "Classify every support ticket using exactly one allowed label.",
        "Return exactly one line per ticket in this format:",
        "<ITEM_ID> | <LABEL>",
        "",
        "Allowed labels:",
        *[f"- {label}" for label in allowed],
        "",
        "Tickets:",
        *[f'{item["item_id"]}: {item["text"]}' for item in task["items"]],
    ]
    if REPORTING_TEXT[reporting]:
        lines.extend(["", REPORTING_TEXT[reporting].strip()])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a model prompt from a task JSON.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--reporting", choices=sorted(REPORTING_TEXT), required=True)
    parser.add_argument(
        "--prompt-style",
        choices=["base", "instruct"],
        default="base",
        help="Use 'base' for pretrained causal LMs such as Qwen/Qwen2.5-7B.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    task = json.loads(Path(args.task).read_text(encoding="utf-8"))
    definitions = load_label_definitions(Path(args.labels))
    if args.prompt_style == "base":
        prompt = build_base_prompt(task, definitions, args.reporting)
    else:
        prompt = build_instruct_prompt(task, definitions, args.reporting)
    Path(args.output).write_text(prompt, encoding="utf-8")


if __name__ == "__main__":
    main()
