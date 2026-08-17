"""Two-sided substrate screen — the gate my first pre-check was missing.

Detection of a planted defect is only half of capability. A model that flags
correct lines as WRONG has not demonstrated it can verify; ARITH passed the
one-sided screen at 99.6% and then flagged 76% of correct `+` lines as WRONG.

This screens candidate item types in the FULL condition (which is what the
experiment actually runs) at a *small* ledger size, where workload cannot yet be
an excuse, and requires BOTH:

    sensitivity  >= 0.95   (planted defect flagged WRONG)
    specificity  >= 0.98   (clean lines called OK)

An item type that fails here cannot be used: its FULL failures would be
capability failure, not under-utilisation.

Usage:
  python -m v4.screen --model Qwen/Qwen2.5-3B-Instruct --out runs/v4_screen_3b
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from v3.run_sifo import Generator, SYSTEM_PROMPT, append_jsonl, read_jsonl, write_json
from v4.ledger import TASK_VERSION, build_ledger, build_prompt, parse_verdicts

DEFAULT_SEED = 141421
SENSITIVITY_GATE = 0.95
SPECIFICITY_GATE = 0.98


def messages_for(prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--item-types", default="ARITH,UPPER,COPY")
    parser.add_argument("--n-lines", default=10, type=int, help="small on purpose")
    parser.add_argument("--trials", default=25, type=int)
    args = parser.parse_args()

    item_types = [x.strip() for x in args.item_types.split(",") if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "screen_records.jsonl"
    done = {(r["item_type"], r["seed"]) for r in read_jsonl(checkpoint)}

    generator = Generator(args.model, args.device)
    write_json(args.out / "run_config_screen.json", {
        "task_version": TASK_VERSION, "model": args.model,
        "model_revision": generator.model_revision,
        "chat_template_sha256": generator.template_sha256,
        "seed": DEFAULT_SEED, "temperature": 0.0, "top_p": 1.0,
        "item_types": item_types, "n_lines": args.n_lines, "trials": args.trials,
        "sensitivity_gate": SENSITIVITY_GATE, "specificity_gate": SPECIFICITY_GATE,
        "condition": "FULL", "system_prompt": SYSTEM_PROMPT,
    })

    for item_type in item_types:
        for trial in range(args.trials):
            seed = DEFAULT_SEED + 7919 * trial
            if (item_type, seed) in done:
                continue
            position = 1 + (trial % args.n_lines)
            ledger = build_ledger(args.n_lines, position, seed, item_type)
            prompt = build_prompt(ledger, "FULL")
            response, metadata = generator.generate(messages_for(prompt), 200 + 24 * args.n_lines)
            verdicts = parse_verdicts(response)["verdicts"]
            per_line = []
            for line in ledger["lines"]:
                claimed = verdicts.get(line["line_id"])
                per_line.append({
                    "line_id": line["line_id"], "is_wrong": line["is_wrong"],
                    "claimed_wrong": claimed, "correct": claimed == line["is_wrong"],
                })
            append_jsonl(checkpoint, {
                "task_version": TASK_VERSION, "item_type": item_type, "seed": seed,
                "trial": trial, "n_lines": args.n_lines, "defect_position": position,
                "prompt": prompt, "raw_response": response, "per_line": per_line,
                "technical_failure": metadata["reached_generation_limit"],
                "metadata": metadata,
            })

    records = read_jsonl(checkpoint)
    stats = defaultdict(lambda: {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "unparsed": 0,
                                 "trials": 0, "technical_failures": 0})
    for record in records:
        bucket = stats[record["item_type"]]
        bucket["trials"] += 1
        bucket["technical_failures"] += record["technical_failure"]
        for line in record["per_line"]:
            if line["claimed_wrong"] is None:
                bucket["unparsed"] += 1
            elif line["is_wrong"]:
                bucket["tp" if line["claimed_wrong"] else "fn"] += 1
            else:
                bucket["fp" if line["claimed_wrong"] else "tn"] += 1
    result = {}
    for item_type, bucket in stats.items():
        sensitivity = bucket["tp"] / max(1, bucket["tp"] + bucket["fn"])
        specificity = bucket["tn"] / max(1, bucket["tn"] + bucket["fp"])
        result[item_type] = {
            **bucket, "sensitivity": sensitivity, "specificity": specificity,
            "passes_gate": sensitivity >= SENSITIVITY_GATE and specificity >= SPECIFICITY_GATE
            and bucket["unparsed"] == 0,
        }
    summary = {"task_version": TASK_VERSION, "n_lines": args.n_lines,
               "sensitivity_gate": SENSITIVITY_GATE, "specificity_gate": SPECIFICITY_GATE,
               "by_item_type": result,
               "usable_item_types": sorted(k for k, v in result.items() if v["passes_gate"])}
    write_json(args.out / "screen_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
