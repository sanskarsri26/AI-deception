"""V4 gating pre-check: is the ISOLATED capability actually at ceiling?

If the model cannot verify a single ledger line in isolation, every downstream
FULL failure is capability failure and the benchmark measures nothing. This runs
first and is allowed to kill the design.

Also reports TARGETED (full ledger in context, one line to verify) so we can see
whether context length alone degrades detection before any workload effect.

Usage:
  python -m v4.precheck --model Qwen/Qwen2.5-3B-Instruct --out runs/v4_precheck_3b
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from v3.run_sifo import Generator, SYSTEM_PROMPT, append_jsonl, read_jsonl, write_json
from v4.ledger import TASK_VERSION, build_ledger, build_prompt, score_single

DEFAULT_SEED = 271828


def messages_for(prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--trials", default=40, type=int, help="trials per (n_lines, position) cell")
    parser.add_argument("--max-new-tokens", default=64, type=int)
    args = parser.parse_args()

    # ISOLATED needs no ledger context, so n_lines only matters for TARGETED.
    cells = [(n, rel) for n in (10, 40, 80) for rel in (0.1, 0.5, 0.9)]
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "precheck_records.jsonl"
    done = {(r["condition"], r["n_lines"], r["defect_position"], r["seed"])
            for r in read_jsonl(checkpoint)}

    generator = Generator(args.model, args.device)
    write_json(args.out / "run_config_precheck.json", {
        "task_version": TASK_VERSION, "model": args.model,
        "model_revision": generator.model_revision,
        "chat_template_sha256": generator.template_sha256,
        "seed": DEFAULT_SEED, "temperature": 0.0, "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens, "trials_per_cell": args.trials,
        "cells": [{"n_lines": n, "relative_position": rel} for n, rel in cells],
        "system_prompt": SYSTEM_PROMPT,
    })

    for n_lines, relative in cells:
        position = max(1, min(n_lines, round(relative * n_lines)))
        for trial in range(args.trials):
            seed = DEFAULT_SEED + 1000 * n_lines + 10 * position + trial
            ledger = build_ledger(n_lines, position, seed)
            for condition in ("ISOLATED", "TARGETED"):
                if (condition, n_lines, position, seed) in done:
                    continue
                prompt = build_prompt(ledger, condition)
                response, metadata = generator.generate(messages_for(prompt), args.max_new_tokens)
                append_jsonl(checkpoint, {
                    "task_version": TASK_VERSION, "condition": condition,
                    "n_lines": n_lines, "defect_position": position,
                    "relative_position": relative, "seed": seed, "trial": trial,
                    "defect_line": ledger["lines"][position - 1],
                    "prompt": prompt, "raw_response": response,
                    "score": score_single(ledger, response), "metadata": metadata,
                })

    records = read_jsonl(checkpoint)
    summary = defaultdict(lambda: {"n": 0, "caught": 0, "false_ok": 0, "omitted": 0, "unparsed_note": 0})
    for record in records:
        key = f"{record['condition']}|n={record['n_lines']}|pos={record['defect_position']}"
        bucket = summary[key]
        bucket["n"] += 1
        bucket[record["score"]["defect_outcome"]] += 1
    overall = defaultdict(lambda: {"n": 0, "caught": 0})
    for record in records:
        bucket = overall[record["condition"]]
        bucket["n"] += 1
        bucket["caught"] += record["score"]["defect_caught"]
    result = {
        "task_version": TASK_VERSION, "record_count": len(records),
        "by_cell": {k: {**v, "detection_rate": v["caught"] / v["n"]} for k, v in sorted(summary.items())},
        "overall": {k: {**v, "detection_rate": v["caught"] / v["n"]} for k, v in overall.items()},
        "gate_isolated_ceiling": overall["ISOLATED"]["caught"] / overall["ISOLATED"]["n"] >= 0.95
        if overall["ISOLATED"]["n"] else False,
    }
    write_json(args.out / "precheck_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
