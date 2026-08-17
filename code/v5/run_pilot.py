"""V5 pilot: does elision appear once required output becomes expensive?

Paired per manifest: `TARGETED` (one block, full manifest in context) establishes
that the block for item k is producible; `FULL` (every block) measures whether it
is actually produced. Token budget is deliberately generous so a truncated
generation is never mistaken for elision - that distinction is recorded.

Usage:
  python -m v5.run_pilot --model Qwen/Qwen2.5-7B-Instruct --out runs/v5_pilot_7b
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from v3.run_sifo import Generator, SYSTEM_PROMPT, append_jsonl, read_jsonl, write_json
from v5.elision import TASK_VERSION, build_manifest, build_prompt, score_full, score_targeted

DEFAULT_SEED = 577215


def messages_for(prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def budget_for(n_items: int) -> int:
    """~90 tokens per required block plus headroom; elision must be a choice."""
    return 400 + 90 * n_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--item-counts", default="10,30,60")
    parser.add_argument("--trials", default=10, type=int)
    parser.add_argument("--seed-base", default=DEFAULT_SEED, type=int,
                        help="disjoint from the exploratory sweep for confirmatory runs")
    args = parser.parse_args()

    item_counts = [int(x) for x in args.item_counts.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "pilot_records.jsonl"
    done = {(r["n_items"], r["seed"]) for r in read_jsonl(checkpoint)}

    generator = Generator(args.model, args.device)
    write_json(args.out / "run_config_pilot.json", {
        "task_version": TASK_VERSION, "model": args.model,
        "model_revision": generator.model_revision,
        "chat_template_sha256": generator.template_sha256,
        "seed": args.seed_base, "temperature": 0.0, "top_p": 1.0,
        "item_counts": item_counts, "trials": args.trials,
        "max_new_tokens_rule": "400 + 90 * n_items", "system_prompt": SYSTEM_PROMPT,
        "endpoint": "coverage, elision markers, in-artifact completion claims",
    })

    for n_items in item_counts:
        budget = budget_for(n_items)
        for trial in range(args.trials):
            seed = args.seed_base + 977 * n_items + trial
            if (n_items, seed) in done:
                continue
            manifest = build_manifest(n_items, seed)
            manifest["target_position"] = 1 + (trial % n_items)
            target_id = manifest["items"][manifest["target_position"] - 1]["item_id"]

            targeted_raw, targeted_meta = generator.generate(
                messages_for(build_prompt(manifest, "TARGETED")), 300)
            full_raw, full_meta = generator.generate(
                messages_for(build_prompt(manifest, "FULL")), budget)

            append_jsonl(checkpoint, {
                "task_version": TASK_VERSION, "n_items": n_items, "seed": seed,
                "trial": trial, "target_position": manifest["target_position"],
                "target_item_id": target_id,
                "targeted_raw_response": targeted_raw,
                "targeted_score": score_targeted(manifest, targeted_raw, target_id),
                "full_raw_response": full_raw,
                "full_score": score_full(manifest, full_raw),
                "max_new_tokens": budget,
                "technical_failure": full_meta["reached_generation_limit"],
                "targeted_metadata": targeted_meta, "full_metadata": full_meta,
            })
            print(f"n={n_items} trial={trial} done", flush=True)

    records = read_jsonl(checkpoint)
    summary = {}
    for n_items in sorted({r["n_items"] for r in records}):
        selected = [r for r in records if r["n_items"] == n_items]
        clean = [r for r in selected if not r["technical_failure"]]
        summary[str(n_items)] = {
            "trials": len(selected), "truncated": sum(r["technical_failure"] for r in selected),
            "targeted_produced": sum(r["targeted_score"]["produced"] for r in selected),
            "mean_coverage": sum(r["full_score"]["coverage"] for r in clean) / len(clean) if clean else None,
            "any_missing": sum(bool(r["full_score"]["missing_item_ids"]) for r in clean),
            "has_elision": sum(r["full_score"]["has_elision"] for r in clean),
            "asserts_completion": sum(r["full_score"]["asserts_completion"] for r in clean),
            "elided_with_completion_claim": sum(r["full_score"]["elided_with_completion_claim"] for r in clean),
            "clean_n": len(clean),
        }
    write_json(args.out / "pilot_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
