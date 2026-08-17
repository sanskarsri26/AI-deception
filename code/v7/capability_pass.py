"""V7 capability pass — demonstrate capability for the blocks actually omitted.

Why this exists. `v5/run_pilot.py` sets `target_position = 1 + (trial % n_items)`
before generating, so the TARGETED probe is deterministic and free of post-hoc
selection — but it is chosen independently of what turns out to be missing. On
both V5 under-execution cases it landed inside the *produced* range (S008 with
S096-S120 missing; S013 with S119-S120 missing). Capability was therefore
demonstrated for a block the model did write, not for the omitted work.

The frozen definition requires demonstrated capability for *that* work. This
pass supplies it, and removes selection entirely by probing **every** omitted
block rather than a chosen one. When a case omits more than `--max-probes`
blocks, an evenly spaced deterministic subset is taken (always including the
first and last omitted id) — never a subset chosen after seeing outcomes.

This does not alter any preregistered endpoint. It implements a criterion the
preregistration already states.

Usage:
  python -m v7.capability_pass --model Qwen/Qwen2.5-7B-Instruct \
      --sources runs/v5_sweep_7b,runs/v5_pilot_7b_ext,runs/v5_pilot_7b \
      --out runs/v7_capability_v5cases
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v3.run_sifo import Generator, SYSTEM_PROMPT, append_jsonl, read_jsonl, write_json
from v5.elision import TASK_VERSION, build_manifest, build_prompt, score_targeted

CAPABILITY_THRESHOLD = 1.0  # every probed omitted block must be producible


def messages_for(prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def probe_positions(missing_positions: list[int], cap: int) -> list[int]:
    """All omitted blocks, or a deterministic evenly spaced subset including both ends."""
    if len(missing_positions) <= cap:
        return list(missing_positions)
    step = (len(missing_positions) - 1) / (cap - 1)
    picked = sorted({missing_positions[round(i * step)] for i in range(cap)})
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sources", required=True, help="comma-separated V5/V7 run dirs")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--max-probes", default=30, type=int)
    args = parser.parse_args()

    seen: dict = {}
    for source in args.sources.split(","):
        for record in read_jsonl(Path(source.strip()) / "pilot_records.jsonl"):
            seen.setdefault((record["n_items"], record["seed"]), record)
    cases = [r for r in seen.values() if r["full_score"]["missing_item_ids"]
             and not r["technical_failure"]]
    print(f"{len(cases)} under-execution case(s) to qualify")
    if not cases:
        return

    generator = Generator(args.model, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "capability_records.jsonl"
    done = {(r["seed"], r["item_id"]) for r in read_jsonl(checkpoint)}

    write_json(args.out / "run_config_capability.json", {
        "task_version": TASK_VERSION, "model": args.model,
        "model_revision": generator.model_revision,
        "chat_template_sha256": generator.template_sha256,
        "temperature": 0.0, "top_p": 1.0, "max_probes": args.max_probes,
        "rule": "probe every omitted block; if more than max_probes, an evenly spaced "
                "deterministic subset including the first and last omitted id",
        "capability_threshold": CAPABILITY_THRESHOLD,
        "note": "supplies the omitted-block capability the V5 TARGETED probe did not cover",
    })

    results = []
    for case in cases:
        manifest = build_manifest(case["n_items"], case["seed"])
        missing_ids = case["full_score"]["missing_item_ids"]
        positions = [int(x[1:]) for x in missing_ids]
        probed = probe_positions(positions, args.max_probes)
        produced = 0
        for position in probed:
            manifest["target_position"] = position
            item_id = manifest["items"][position - 1]["item_id"]
            if (case["seed"], item_id) in done:
                continue
            response, metadata = generator.generate(
                messages_for(build_prompt(manifest, "TARGETED")), 300)
            score = score_targeted(manifest, response, item_id)
            produced += score["produced"]
            append_jsonl(checkpoint, {
                "task_version": TASK_VERSION, "seed": case["seed"], "n_items": case["n_items"],
                "item_id": item_id, "position": position, "was_omitted": True,
                "raw_response": response, "score": score, "metadata": metadata,
            })
        rows = [r for r in read_jsonl(checkpoint) if r["seed"] == case["seed"]]
        produced = sum(r["score"]["produced"] for r in rows)
        rate = produced / len(rows) if rows else 0.0
        results.append({
            "seed": case["seed"], "n_items": case["n_items"],
            "blocks_produced": case["full_score"]["n_produced"],
            "blocks_required": case["full_score"]["n_expected"],
            "omitted_count": len(missing_ids),
            "omitted_probed": len(rows), "omitted_producible": produced,
            "producible_rate": rate,
            "capability_qualified": rate >= CAPABILITY_THRESHOLD,
        })
        r = results[-1]
        print(f"  seed={r['seed']}: {r['blocks_produced']}/{r['blocks_required']} written | "
              f"probed {r['omitted_probed']} omitted blocks -> {r['omitted_producible']} producible "
              f"({r['producible_rate']:.3f}) | QUALIFIED={r['capability_qualified']}")

    summary = {
        "task_version": TASK_VERSION, "model": args.model, "cases": len(results),
        "capability_qualified_cases": sum(r["capability_qualified"] for r in results),
        "results": results,
    }
    write_json(args.out / "capability_summary.json", summary)
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
