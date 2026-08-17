"""V9 driver: matched-workload capability utilization on the V5/V7/V8 substrate.

Three independent branches per fresh manifest - NORMAL, PLACEBO, CAREFUL - each
generated from the identical task state, none seeing another's output. Identical
manifests across conditions, so every comparison is paired.

Primary endpoints:

    H1  Coverage(CAREFUL) > Coverage(NORMAL)      necessary
    H2  Coverage(CAREFUL) > Coverage(PLACEBO)     the one that must hold

Plus a determinism check: the first `--determinism-manifests` seeds are
generated twice per condition. V3.2 found greedy decoding was byte-identical on
only 386/598 tasks on this stack, so a replicate pair establishes whether V9's
differences are stable or environment noise. It is not part of any endpoint.

After this, `v7.capability_pass` and `v7.audit_battery` run against the NORMAL
branch exactly as in V8 Stage 2 - unchanged code, so the numbers are directly
comparable.

Usage:
  python -m v9.run_elicit --model google/gemma-2-9b-it --out runs/v9_elicit_gemma \
      --n-items 120 --trials 60
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from v3.run_sifo import Generator, SYSTEM_PROMPT, append_jsonl, read_jsonl, write_json
from v5.elision import TASK_VERSION, build_manifest, score_full
from v9.elicit import CONDITIONS, build_prompt_v9, placebo_is_clean, suffix_lengths

DEFAULT_SEED = 9300000


def messages_for(prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def budget_for(n_items: int) -> int:
    """Identical rule to V5/V7/V8 so coverage numbers stay comparable."""
    return 400 + 90 * n_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--n-items", default=120, type=int)
    parser.add_argument("--trials", default=60, type=int, help="fresh manifests; frozen at 60")
    parser.add_argument("--seed-base", default=DEFAULT_SEED, type=int)
    parser.add_argument("--determinism-manifests", default=10, type=int,
                        help="first N manifests generated twice per condition")
    parser.add_argument("--shard", default=0, type=int,
                        help="this process's shard index; trials are split by "
                             "trial %% n_shards so shards never overlap")
    parser.add_argument("--n-shards", default=1, type=int,
                        help="set to the number of GPUs; run one process per GPU with "
                             "CUDA_VISIBLE_DEVICES set, all pointing at the same --out")
    args = parser.parse_args()
    if not 0 <= args.shard < args.n_shards:
        raise SystemExit(f"--shard {args.shard} out of range for --n-shards {args.n_shards}")

    if not placebo_is_clean():
        raise SystemExit("PLACEBO contains diligence language - aborting before generation")

    args.out.mkdir(parents=True, exist_ok=True)
    # Each shard appends to its own file so concurrent processes never interleave
    # writes; the resume set is read across every shard file plus any single-shard
    # file from an earlier run, so restarting with a different shard count is safe.
    checkpoint = (args.out / "elicit_records.jsonl" if args.n_shards == 1
                  else args.out / f"elicit_records_shard{args.shard}.jsonl")
    done = set()
    for path in sorted(args.out.glob("elicit_records*.jsonl")):
        done |= {(r["seed"], r["condition"], r["replicate"]) for r in read_jsonl(path)}
    budget = budget_for(args.n_items)

    generator = Generator(args.model, args.device)
    write_json(args.out / "run_config_elicit.json", {
        "task_version": TASK_VERSION, "model": args.model,
        "model_revision": generator.model_revision,
        "chat_template_sha256": generator.template_sha256,
        "temperature": 0.0, "top_p": 1.0, "system_prompt": SYSTEM_PROMPT,
        "n_items": args.n_items, "trials": args.trials, "seed_base": args.seed_base,
        "max_new_tokens": budget, "max_new_tokens_rule": "400 + 90 * n_items",
        "conditions": list(CONDITIONS), "suffix_lengths": suffix_lengths(),
        "determinism_manifests": args.determinism_manifests,
        "shard": args.shard, "n_shards": args.n_shards,
        "primary": "H1 coverage(CAREFUL) > coverage(NORMAL); "
                   "H2 coverage(CAREFUL) > coverage(PLACEBO)",
        "note": "NORMAL is the V5 FULL prompt byte-identical; PLACEBO and CAREFUL append "
                "one sentence each. Same substrate, scorer and budget as V5/V7/V8.",
    })

    for trial in range(args.trials):
        if trial % args.n_shards != args.shard:
            continue
        seed = args.seed_base + 7919 * trial
        manifest = build_manifest(args.n_items, seed)
        replicates = 2 if trial < args.determinism_manifests else 1
        for condition in CONDITIONS:
            for replicate in range(replicates):
                if (seed, condition, replicate) in done:
                    continue
                response, metadata = generator.generate(
                    messages_for(build_prompt_v9(manifest, condition)), budget)
                score = score_full(manifest, response)
                append_jsonl(checkpoint, {
                    "task_version": TASK_VERSION, "seed": seed, "trial": trial,
                    "condition": condition, "replicate": replicate,
                    "n_items": args.n_items, "raw_response": response,
                    "full_score": score, "max_new_tokens": budget,
                    "technical_failure": metadata["reached_generation_limit"],
                    "metadata": metadata,
                })
                print(f"trial={trial} {condition} rep={replicate} "
                      f"coverage={score['coverage']:.3f}", flush=True)

    # Summarise across every shard, not just this process's own file, so the
    # last shard to finish writes a complete summary.
    records = []
    for path in sorted(args.out.glob("elicit_records*.jsonl")):
        records.extend(read_jsonl(path))
    primary = [r for r in records if r["replicate"] == 0]
    by_condition = defaultdict(list)
    for record in primary:
        by_condition[record["condition"]].append(record)

    summary = {"task_version": TASK_VERSION, "model": args.model,
               "trials": args.trials, "records": len(records), "by_condition": {}}
    for condition in CONDITIONS:
        rows = by_condition.get(condition, [])
        clean = [r for r in rows if not r["technical_failure"]]
        if not clean:
            continue
        summary["by_condition"][condition] = {
            "n": len(rows), "clean_n": len(clean),
            "truncated": sum(r["technical_failure"] for r in rows),
            "mean_coverage": sum(r["full_score"]["coverage"] for r in clean) / len(clean),
            "fully_complete": sum(r["full_score"]["coverage"] >= 1.0 for r in clean),
            "any_missing": sum(bool(r["full_score"]["missing_item_ids"]) for r in clean),
            "has_elision": sum(r["full_score"]["has_elision"] for r in clean),
            "asserts_completion": sum(r["full_score"]["asserts_completion"] for r in clean),
        }

    # Determinism: byte-identical replicate pairs, per condition.
    pairs = defaultdict(lambda: [0, 0])
    grouped = defaultdict(dict)
    for record in records:
        grouped[(record["seed"], record["condition"])][record["replicate"]] = record
    for (seed, condition), reps in grouped.items():
        if 0 in reps and 1 in reps:
            pairs[condition][1] += 1
            pairs[condition][0] += int(reps[0]["raw_response"] == reps[1]["raw_response"])
    summary["determinism"] = {c: {"identical": k, "pairs": n,
                                  "rate": (k / n if n else None)}
                              for c, (k, n) in sorted(pairs.items())}

    write_json(args.out / "elicit_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
