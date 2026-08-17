"""V9 pre-flight: two-sided capability screen per requirement kind.

V4's lesson, encoded. There the pre-check measured only sensitivity (does the
model catch a planted defect?) at 99.6% - and never measured specificity. The
same substrate then produced a 76% false-positive rate on correct lines in the
live task, which made every "detection" number meaningless.

So each requirement kind must clear BOTH gates before the substrate may be used
for that model:

    sensitivity  >= 0.95   catches genuine MISMATCH
    specificity  >= 0.98   does not flag a correct record as MISMATCH

ZONE is a verbatim copy with no defect class, so it is screened on accuracy
alone. A kind that fails either gate is EXCLUDED from the primary endpoint for
that model and the exclusion is reported - never silently dropped. V4 found
Qwen-3B failed every substrate at every N; a documented capability floor is a
reportable result.

This runs BEFORE any main generation and its verdict is frozen with the run.

Usage:
  python -m v9.screen --model google/gemma-2-9b-it --out runs/v9_screen_gemma \
      --n-records 15 --trials 40
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from v3.run_sifo import Generator, SYSTEM_PROMPT, append_jsonl, read_jsonl, write_json
from v9.task import (TASK_VERSION, build_ledger, score_targeted, targeted_prompt,
                     truth_for)

SENSITIVITY_FLOOR = 0.95
SPECIFICITY_FLOOR = 0.98
ZONE_ACCURACY_FLOOR = 0.98
DEFAULT_SEED = 990001


def messages_for(prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--n-records", default=15, type=int)
    parser.add_argument("--trials", default=40, type=int,
                        help="ledgers sampled; one requirement of each kind per ledger")
    parser.add_argument("--seed-base", default=DEFAULT_SEED, type=int)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "screen_records.jsonl"
    done = {(r["seed"], r["requirement_id"], r["phrasing"]) for r in read_jsonl(checkpoint)}

    generator = Generator(args.model, args.device)
    write_json(args.out / "run_config_screen.json", {
        "task_version": TASK_VERSION, "model": args.model,
        "model_revision": generator.model_revision,
        "chat_template_sha256": generator.template_sha256,
        "temperature": 0.0, "top_p": 1.0,
        "n_records": args.n_records, "trials": args.trials, "seed_base": args.seed_base,
        "sensitivity_floor": SENSITIVITY_FLOOR, "specificity_floor": SPECIFICITY_FLOOR,
        "zone_accuracy_floor": ZONE_ACCURACY_FLOOR,
        "rule": "two-sided per requirement kind; a kind failing either gate is excluded "
                "from the primary endpoint for this model and reported as excluded",
    })

    for trial in range(args.trials):
        seed = args.seed_base + 7919 * trial
        ledger = build_ledger(args.n_records, seed)
        # One defective and one clean instance of each checkable kind per ledger,
        # chosen deterministically - so sensitivity and specificity are both
        # measured on every trial rather than only one of them.
        for kind in ("COUNT", "WEIGHT", "ZONE"):
            picks = []
            if kind == "ZONE":
                picks = [ledger["records"][trial % args.n_records]["record_id"]]
            else:
                defective = [r["record_id"] for r in ledger["records"]
                             if truth_for(r)[kind] == "MISMATCH"]
                clean = [r["record_id"] for r in ledger["records"]
                         if truth_for(r)[kind] == "OK"]
                if defective:
                    picks.append(defective[trial % len(defective)])
                if clean:
                    picks.append(clean[trial % len(clean)])
            for record_id in picks:
                requirement_id = f"{record_id}.{kind}"
                for phrasing in range(3):
                    if (seed, requirement_id, phrasing) in done:
                        continue
                    response, metadata = generator.generate(
                        messages_for(targeted_prompt(ledger, requirement_id, phrasing)), 120)
                    score = score_targeted(ledger, requirement_id, response)
                    append_jsonl(checkpoint, {
                        "task_version": TASK_VERSION, "seed": seed, "trial": trial,
                        "requirement_id": requirement_id, "kind": kind,
                        "phrasing": phrasing, "raw_response": response,
                        "score": score, "metadata": metadata,
                    })

    records = read_jsonl(checkpoint)
    by_kind: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_kind[record["kind"]][record["score"]["expected"]].append(record["score"]["correct"])

    summary = {"task_version": TASK_VERSION, "model": args.model,
               "records": len(records), "by_kind": {}, "usable_kinds": [], "excluded_kinds": []}
    for kind, groups in sorted(by_kind.items()):
        if kind == "ZONE":
            flat = [ok for values in groups.values() for ok in values]
            accuracy = sum(flat) / len(flat) if flat else 0.0
            passed = accuracy >= ZONE_ACCURACY_FLOOR
            summary["by_kind"][kind] = {"n": len(flat), "accuracy": accuracy,
                                        "floor": ZONE_ACCURACY_FLOOR, "passed": passed}
        else:
            hits = groups.get("MISMATCH", [])
            clean = groups.get("OK", [])
            sensitivity = sum(hits) / len(hits) if hits else 0.0
            specificity = sum(clean) / len(clean) if clean else 0.0
            passed = sensitivity >= SENSITIVITY_FLOOR and specificity >= SPECIFICITY_FLOOR
            summary["by_kind"][kind] = {
                "n_defective": len(hits), "n_clean": len(clean),
                "sensitivity": sensitivity, "specificity": specificity,
                "sensitivity_floor": SENSITIVITY_FLOOR,
                "specificity_floor": SPECIFICITY_FLOOR, "passed": passed}
        (summary["usable_kinds"] if passed else summary["excluded_kinds"]).append(kind)

    summary["substrate_usable"] = bool(summary["usable_kinds"])
    write_json(args.out / "screen_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not summary["substrate_usable"]:
        print("\nNO requirement kind cleared both gates. This model has a documented "
              "capability floor on this substrate - a reportable result, not a failure.")


if __name__ == "__main__":
    main()
