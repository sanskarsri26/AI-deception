"""Preregistered analysis of the V3.2 control run (see v32/PREREGISTRATION.md).

Reads the frozen V3.1 main records (for the Normal / N1 condition and for the
E1 ∧ E2 ∧ E3 capability qualification) plus the V3.2 control records, re-scores
everything with the repaired V3.2 scorer, and reports the preregistered
contrasts. Writes nothing into the V3.1 directory.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

from v3.sifo import instruction_count
from v32.rescore_v31 import CAPABILITY_PASSES, load_rows
from v32.scoring import score_sifo_response_v32

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 314159


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial sign test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def clustered_bootstrap(pairs: list[tuple[int, int]], resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, float]:
    """Task-clustered bootstrap CI for mean(a) - mean(b) over paired task outcomes."""
    if not pairs:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(pairs)
    point = sum(a for a, _ in pairs) / n - sum(b for _, b in pairs) / n
    deltas = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        deltas.append(sum(a for a, _ in sample) / n - sum(b for _, b in sample) / n)
    deltas.sort()
    return {
        "point": point,
        "ci_low": deltas[int(0.025 * resamples)],
        "ci_high": deltas[int(0.975 * resamples) - 1],
    }


def format_violation(score: dict[str, Any]) -> int:
    """Primary endpoint: SIFo's explicit output-format requirement was not met."""
    return 0 if score.get("structure") == "single_object" else 1


def contrast(name: str, tasks: list, left: dict, right: dict, indicator) -> dict[str, Any]:
    pairs = [(indicator(left[t]), indicator(right[t])) for t in tasks]
    b = sum(1 for x, y in pairs if x and not y)   # left violates, right does not
    c = sum(1 for x, y in pairs if y and not x)
    boot = clustered_bootstrap(pairs)
    return {
        "contrast": name, "n_tasks": len(pairs),
        "left_rate": sum(x for x, _ in pairs) / len(pairs),
        "right_rate": sum(y for _, y in pairs) / len(pairs),
        "discordant_left_only": b, "discordant_right_only": c,
        "mcnemar_exact_p": mcnemar_exact(b, c),
        "difference_point": boot["point"],
        "difference_ci95": [boot["ci_low"], boot["ci_high"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v31-run", default=Path("runs/v31_qwen25_3b_instruct"), type=Path)
    parser.add_argument("--control-run", required=True, type=Path)
    parser.add_argument("--sifo-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = load_rows(args.sifo_root)
    v31 = [json.loads(l) for l in (args.v31_run / "main_records.jsonl").read_text().splitlines() if l.strip()]
    control = [json.loads(l) for l in (args.control_run / "control_records.jsonl").read_text().splitlines() if l.strip()]

    # condition -> task key -> score  (Normal/E1..E3 rescored from frozen V3.1 text)
    scores: dict[str, dict[tuple, dict]] = defaultdict(dict)
    raw: dict[str, dict[tuple, str]] = defaultdict(dict)
    capability: dict[tuple, list[int]] = {}
    depths: dict[tuple, int] = {}
    for record in v31:
        key = (record["family"], record["original_example_id"])
        row = rows[key]
        depth = instruction_count(row)
        depths[key] = depth
        elicited = []
        for condition in record["elicitation_conditions"]:
            score = score_sifo_response_v32(row, condition["raw_response"], depth, record["family"])
            scores[condition["elicitor_id"]][key] = score
            raw[condition["elicitor_id"]][key] = condition["raw_response"]
            elicited.append(score)
        scores["N1"][key] = score_sifo_response_v32(row, record["normal_raw_response"], depth, record["family"])
        raw["N1"][key] = record["normal_raw_response"]
        capability[key] = [
            step for step in range(1, depth + 1)
            if sum(step in s["correct_steps"] for s in elicited) >= CAPABILITY_PASSES
        ]
    for record in control:
        key = (record["family"], record["original_example_id"])
        scores[record["condition"]][key] = record["score"]
        raw[record["condition"]][key] = record["raw_response"]

    present = [c for c in ("N1", "N2", "E1", "E2", "E3", "E4", "E5", "P1", "P2") if c in scores]
    tasks = sorted(set.intersection(*(set(scores[c]) for c in present)))

    report: dict[str, Any] = {
        "conditions_present": present, "n_tasks_complete_across_conditions": len(tasks),
        "endpoint_primary": "SIFo explicit output-format requirement violated (response is not exactly one well-formed JSON object)",
    }

    # --- determinism check ---------------------------------------------------
    if "N2" in scores:
        shared = [t for t in tasks if t in raw["N2"]]
        identical = sum(raw["N2"][t] == raw["N1"][t] for t in shared)
        report["determinism"] = {
            "n_compared": len(shared), "byte_identical_to_N1": identical,
            "identical_fraction": identical / len(shared) if shared else None,
            "note": "greedy decoding: if ~1.0, repeated-normal runs carry no information "
                    "and run-to-run sampling noise cannot explain normal/elicited gaps",
        }

    # --- primary endpoint: format-requirement violation ----------------------
    report["format_violation_rates"] = {
        c: sum(format_violation(scores[c][t]) for t in tasks) / len(tasks) for c in present
    }
    report["format_violation_counts"] = {
        c: sum(format_violation(scores[c][t]) for t in tasks) for c in present
    }
    planned = [("E4", "P1"), ("E4", "N1"), ("P1", "N1"), ("E5", "P2"), ("E5", "N1"), ("P2", "N1"),
               ("E1", "N1"), ("E2", "N1"), ("E3", "N1"), ("E1", "P1")]
    report["format_contrasts"] = [
        contrast(f"{a} vs {b}", tasks, scores[a], scores[b], format_violation)
        for a, b in planned if a in scores and b in scores
    ]

    # --- secondary: satisfaction among E1^E2^E3-qualified requirements -------
    qualified_tasks = [t for t in tasks if capability[t]]
    secondary = {}
    for condition in present:
        hit = total = 0
        for t in qualified_tasks:
            correct = set(scores[condition][t]["correct_steps"])
            for step in capability[t]:
                total += 1
                hit += step in correct
        secondary[condition] = {"qualified_requirements": total,
                                "satisfied": hit, "rate": hit / total if total else None}
    report["qualified_requirement_satisfaction"] = secondary
    report["qualified_requirement_note"] = (
        "E1/E2/E3 are 1.0 by construction (they define qualification). Only N1/N2/E4/E5/P1/P2 "
        "are unbiased here; E4 vs P1 is the confirmatory comparison."
    )

    # Task-level paired test on "all qualified requirements satisfied".
    def all_qualified_ok(condition):
        return lambda t: 0 if set(capability[t]) <= set(scores[condition][t]["correct_steps"]) else 1
    report["qualified_task_contrasts"] = []
    for a, b in (("E4", "P1"), ("E4", "N1"), ("P1", "N1"), ("E5", "P2"), ("N2", "N1")):
        if a in scores and b in scores:
            pairs = [(all_qualified_ok(a)(t), all_qualified_ok(b)(t)) for t in qualified_tasks]
            bb = sum(1 for x, y in pairs if x and not y)
            cc = sum(1 for x, y in pairs if y and not x)
            boot = clustered_bootstrap(pairs)
            report["qualified_task_contrasts"].append({
                "contrast": f"{a} vs {b}", "n_tasks": len(pairs),
                "left_failure_rate": sum(x for x, _ in pairs) / len(pairs),
                "right_failure_rate": sum(y for _, y in pairs) / len(pairs),
                "discordant_left_only": bb, "discordant_right_only": cc,
                "mcnemar_exact_p": mcnemar_exact(bb, cc),
                "difference_point": boot["point"], "difference_ci95": [boot["ci_low"], boot["ci_high"]],
            })

    # --- selection-similarity check (added after the preregistered contrasts) --
    # E4/E5 are carefulness prompts and qualification selected requirements that
    # carefulness prompts get right, so E4 may beat P1 on the qualified subset
    # purely by resembling the selector. The same contrast computed WITHOUT any
    # gating has no such advantage; if the qualified-subset effect is causal it
    # should survive here, and if it is selection it should vanish.
    total_requirements = sum(depths[t] for t in tasks)
    report["ungated_requirement_satisfaction"] = {
        c: {"satisfied": sum(len(scores[c][t]["correct_steps"]) for t in tasks),
            "requirements": total_requirements,
            "rate": sum(len(scores[c][t]["correct_steps"]) for t in tasks) / total_requirements}
        for c in present
    }
    report["ungated_task_contrasts"] = []
    for a, b in (("E4", "P1"), ("E4", "N1"), ("P1", "N1"), ("E5", "P2"), ("N2", "N1"), ("E1", "P1")):
        if a in scores and b in scores:
            pairs = [(0 if scores[a][t]["all_steps_correct"] else 1,
                      0 if scores[b][t]["all_steps_correct"] else 1) for t in tasks]
            bb = sum(1 for x, y in pairs if x and not y)
            cc = sum(1 for x, y in pairs if y and not x)
            boot = clustered_bootstrap(pairs)
            report["ungated_task_contrasts"].append({
                "contrast": f"{a} vs {b}", "n_tasks": len(pairs),
                "left_failure_rate": sum(x for x, _ in pairs) / len(pairs),
                "right_failure_rate": sum(y for _, y in pairs) / len(pairs),
                "discordant_left_only": bb, "discordant_right_only": cc,
                "mcnemar_exact_p": mcnemar_exact(bb, cc),
                "difference_point": boot["point"], "difference_ci95": [boot["ci_low"], boot["ci_high"]],
            })

    # --- tertiary: required-element omissions among qualified requirements ---
    omissions = {}
    for condition in present:
        count = 0
        for t in qualified_tasks:
            score = scores[condition][t]
            if score["omission_status"] != "observable":
                continue
            count += sum(step in score["genuinely_omitted_steps"] for step in capability[t])
        omissions[condition] = count
    report["qualified_requirement_omissions"] = omissions

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "control_analysis.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
