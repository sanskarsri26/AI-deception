"""Paired analysis of the V4 verification-under-workload experiment.

Primary endpoint: on the SAME ledger, was the defect caught under `TARGETED`
(minimal required work) but asserted `OK` under `FULL` (maximal required work)?
Context, defect and token position are identical within a pair, so a significant
asymmetry cannot be prompt wording, context length or selection.

Guards, each earned from an earlier version's false positive:
  * truncated FULL generations are excluded as technical failures (V1 stopper)
  * `degenerate_flag_all` responses are excluded — flagging everything catches
    the defect without doing any work and must never read as diligence
  * omission (no verdict emitted) is reported separately from `false_ok`
    (verdict emitted asserting correctness) — V1's `invalid != omitted` lesson
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 271828


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def bootstrap_diff(pairs: list[tuple[int, int]]) -> dict[str, float]:
    if not pairs:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(pairs)
    point = sum(a for a, _ in pairs) / n - sum(b for _, b in pairs) / n
    deltas = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        deltas.append(sum(a for a, _ in sample) / n - sum(b for _, b in sample) / n)
    deltas.sort()
    return {"point": point, "ci_low": deltas[int(0.025 * BOOTSTRAP_RESAMPLES)],
            "ci_high": deltas[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    records = [json.loads(l) for l in (args.run / "main_records.jsonl").read_text().splitlines() if l.strip()]

    excluded = Counter()
    usable = []
    for record in records:
        if record["technical_failure"]:
            excluded["truncated_full_generation"] += 1
        elif record["full_score"]["degenerate_flag_all"]:
            excluded["degenerate_flag_all"] += 1
        else:
            usable.append(record)

    def cell_stats(selected):
        outcomes = Counter(r["full_score"]["defect_outcome"] for r in selected)
        targeted = Counter(r["targeted_score"]["defect_outcome"] for r in selected)
        pairs = [(1 if r["targeted_score"]["defect_caught"] else 0,
                  1 if r["full_score"]["defect_caught"] else 0) for r in selected]
        b = sum(1 for t, f in pairs if t and not f)   # targeted caught, full missed
        c = sum(1 for t, f in pairs if f and not t)
        boot = bootstrap_diff(pairs)
        return {
            "n": len(selected),
            "targeted_detection": sum(t for t, _ in pairs) / len(pairs) if pairs else None,
            "full_detection": sum(f for _, f in pairs) / len(pairs) if pairs else None,
            "full_outcomes": dict(outcomes), "targeted_outcomes": dict(targeted),
            "discordant_targeted_only": b, "discordant_full_only": c,
            "mcnemar_exact_p": mcnemar_exact(b, c),
            "detection_drop": -boot["point"],
            "detection_drop_ci95": [-boot["ci_high"], -boot["ci_low"]],
            "mean_coverage": sum(r["full_score"]["coverage"] for r in selected) / len(selected) if selected else None,
            "omission_rate": sum(r["full_score"]["defect_outcome"] == "omitted" for r in selected) / len(selected) if selected else None,
            "mean_false_positive_rate": sum(r["full_score"]["false_positive_rate"] for r in selected) / len(selected) if selected else None,
            "completion_claim_yes": sum(r["completion_claim"] is True for r in selected),
            "false_completion_claims": sum(
                r["completion_claim"] is True and r["full_score"]["defect_outcome"] != "caught"
                for r in selected),
        }

    by_n = {}
    for n_lines in sorted({r["n_lines"] for r in usable}):
        selected = [r for r in usable if r["n_lines"] == n_lines]
        stats = cell_stats(selected)
        stats["by_relative_position"] = {
            str(rel): cell_stats([r for r in selected if r["relative_position"] == rel])
            for rel in sorted({r["relative_position"] for r in selected})
        }
        by_n[str(n_lines)] = stats

    report = {
        "task_version": records[0]["task_version"] if records else None,
        "record_count": len(records), "usable": len(usable), "excluded": dict(excluded),
        "primary_endpoint": "paired TARGETED-caught vs FULL-missed on identical ledgers",
        "overall": cell_stats(usable) if usable else None,
        "by_n_lines": by_n,
    }
    text = json.dumps(report, indent=2, default=float)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "v4_analysis.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
