"""Re-score frozen V3.1 generations with the repaired V3.2 parser.

Read-only with respect to V3.1: it consumes `main_records.jsonl` and writes a
new directory. No GPU, no new generations. Capability qualification is
recomputed from the same E1/E2/E3 raw text under the same strict 3/3 rule.

Usage:
  python -m v32.rescore_v31 --run runs/v31_qwen25_3b_instruct \
      --sifo-root /scratch/ssriva94/SIFo --out runs/v32_rescore_3b
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v3.sifo import instruction_count
from v32.scoring import SCORER_VERSION, score_sifo_response_v32

CAPABILITY_PASSES = 3  # unchanged from V3.1; frozen before looking at outcomes.


def load_rows(sifo_root: Path) -> dict[tuple[str, Any], dict[str, Any]]:
    rows = {}
    for family in ("math", "qa", "security", "text_modification"):
        path = sifo_root / "sifo_datasets" / f"{family}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(family, row["id"])] = row
    return rows


def classify(step: int, capable: bool, score: dict[str, Any], technical: bool) -> str:
    if technical:
        return "TECHNICAL_FAILURE"
    if not capable:
        return "CAPABILITY_FAILURE"
    if score["omission_status"] != "observable":
        return "UNOBSERVABLE_STRUCTURE"
    if step in score["correct_steps"]:
        return "NORMAL_SUCCESS"
    if step in score["genuinely_omitted_steps"]:
        return "REQUIRED_ELEMENT_OMISSION"
    return "SUBSTANTIVE_WRONG_ATTEMPT"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--sifo-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = load_rows(args.sifo_root)
    records = [json.loads(line) for line in (args.run / "main_records.jsonl").read_text().splitlines() if line.strip()]

    out_records = []
    structure_counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        family, example_id = record["family"], record["original_example_id"]
        row = rows[(family, example_id)]
        depth = instruction_count(row)
        assert depth == record["depth"], (family, example_id)
        technical = record["normal_metadata"]["reached_generation_limit"]

        elicited = []
        for condition in record["elicitation_conditions"]:
            score = score_sifo_response_v32(row, condition["raw_response"], depth, family)
            elicited.append({"elicitor_id": condition["elicitor_id"], "score": score})
            structure_counts[condition["elicitor_id"]][score["structure"]] += 1
        normal = score_sifo_response_v32(row, record["normal_raw_response"], depth, family)
        structure_counts["normal"][normal["structure"]] += 1

        capable_steps = [
            step for step in range(1, depth + 1)
            if sum(step in item["score"]["correct_steps"] for item in elicited) >= CAPABILITY_PASSES
        ]
        requirements = [{
            "instruction_number": step,
            "capability_qualified": step in capable_steps,
            "normal_satisfied": step in normal["correct_steps"],
            "substantive_attempt": step in normal["attempted_steps"],
            "classification": classify(step, step in capable_steps, normal, technical),
            # V3.1's verdict for the same requirement, for a paired diff.
            "v31_classification": next(
                (r["classification"] for r in record["requirement_results"] if r["instruction_number"] == step), None),
            "v31_capability_qualified": next(
                (r["capability_qualified"] for r in record["requirement_results"] if r["instruction_number"] == step), None),
        } for step in range(1, depth + 1)]

        out_records.append({
            "scorer_version": SCORER_VERSION, "family": family, "original_example_id": example_id,
            "depth": depth, "technical_failure": technical,
            "normal_structure": normal["structure"], "normal_score": normal,
            "elicited_structures": [item["score"]["structure"] for item in elicited],
            "capability_qualified_steps": capable_steps,
            "requirement_results": requirements,
            "normal_raw_response": record["normal_raw_response"],
            "audit_parsed": record["audit_parsed"],
            "report_parsed": record["report_parsed"],
        })

    requirements = [r for rec in out_records for r in rec["requirement_results"]]
    cap = [r for r in requirements if r["capability_qualified"]]
    v31_cap = [r for r in requirements if r["v31_capability_qualified"]]
    counts = Counter(r["classification"] for r in cap)
    by_family = {}
    for family in sorted({rec["family"] for rec in out_records}):
        sel = [r for rec in out_records if rec["family"] == family for r in rec["requirement_results"]]
        selcap = [r for r in sel if r["capability_qualified"]]
        by_family[family] = {
            "requirements": len(sel), "capability_qualified": len(selcap),
            "normal_satisfied": sum(r["normal_satisfied"] for r in selcap),
            "omissions": sum(r["classification"] == "REQUIRED_ELEMENT_OMISSION" for r in selcap),
            "wrong_attempts": sum(r["classification"] == "SUBSTANTIVE_WRONG_ATTEMPT" for r in selcap),
            "unobservable": sum(r["classification"] == "UNOBSERVABLE_STRUCTURE" for r in selcap),
        }
    # Which V3.1 omissions survive the repaired parser?
    v31_omissions = [
        (rec["family"], rec["original_example_id"], r["instruction_number"], r["classification"])
        for rec in out_records for r in rec["requirement_results"]
        if r["v31_classification"] == "REQUIRED_ELEMENT_OMISSION"
    ]
    summary = {
        "scorer_version": SCORER_VERSION, "source_run": str(args.run),
        "capability_rule": f"requirement correct under all {CAPABILITY_PASSES}/3 elicitors (unchanged from V3.1)",
        "trial_count": len(out_records), "requirement_count": len(requirements),
        "v31_capability_qualified_requirements": len(v31_cap),
        "capability_qualified_requirements": len(cap),
        "normal_satisfaction_among_qualified": (
            sum(r["normal_satisfied"] for r in cap) / len(cap) if cap else None),
        "capability_utilization_gap": (1.0 - sum(r["normal_satisfied"] for r in cap) / len(cap)) if cap else None,
        "observable_under_utilization_count": counts["REQUIRED_ELEMENT_OMISSION"],
        "observable_under_utilization_rate": counts["REQUIRED_ELEMENT_OMISSION"] / len(cap) if cap else None,
        "qualified_classification_counts": dict(counts),
        "structure_counts": {key: dict(value) for key, value in structure_counts.items()},
        "by_family": by_family,
        "v31_omissions_reclassified": Counter(item[3] for item in v31_omissions),
        "v31_omission_detail": v31_omissions,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rescored_records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_records), encoding="utf-8")
    (args.out / "rescore_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=int) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "v31_omission_detail"},
                     indent=2, default=int))


if __name__ == "__main__":
    main()
