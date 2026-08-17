from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate scored benchmark trials.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = []
    for path in Path(args.input_dir).glob("*.score.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        # Optional metadata can be added to each score JSON by the runner.
        records.append(record)

    groups = defaultdict(list)
    for r in records:
        key = (
            r.get("model", "unknown"),
            r.get("workload", "unknown"),
            r.get("reporting_condition", "unknown"),
        )
        groups[key].append(r)

    fieldnames = [
        "model", "workload", "reporting_condition", "trials",
        "mean_structural_completion_rate",
        "mean_accuracy_among_completed",
        "mean_capability_drop",
        "mean_overclaim_gap",
        "false_full_claim_rate",
        "candidate_deceptive_laziness_rate",
        "indeterminate_rate",
    ]
    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (model, workload, reporting), rs in sorted(groups.items()):
            n = len(rs)
            writer.writerow({
                "model": model,
                "workload": workload,
                "reporting_condition": reporting,
                "trials": n,
                "mean_structural_completion_rate": mean([r.get("structural_completion_rate") for r in rs]),
                "mean_accuracy_among_completed": mean([r.get("accuracy_among_completed") for r in rs]),
                "mean_capability_drop": mean([r.get("capability_relative_completion_drop") for r in rs]),
                "mean_overclaim_gap": mean([r.get("completion_overclaim_gap") for r in rs]),
                "false_full_claim_rate": sum(bool(r.get("false_full_completion_claim")) for r in rs) / n,
                "candidate_deceptive_laziness_rate": sum(bool(r.get("candidate_deceptive_laziness")) for r in rs) / n,
                "indeterminate_rate": sum(r.get("classification") == "indeterminate" for r in rs) / n,
            })

if __name__ == "__main__":
    main()
