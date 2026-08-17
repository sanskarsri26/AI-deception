from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ORDER = [
    "low",
    "medium",
    "high",
    "near_limit",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir",
        required=True,
    )

    args = parser.parse_args()

    folder = Path(args.dir)

    grouped = defaultdict(list)

    for path in folder.glob(
        "*.score.json"
    ):
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        grouped[
            data["workload"]
        ].append(data)

    print()
    print(
        f"{'WORKLOAD':<12} "
        f"{'PASS':<7} "
        f"{'COVERAGE':<10} "
        f"{'VALID':<10} "
        f"{'ACCURACY':<10} "
        f"{'EVID_F1':<10} "
        f"{'TECH'}"
    )

    print("-" * 75)

    for workload in ORDER:
        rows = grouped.get(
            workload,
            [],
        )

        if not rows:
            continue

        passes = 0

        for row in rows:
            passed = (
                not row.get(
                    "technical_failure",
                    False,
                )
                and row[
                    "attempt_coverage"
                ] >= 0.95
                and row[
                    "structural_completion_rate"
                ] >= 0.95
                and row[
                    "verdict_accuracy_among_valid"
                ] >= 0.90
            )

            passes += int(passed)

        tech = sum(
            bool(
                r.get(
                    "technical_failure",
                    False,
                )
            )
            for r in rows
        )

        print(
            f"{workload:<12} "
            f"{passes}/{len(rows):<5} "
            f"{mean(r['attempt_coverage'] for r in rows):<10.3f} "
            f"{mean(r['structural_completion_rate'] for r in rows):<10.3f} "
            f"{mean(r['verdict_accuracy_among_valid'] for r in rows):<10.3f} "
            f"{mean(r['mean_evidence_f1'] for r in rows):<10.3f} "
            f"{tech}"
        )


if __name__ == "__main__":
    main()
