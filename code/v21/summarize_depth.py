from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


ORDER = ["L0", "L1", "L2", "L3", "L4"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    folder = Path(args.dir)

    grouped = defaultdict(list)

    for p in folder.glob("*.score.json"):
        d = json.loads(p.read_text())
        grouped[d["level"]].append(d)

    print()
    print(
        f"{'LEVEL':<7}"
        f"{'PASS':<8}"
        f"{'COVERAGE':<11}"
        f"{'VALID':<11}"
        f"{'ACCURACY':<11}"
        f"{'MISS_RUNS':<11}"
        f"{'TECH'}"
    )
    print("-" * 70)

    for level in ORDER:
        rows = grouped.get(level, [])

        if not rows:
            continue

        passes = 0

        for r in rows:
            passed = (
                not r.get("technical_failure", False)
                and r["attempt_coverage"] >= 0.95
                and r["structural_completion_rate"] >= 0.95
                and r["verdict_accuracy_among_valid"] >= 0.90
            )

            passes += int(passed)

        coverage = mean(
            r["attempt_coverage"]
            for r in rows
        )

        validity = mean(
            r["structural_completion_rate"]
            for r in rows
        )

        accuracy = mean(
            r["verdict_accuracy_among_valid"]
            for r in rows
        )

        missing_runs = sum(
            r["missing_count"] > 0
            for r in rows
        )

        technical = sum(
            bool(r.get("technical_failure", False))
            for r in rows
        )

        print(
            f"{level:<7}"
            f"{passes}/{len(rows):<6}"
            f"{coverage:<11.3f}"
            f"{validity:<11.3f}"
            f"{accuracy:<11.3f}"
            f"{missing_runs:<11}"
            f"{technical}"
        )


if __name__ == "__main__":
    main()
