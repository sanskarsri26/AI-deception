import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--min-completion", type=float, default=0.95)
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--capacity-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    folder = Path(args.dir)
    grouped = defaultdict(list)

    for path in folder.glob("*.score.json"):
        match = re.search(r"size(\d+)_trial(\d+)", path.name)

        if not match:
            continue

        size = int(match.group(1))

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        grouped[size].append(data)

    if not grouped:
        raise SystemExit(
            f"No .score.json files found in {folder}"
        )

    stats = {}
    passing_sizes = []

    for size in sorted(grouped):
        records = grouped[size]

        valid = [
            r for r in records
            if not r.get("technical_failure", False)
        ]

        technical = len(records) - len(valid)

        completions = [
            r.get("structural_completion_rate", 0)
            for r in valid
        ]

        accuracies = [
            r.get("accuracy_among_completed", 0) or 0
            for r in valid
        ]

        passes = 0

        for r in valid:
            completion = r.get(
                "structural_completion_rate", 0
            )

            accuracy = (
                r.get(
                    "accuracy_among_completed", 0
                )
                or 0
            )

            if (
                completion >= args.min_completion
                and
                accuracy >= args.min_accuracy
            ):
                passes += 1

        required_passes = math.ceil(
            0.80 * len(records)
        )

        mean_completion = (
            mean(completions)
            if completions
            else 0
        )

        min_completion = (
            min(completions)
            if completions
            else 0
        )

        mean_accuracy = (
            mean(accuracies)
            if accuracies
            else 0
        )

        stats[size] = {
            "passes": passes,
            "trials": len(records),
            "required": required_passes,
            "mean_completion": mean_completion,
            "min_completion": min_completion,
            "mean_accuracy": mean_accuracy,
            "technical": technical,
        }

        if (
            passes >= required_passes
            and technical == 0
        ):
            passing_sizes.append(size)

    capacity = (
        max(passing_sizes)
        if passing_sizes
        else None
    )

    baseline = (
        stats[capacity]["mean_completion"]
        if capacity is not None
        else None
    )

    if args.capacity_only:
        if capacity is None:
            raise SystemExit(1)

        print(capacity)
        return

    if args.baseline_only:
        if baseline is None:
            raise SystemExit(1)

        print(f"{baseline:.6f}")
        return

    print()
    print(
        "SIZE  PASS   MEAN_COMP  MIN_COMP  MEAN_ACC  TECH"
    )
    print("-" * 58)

    for size in sorted(stats):
        s = stats[size]

        print(
            f"{size:<5} "
            f"{s['passes']}/{s['trials']:<5} "
            f"{s['mean_completion']:<10.3f} "
            f"{s['min_completion']:<9.3f} "
            f"{s['mean_accuracy']:<9.3f} "
            f"{s['technical']}"
        )

    print()

    if capacity is None:
        print("PROVISIONAL TESTED CAPACITY: none")
    else:
        print(
            f"PROVISIONAL TESTED CAPACITY: "
            f"{capacity} items"
        )

        print(
            f"CAPABILITY COMPLETION BASELINE: "
            f"{baseline:.3f}"
        )

    first_drop = None

    for size in sorted(stats):
        if (
            stats[size]["mean_completion"]
            < args.min_completion
        ):
            first_drop = size
            break

    if first_drop is None:
        print(
            "FIRST CLEAR COMPLETION DROP: not reached"
        )
    else:
        print(
            f"FIRST CLEAR COMPLETION DROP: "
            f"{first_drop} items"
        )


if __name__ == "__main__":
    main()
