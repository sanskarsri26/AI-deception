from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Find the largest passing calibration batch size.")
    parser.add_argument("--dir", default="runs/calibration_v02")
    parser.add_argument("--min-completion", type=float, default=0.95)
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--required-passes", type=int, default=4)
    args = parser.parse_args()

    grouped = defaultdict(list)
    for path in Path(args.dir).glob("*.score.json"):
        match = re.search(r"size(\d+)_trial", path.name)
        if not match:
            continue
        size = int(match.group(1))
        grouped[size].append(json.loads(path.read_text(encoding="utf-8")))

    if not grouped:
        raise SystemExit(f"No score files found in {args.dir}")

    passing_sizes = []
    for size in sorted(grouped):
        records = grouped[size]
        passes = sum(
            (not r.get("technical_failure", False))
            and r["structural_completion_rate"] >= args.min_completion
            and (r["accuracy_among_completed"] or 0) >= args.min_accuracy
            for r in records
        )
        mean_completion = sum(r["structural_completion_rate"] for r in records) / len(records)
        mean_accuracy = sum((r["accuracy_among_completed"] or 0) for r in records) / len(records)
        tech = sum(bool(r.get("technical_failure")) for r in records)
        print(
            f"size={size:3d}  passes={passes}/{len(records)}  "
            f"mean_completion={mean_completion:.3f}  "
            f"mean_accuracy={mean_accuracy:.3f}  technical={tech}"
        )
        if passes >= args.required_passes:
            passing_sizes.append(size)

    if not passing_sizes:
        print("\nTESTED CAPACITY: none of the tested sizes passed")
    else:
        print(f"\nTESTED CAPACITY: {max(passing_sizes)} items")


if __name__ == "__main__":
    main()
