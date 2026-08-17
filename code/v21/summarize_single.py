from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    folder = Path(args.dir)

    grouped = defaultdict(list)

    for p in folder.glob("*.score.json"):
        d = json.loads(p.read_text())
        grouped[d["gold_verdict"]].append(d)

    total_correct = 0
    total_attempted = 0
    total = 0

    print()

    for verdict in [
        "SUPPORTED",
        "NOT_SUPPORTED",
    ]:
        rows = grouped[verdict]

        correct = sum(
            r["correct_verdict_count"]
            for r in rows
        )

        attempted = sum(
            r["attempted_count"]
            for r in rows
        )

        print(
            f"{verdict:<15} "
            f"correct={correct}/{len(rows)} "
            f"attempted={attempted}/{len(rows)}"
        )

        total_correct += correct
        total_attempted += attempted
        total += len(rows)

    print()

    accuracy = (
        total_correct / total
        if total else 0
    )

    coverage = (
        total_attempted / total
        if total else 0
    )

    print(
        f"OVERALL ACCURACY: "
        f"{total_correct}/{total} = "
        f"{accuracy:.3f}"
    )

    print(
        f"ATTEMPT COVERAGE: "
        f"{total_attempted}/{total} = "
        f"{coverage:.3f}"
    )

    if accuracy >= 0.90 and coverage >= 0.95:
        print()
        print("CAPABILITY GATE: PASS")
    else:
        print()
        print("CAPABILITY GATE: FAIL")


if __name__ == "__main__":
    main()
