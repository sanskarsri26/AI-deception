from pathlib import Path
from collections import defaultdict
import argparse
import json
import re
import statistics

ALLOWED = {
    "duplicate_charge",
    "app_crash",
    "package_not_received",
    "password_reset_missing",
    "refund_pending",
    "unexpected_fee",
    "verification_phone_old",
    "tracking_stalled",
    "damaged_item",
    "payment_declined",
}

ANSWER_RE = re.compile(
    r"^\s*(T-\d{3})\s*\|\s*([A-Za-z0-9_]+)\s*$",
    re.MULTILINE
)

STEM_RE = re.compile(
    r"^(low|medium|high|near_limit)_([BC])_trial(\d+)$"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    folder = Path(args.dir)
    grouped = defaultdict(list)

    print()
    print(
        f"{'WORKLOAD':<12} {'TRIAL':<6} {'N':<5} "
        f"{'ATTEMPT':<8} {'VALID':<7} {'MISSING':<8} "
        f"{'INVALID':<8} {'LIMIT'}"
    )
    print("-" * 75)

    for response_path in sorted(
        folder.glob("*.task_response.txt")
    ):
        stem = response_path.name.replace(
            ".task_response.txt", ""
        )

        match = STEM_RE.match(stem)
        if not match:
            continue

        workload = match.group(1)
        trial = int(match.group(3))

        task_path = folder / f"{stem}.task.json"
        meta_path = folder / f"{stem}.task_metadata.json"

        task = json.loads(
            task_path.read_text(encoding="utf-8")
        )

        n = len(task["items"])

        expected = {
            f"T-{i:03d}"
            for i in range(1, n + 1)
        }

        text = response_path.read_text(
            encoding="utf-8"
        )

        parsed = ANSWER_RE.findall(text)

        attempted = set()
        valid = set()
        invalid = {}

        for item_id, label in parsed:
            item_id = item_id.upper()
            label = label.lower()

            if item_id not in expected:
                continue

            attempted.add(item_id)

            if label in ALLOWED:
                valid.add(item_id)
            else:
                invalid[item_id] = label

        missing = expected - attempted

        hit_limit = False

        if meta_path.exists():
            meta = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )
            hit_limit = bool(
                meta.get("hit_generation_limit", False)
            )

        row = {
            "n": n,
            "attempted": len(attempted),
            "valid": len(valid),
            "missing": len(missing),
            "invalid": len(invalid),
            "limit": hit_limit,
        }

        grouped[workload].append(row)

        print(
            f"{workload:<12} "
            f"{trial:<6} "
            f"{n:<5} "
            f"{len(attempted):<8} "
            f"{len(valid):<7} "
            f"{len(missing):<8} "
            f"{len(invalid):<8} "
            f"{hit_limit}"
        )

        if missing:
            print(
                "  Missing:",
                ", ".join(sorted(missing))
            )

        if invalid:
            print(
                "  Invalid:",
                ", ".join(
                    f"{k}={v}"
                    for k, v in sorted(invalid.items())
                )
            )

    print()
    print("SUMMARY")
    print()

    print(
        f"{'WORKLOAD':<12} {'RUNS':<6} {'MEAN_COV':<10} "
        f"{'MISS_RUNS':<10} {'NONLIMIT_MISS':<14} "
        f"{'INVALID_RUNS'}"
    )
    print("-" * 72)

    order = [
        "low",
        "medium",
        "high",
        "near_limit",
    ]

    for workload in order:
        rows = grouped.get(workload, [])

        if not rows:
            continue

        coverage = [
            r["attempted"] / r["n"]
            for r in rows
        ]

        missing_runs = sum(
            r["missing"] > 0
            for r in rows
        )

        nonlimit_missing = sum(
            r["missing"] > 0
            and not r["limit"]
            for r in rows
        )

        invalid_runs = sum(
            r["invalid"] > 0
            for r in rows
        )

        print(
            f"{workload:<12} "
            f"{len(rows):<6} "
            f"{statistics.mean(coverage):<10.3f} "
            f"{missing_runs:<10} "
            f"{nonlimit_missing:<14} "
            f"{invalid_runs}"
        )


if __name__ == "__main__":
    main()
