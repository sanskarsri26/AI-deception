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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    folder = Path(args.dir)

    grouped = defaultdict(list)

    print()
    print(
        f"{'SIZE':<6} {'TRIAL':<7} {'ATTEMPT':<9} "
        f"{'VALID':<7} {'MISSING':<8} {'INVALID':<8} "
        f"{'LIMIT':<6}"
    )
    print("-" * 65)

    for path in sorted(folder.glob("size*_trial*.response.txt")):

        m = re.search(
            r"size(\d+)_trial(\d+)",
            path.name
        )

        if not m:
            continue

        size = int(m.group(1))
        trial = int(m.group(2))

        expected = {
            f"T-{i:03d}"
            for i in range(1, size + 1)
        }

        text = path.read_text(
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

        meta_path = folder / (
            path.stem.replace(
                ".response",
                ""
            )
            + ".metadata.json"
        )

        hit_limit = False

        if meta_path.exists():
            meta = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )

            hit_limit = bool(
                meta.get(
                    "hit_generation_limit",
                    False
                )
            )

        record = {
            "attempted": len(attempted),
            "valid": len(valid),
            "missing": len(missing),
            "invalid": len(invalid),
            "hit_limit": hit_limit,
            "missing_ids": sorted(missing),
            "invalid_ids": invalid,
        }

        grouped[size].append(record)

        print(
            f"{size:<6} "
            f"{trial:<7} "
            f"{len(attempted):<9} "
            f"{len(valid):<7} "
            f"{len(missing):<8} "
            f"{len(invalid):<8} "
            f"{str(hit_limit):<6}"
        )

        if missing:
            print(
                "       Missing:",
                ", ".join(sorted(missing))
            )

        if invalid:
            print(
                "       Invalid:",
                ", ".join(
                    f"{k}={v}"
                    for k, v in sorted(
                        invalid.items()
                    )
                )
            )

    print()
    print("SUMMARY")
    print()

    print(
        f"{'SIZE':<6} {'TRIALS':<7} {'MEAN_COV':<10} "
        f"{'MISS_RUNS':<10} {'NONLIMIT_MISS':<14} "
        f"{'INVALID_RUNS':<12}"
    )

    print("-" * 70)

    for size in sorted(grouped):

        rows = grouped[size]

        coverage = [
            r["attempted"] / size
            for r in rows
        ]

        missing_runs = sum(
            r["missing"] > 0
            for r in rows
        )

        nonlimit_missing = sum(
            r["missing"] > 0
            and not r["hit_limit"]
            for r in rows
        )

        invalid_runs = sum(
            r["invalid"] > 0
            for r in rows
        )

        print(
            f"{size:<6} "
            f"{len(rows):<7} "
            f"{statistics.mean(coverage):<10.3f} "
            f"{missing_runs:<10} "
            f"{nonlimit_missing:<14} "
            f"{invalid_runs:<12}"
        )


if __name__ == "__main__":
    main()
