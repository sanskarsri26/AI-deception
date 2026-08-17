from __future__ import annotations
import argparse
import json
import random
from pathlib import Path


def load_pool(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_allowed_labels(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(data.keys())


def balanced_sample(rows: list[dict], version: str, size: int, seed: int) -> list[dict]:
    candidates = [r for r in rows if r["version"] == version]
    if size > len(candidates):
        raise ValueError(
            f"Requested {size} items, but version {version} has only {len(candidates)}."
        )

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in candidates:
        by_label.setdefault(row["label"], []).append(row)
    for bucket in by_label.values():
        rng.shuffle(bucket)

    selected: list[dict] = []
    labels = sorted(by_label)
    while len(selected) < size:
        made_progress = False
        for label in labels:
            if by_label[label] and len(selected) < size:
                selected.append(by_label[label].pop())
                made_progress = True
        if not made_progress:
            break

    rng.shuffle(selected)
    return selected


def remap_ids(selected: list[dict]) -> list[dict]:
    """Use benchmark-local sequential IDs while preserving the source IDs."""
    remapped = []
    for index, row in enumerate(selected, start=1):
        item = dict(row)
        item["source_item_id"] = row["item_id"]
        item["item_id"] = f"T-{index:03d}"
        remapped.append(item)
    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced benchmark task.")
    parser.add_argument("--pool", default="data/pilot_pool.jsonl")
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--version", choices=["A", "B", "C"], required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=575)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_pool(Path(args.pool))
    selected = balanced_sample(rows, args.version, args.size, args.seed)
    items = remap_ids(selected)

    task = {
        "task_id": f"ticket-{args.version}-{args.size}-{args.seed}",
        "task_family": "support_ticket_classification",
        "version": args.version,
        "size": args.size,
        "seed": args.seed,
        "allowed_labels": load_allowed_labels(Path(args.labels)),
        "items": items,
    }
    Path(args.output).write_text(json.dumps(task, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
