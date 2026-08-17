"""CPU-only structural/provenance audit of the two locally inspected upstreams."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def audit_followbench(root: Path) -> dict[str, Any]:
    files = sorted((root / "data").glob("*_constraints.json"))
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    file_details = {}
    for path in files:
        rows = json.loads(path.read_text(encoding="utf-8"))
        levels = Counter(row["level"] for row in rows)
        family = path.stem.removesuffix("_constraints")
        for row in rows:
            groups[(family, row["example_id"])].append(row["level"])
        file_details[path.name] = {
            "sha256": sha256(path), "rows": len(rows),
            "levels": dict(sorted(levels.items())),
        }
    malformed_paths = {
        f"{category}:{example_id}": sorted(levels)
        for (category, example_id), levels in groups.items()
        if sorted(levels) not in ([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    }
    return {
        "repository": "https://github.com/YJiangcm/FollowBench", "commit": commit(root),
        "files": file_details, "evolution_paths": len(groups),
        "malformed_evolution_paths": malformed_paths,
        "official_scorers": {
            "rule_based_sha256": sha256(root / "code" / "rule_based_evaluation.py"),
            "llm_based_sha256": sha256(root / "code" / "gpt4_based_evaluation.py"),
        },
        "requirement_level_scores_available": True,
        "attempt_vs_omission_identifiable": False,
    }


def audit_sifo(root: Path) -> dict[str, Any]:
    files = sorted((root / "sifo_datasets").glob("*.jsonl"))
    details = {}
    all_ids = []
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        depths = Counter(sum(bool(row.get(f"instruction_{i}")) for i in range(1, 7)) for row in rows)
        ids = [row["id"] for row in rows]
        all_ids.extend(ids)
        details[path.name] = {
            "sha256": sha256(path), "rows": len(rows), "unique_ids": len(set(ids)),
            "depths": dict(sorted(depths.items())),
        }
    return {
        "repository": "https://github.com/shin-ee-chen/SIFo", "commit": commit(root),
        "files": details, "ids_unique_across_families": len(all_ids) == len(set(all_ids)),
        "official_metrics_sha256": sha256(root / "metrics.py"),
        "requirement_level_scores_available": True,
        "attempt_vs_omission_identifiable_for_valid_json": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--followbench-root", required=True, type=Path)
    parser.add_argument("--sifo-root", required=True, type=Path)
    parser.add_argument("--output", default=Path("v3/manifests/benchmark_audit.json"), type=Path)
    args = parser.parse_args()
    result = {
        "FollowBench": audit_followbench(args.followbench_root),
        "SIFo": audit_sifo(args.sifo_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
