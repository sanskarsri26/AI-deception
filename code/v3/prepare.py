from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from v3.sifo import DEFAULT_SEED, SIFO_COMMIT, SIFO_REPOSITORY, instruction_count, prepare_splits


EXPECTED_HASHES = {
    "math": "2f767554407389d72179197334a2441d8f0554795f90ea7dae02746ac687ef1b",
    "qa": "128be4f6fb74d71219b07dc5a324cf723f571ad48217e7d7deab3205a851c761",
    "security": "1fa3bc917a6d01b791c0954c48396375d2610ff094cc7eb971de7f2df085b372",
    "text_modification": "d13c5288bf13974d06fb61d5c5a17d6cff2714be51cea80d1cf76931f29d0d17",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare disjoint SIFo V3 manifests without copying upstream data")
    parser.add_argument("--sifo-root", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("v3/manifests"), type=Path)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--calibration-fraction", default=0.25, type=float)
    parser.add_argument("--allow-upstream-mismatch", action="store_true")
    args = parser.parse_args()
    dataset_dir = args.sifo_root / "sifo_datasets"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit(args.sifo_root)
    if commit != SIFO_COMMIT and not args.allow_upstream_mismatch:
        raise SystemExit(f"SIFo commit mismatch: expected {SIFO_COMMIT}, found {commit}")
    provenance = {
        "upstream_repository": SIFO_REPOSITORY,
        "upstream_commit": commit,
        "expected_upstream_commit": SIFO_COMMIT,
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "adaptations": [
            "The normal condition uses each published task at its original natural instruction depth.",
            "For Instruct models, unchanged published task content is placed in the tokenizer's official chat template.",
            "E1/E2/E3 add only generic carefulness and verification text to the system message.",
            "The original prompt and condition-specific adapted prompt are saved in every record.",
            "A structured self-audit is run from the frozen normal response in an independent branch.",
            "A neutral completion-honesty follow-up is appended only after the original response is frozen.",
        ],
        "families": {},
    }
    split_ids = {"calibration": {}, "main": {}}
    for family, expected_hash in EXPECTED_HASHES.items():
        path = dataset_dir / f"{family}.jsonl"
        actual_hash = file_hash(path)
        if actual_hash != expected_hash and not args.allow_upstream_mismatch:
            raise SystemExit(f"Dataset hash mismatch for {family}: expected {expected_hash}, found {actual_hash}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        splits = prepare_splits(rows, family, args.seed, args.calibration_fraction)
        provenance["families"][family] = {
            "relative_path": f"sifo_datasets/{family}.jsonl",
            "dataset_sha256": actual_hash,
            "example_count": len(rows),
            "instruction_depths": sorted({instruction_count(row) for row in rows}),
        }
        for split, selected in splits.items():
            split_ids[split][family] = [row["id"] for row in selected]
    calibration = {(family, value) for family, ids in split_ids["calibration"].items() for value in ids}
    main_ids = {(family, value) for family, ids in split_ids["main"].items() for value in ids}
    provenance["zero_source_id_overlap"] = not bool(calibration & main_ids)
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for split, families in split_ids.items():
        (args.out_dir / f"{split}_ids.json").write_text(json.dumps(families, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "counts": {
        split: {family: len(ids) for family, ids in families.items()}
        for split, families in split_ids.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
