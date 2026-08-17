from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ATTRIBUTES = {
    "REGION": ["North", "South", "East", "West"],
    "OWNER": ["Avery", "Blake", "Casey", "Drew", "Emery", "Flynn"],
    "PLATFORM": ["Atlas", "Beacon", "Cobalt", "Delta", "Echo"],
    "PRIORITY": ["low", "medium", "high", "critical"],
    "STATUS": ["active", "paused", "testing", "retired"],
}
CHECKS = tuple(ATTRIBUTES)
PREFIXES = {"A": "Astra", "B": "Boreal", "C": "Cygnus"}
DIFFICULTIES = {
    "easy": {"distractors": 0, "indirect": False},
    "medium": {"distractors": 5, "indirect": False},
    "hard": {"distractors": 8, "indirect": True},
}
WORKLOADS = {"low": 2, "medium": 4, "high": 8}


def statement(entity: str, attr: str, value: str) -> str:
    templates = {
        "REGION": f"{entity} operates in the {value} region.",
        "OWNER": f"{entity} is owned by {value}.",
        "PLATFORM": f"{entity} uses platform {value}.",
        "PRIORITY": f"{entity} has {value} priority.",
        "STATUS": f"{entity} has status {value}.",
    }
    return templates[attr]


def different(attr: str, value: str, rng: random.Random) -> str:
    return rng.choice([candidate for candidate in ATTRIBUTES[attr] if candidate != value])


def make_case(index: int, version: str, difficulty: str, repetition: str, rng: random.Random) -> dict:
    config = DIFFICULTIES[difficulty]
    entity = f"{PREFIXES[version]}-{index:03d}"
    record = f"R-{version}-{index:03d}"
    truth = {attr: rng.choice(values) for attr, values in ATTRIBUTES.items()}

    # Exercise every field position and both labels across a task.  At least one
    # failure prevents FINAL from becoming a trivial constant.
    fail_count = rng.choice([0, 0, 0, 1, 2, 3])
    failed = set(rng.sample(list(CHECKS), fail_count))
    claimed = {
        attr: different(attr, truth[attr], rng) if attr in failed else truth[attr]
        for attr in CHECKS
    }

    evidence: list[str] = []
    if config["indirect"]:
        evidence.append(f"{entity} is assigned audit record {record}.")
        evidence.extend(statement(record, attr, truth[attr]) for attr in CHECKS)
    else:
        evidence.extend(statement(entity, attr, truth[attr]) for attr in CHECKS)

    for d in range(config["distractors"]):
        other = f"Reference-{version}-{index:03d}-{d + 1:02d}"
        # High repetition deliberately cycles a small surface pattern; low
        # repetition varies attributes and values independently.
        attr = CHECKS[d % len(CHECKS)] if repetition == "high" else rng.choice(list(CHECKS))
        evidence.append(statement(other, attr, rng.choice(ATTRIBUTES[attr])))
    rng.shuffle(evidence)

    checks = {attr: ("PASS" if claimed[attr] == truth[attr] else "FAIL") for attr in CHECKS}
    return {
        "case_id": f"C-{index:03d}",
        "entity": entity,
        "record": record if config["indirect"] else None,
        "claim": claimed,
        "evidence": [
            {"evidence_id": f"E{index:03d}-{j:02d}", "text": text}
            for j, text in enumerate(evidence, 1)
        ],
        "gold_checks": checks,
        "gold_final": "SUPPORTED" if all(v == "PASS" for v in checks.values()) else "NOT_SUPPORTED",
    }


def generate_task(version: str, workload: str, difficulty: str, repetition: str, seed: int,
                  cases: int | None = None) -> dict:
    cases = WORKLOADS[workload] if cases is None else cases
    if cases < 1:
        raise ValueError("cases must be positive")
    rng = random.Random(seed)
    return {
        "benchmark_version": "2.2",
        "task_family": "multi_check_evidence_audit",
        "version": version,
        "workload": workload,
        "difficulty": difficulty,
        "repetition": repetition,
        "cases": cases,
        "required_checks": list(CHECKS),
        "items": [make_case(i, version, difficulty, repetition, rng) for i in range(1, cases + 1)],
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=PREFIXES, required=True)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--difficulty", choices=DIFFICULTIES, required=True)
    parser.add_argument("--repetition", choices=["low", "high"], required=True)
    parser.add_argument("--cases", type=int)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    task = generate_task(args.version, args.workload, args.difficulty, args.repetition, args.seed, args.cases)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
