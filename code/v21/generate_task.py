from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ATTRIBUTES = {
    "region": ["North", "South", "East", "West"],
    "status": ["active", "paused", "testing", "retired"],
    "owner": ["Avery", "Blake", "Casey", "Drew"],
    "platform": ["Atlas", "Beacon", "Cobalt", "Delta"],
    "priority": ["low", "medium", "high", "critical"],
}

PREFIXES = {
    "A": "Astra",
    "B": "Boreal",
    "C": "Cygnus",
}


def evidence_sentence(entity, attr, value):
    if attr == "region":
        return f"{entity} is in the {value} region."
    if attr == "status":
        return f"{entity} has status {value}."
    if attr == "owner":
        return f"{entity} is owned by {value}."
    if attr == "platform":
        return f"{entity} uses platform {value}."
    if attr == "priority":
        return f"{entity} has {value} priority."
    raise ValueError(attr)


def claim_fragment(attr, value):
    if attr == "region":
        return f"is in the {value} region"
    if attr == "status":
        return f"has status {value}"
    if attr == "owner":
        return f"is owned by {value}"
    if attr == "platform":
        return f"uses platform {value}"
    if attr == "priority":
        return f"has {value} priority"
    raise ValueError(attr)


def alternate(attr, current, rng):
    choices = [x for x in ATTRIBUTES[attr] if x != current]
    return rng.choice(choices)


def make_item(index, version, verdict, facts, distractors, rng):
    entity = f"{PREFIXES[version]}-{index:03d}"

    world = {
        attr: rng.choice(values)
        for attr, values in ATTRIBUTES.items()
    }

    selected = rng.sample(
        list(ATTRIBUTES.keys()),
        facts,
    )

    claim_values = {
        attr: world[attr]
        for attr in selected
    }

    contradicted_attr = None

    if verdict == "NOT_SUPPORTED":
        contradicted_attr = rng.choice(selected)
        claim_values[contradicted_attr] = alternate(
            contradicted_attr,
            world[contradicted_attr],
            rng,
        )

    raw_evidence = []

    # Evidence always explicitly states the true value
    # for every fact in the claim.
    for attr in selected:
        raw_evidence.append(
            evidence_sentence(
                entity,
                attr,
                world[attr],
            )
        )

    # Distractors refer to DIFFERENT entities.
    for d in range(distractors):
        other = f"Reference-{version}-{index:03d}-{d+1:02d}"
        attr = rng.choice(list(ATTRIBUTES.keys()))
        value = rng.choice(ATTRIBUTES[attr])

        raw_evidence.append(
            evidence_sentence(
                other,
                attr,
                value,
            )
        )

    rng.shuffle(raw_evidence)

    evidence = [
        {
            "evidence_id": f"E{index:03d}-{i:02d}",
            "text": text,
        }
        for i, text in enumerate(raw_evidence, start=1)
    ]

    fragments = [
        claim_fragment(attr, claim_values[attr])
        for attr in selected
    ]

    if len(fragments) == 1:
        body = fragments[0]
    else:
        body = ", ".join(fragments[:-1]) + ", and " + fragments[-1]

    claim = f"{entity} {body}."

    return {
        "claim_id": f"C-{index:03d}",
        "entity": entity,
        "claim": claim,
        "evidence": evidence,
        "gold_verdict": verdict,
        "contradicted_attribute": contradicted_attr,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        choices=["A", "B", "C"],
        required=True,
    )

    parser.add_argument(
        "--claims",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--facts",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--distractors",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--forced-verdict",
        choices=["SUPPORTED", "NOT_SUPPORTED"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    if args.facts < 1 or args.facts > len(ATTRIBUTES):
        raise SystemExit("--facts must be between 1 and 5")

    rng = random.Random(args.seed)

    if args.forced_verdict:
        verdicts = [
            args.forced_verdict
            for _ in range(args.claims)
        ]
    else:
        verdicts = [
            "SUPPORTED" if i % 2 == 0 else "NOT_SUPPORTED"
            for i in range(args.claims)
        ]
        rng.shuffle(verdicts)

    items = []

    for i in range(1, args.claims + 1):
        items.append(
            make_item(
                index=i,
                version=args.version,
                verdict=verdicts[i - 1],
                facts=args.facts,
                distractors=args.distractors,
                rng=rng,
            )
        )

    task = {
        "benchmark_version": "2.1",
        "task_family": "binary_evidence_verification",
        "version": args.version,
        "claims": args.claims,
        "facts_per_claim": args.facts,
        "distractors_per_claim": args.distractors,
        "items": items,
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(task, indent=2),
        encoding="utf-8",
    )

    print(
        f"Created {path} "
        f"claims={args.claims} "
        f"facts={args.facts} "
        f"distractors={args.distractors}"
    )


if __name__ == "__main__":
    main()
