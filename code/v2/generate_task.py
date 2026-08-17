from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


WORKLOADS = {
    "low": {
        "clauses": 2,
        "distractors": 2,
    },
    "medium": {
        "clauses": 3,
        "distractors": 4,
    },
    "high": {
        "clauses": 4,
        "distractors": 6,
    },
    "near_limit": {
        "clauses": 5,
        "distractors": 8,
    },
}


ATTRIBUTES = {
    "region": [
        "North",
        "South",
        "East",
        "West",
    ],
    "status": [
        "active",
        "paused",
        "testing",
        "retired",
    ],
    "owner": [
        "Avery",
        "Blake",
        "Casey",
        "Drew",
        "Emery",
        "Flynn",
    ],
    "platform": [
        "Atlas",
        "Beacon",
        "Cobalt",
        "Delta",
        "Echo",
    ],
    "priority": [
        "low",
        "medium",
        "high",
        "critical",
    ],
    "launch_month": [
        "January",
        "March",
        "May",
        "July",
        "September",
        "November",
    ],
}


PREFIXES = {
    "A": "Astra",
    "B": "Boreal",
    "C": "Cygnus",
}


def evidence_sentence(entity, attr, value):
    if attr == "region":
        return f"{entity} is assigned to the {value} region."

    if attr == "status":
        return f"The current status of {entity} is {value}."

    if attr == "owner":
        return f"{entity} is owned by {value}."

    if attr == "platform":
        return f"{entity} uses the {value} platform."

    if attr == "priority":
        return f"{entity} has {value} priority."

    if attr == "launch_month":
        return f"{entity} launched in {value}."

    raise ValueError(attr)


def claim_fragment(attr, value):
    if attr == "region":
        return f"its region is {value}"

    if attr == "status":
        return f"its status is {value}"

    if attr == "owner":
        return f"its owner is {value}"

    if attr == "platform":
        return f"it uses the {value} platform"

    if attr == "priority":
        return f"its priority is {value}"

    if attr == "launch_month":
        return f"it launched in {value}"

    raise ValueError(attr)


def alternate_value(attr, current, rng):
    choices = [
        x for x in ATTRIBUTES[attr]
        if x != current
    ]

    return rng.choice(choices)


def make_item(
    index,
    version,
    verdict,
    clauses,
    distractors,
    rng,
):
    entity = (
        f"{PREFIXES[version]}-"
        f"{index:03d}"
    )

    world = {
        attr: rng.choice(values)
        for attr, values in ATTRIBUTES.items()
    }

    selected_attrs = rng.sample(
        list(ATTRIBUTES),
        clauses,
    )

    claim_values = {
        attr: world[attr]
        for attr in selected_attrs
    }

    special_attr = None

    if verdict == "CONTRADICTED":
        special_attr = rng.choice(
            selected_attrs
        )

        claim_values[special_attr] = (
            alternate_value(
                special_attr,
                world[special_attr],
                rng,
            )
        )

    elif verdict == "INSUFFICIENT":
        special_attr = rng.choice(
            selected_attrs
        )

    raw_evidence = []

    for attr in selected_attrs:

        if (
            verdict == "INSUFFICIENT"
            and attr == special_attr
        ):
            continue

        raw_evidence.append({
            "text": evidence_sentence(
                entity,
                attr,
                world[attr],
            ),
            "decisive": True,
        })

    # Add unrelated evidence as distractors.
    for d in range(distractors):

        other = (
            f"Reference-{version}-"
            f"{index:03d}-{d + 1:02d}"
        )

        attr = rng.choice(
            list(ATTRIBUTES)
        )

        value = rng.choice(
            ATTRIBUTES[attr]
        )

        raw_evidence.append({
            "text": evidence_sentence(
                other,
                attr,
                value,
            ),
            "decisive": False,
        })

    rng.shuffle(raw_evidence)

    evidence = []
    gold_evidence_ids = []

    for j, entry in enumerate(
        raw_evidence,
        start=1,
    ):
        evidence_id = (
            f"E{index:03d}-{j:02d}"
        )

        evidence.append({
            "evidence_id": evidence_id,
            "text": entry["text"],
        })

        if entry["decisive"]:
            gold_evidence_ids.append(
                evidence_id
            )

    fragments = [
        claim_fragment(
            attr,
            claim_values[attr],
        )
        for attr in selected_attrs
    ]

    if len(fragments) == 1:
        body = fragments[0]
    else:
        body = (
            ", ".join(fragments[:-1])
            + ", and "
            + fragments[-1]
        )

    claim = (
        f"For {entity}, {body}."
    )

    return {
        "claim_id": f"C-{index:03d}",
        "entity": entity,
        "claim": claim,
        "evidence": evidence,
        "gold_verdict": verdict,
        "gold_evidence_ids": sorted(
            gold_evidence_ids
        ),
        "special_attribute": special_attr,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        required=True,
        choices=["A", "B", "C"],
    )

    parser.add_argument(
        "--workload",
        required=True,
        choices=list(WORKLOADS),
    )

    parser.add_argument(
        "--claims",
        type=int,
        default=24,
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

    rng = random.Random(args.seed)

    config = WORKLOADS[
        args.workload
    ]

    verdicts = [
        [
            "SUPPORTED",
            "CONTRADICTED",
            "INSUFFICIENT",
        ][i % 3]
        for i in range(args.claims)
    ]

    rng.shuffle(verdicts)

    items = []

    for i in range(
        1,
        args.claims + 1,
    ):
        items.append(
            make_item(
                index=i,
                version=args.version,
                verdict=verdicts[i - 1],
                clauses=config[
                    "clauses"
                ],
                distractors=config[
                    "distractors"
                ],
                rng=rng,
            )
        )

    task = {
        "benchmark_version": 2,
        "task_family": (
            "evidence_verification"
        ),
        "version": args.version,
        "workload": args.workload,
        "claims": args.claims,
        "clauses_per_claim": (
            config["clauses"]
        ),
        "distractors_per_claim": (
            config["distractors"]
        ),
        "items": items,
    }

    out = Path(args.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            task,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Created {out}"
    )

    print(
        f"Version={args.version} "
        f"workload={args.workload} "
        f"claims={args.claims}"
    )


if __name__ == "__main__":
    main()
