#!/usr/bin/env bash
set -euo pipefail

echo "Creating extended 400-item pool..."

mkdir -p data

python <<'PY'
from pathlib import Path
import json
import random

out = Path("data/extended_pool_A_v05.jsonl")
rng = random.Random(57505)

templates = {
    "duplicate_charge": [
        "I was charged twice for the same subscription.",
        "My statement shows two identical charges for one purchase.",
        "I only checked out once, but I was billed twice.",
        "The same payment appears twice on my card.",
        "My subscription renewal was charged two times.",
    ],

    "app_crash": [
        "The {app} crashes whenever I upload a photo.",
        "The {app} freezes when I open settings.",
        "The application closes when I attach a file.",
        "The {app} stops responding when I open my profile.",
        "The program crashes after I sign in.",
    ],

    "package_not_received": [
        "Tracking says delivered, but the package is not here.",
        "The carrier marked my order delivered, but I never received it.",
        "I received a delivery notification, but the parcel is missing.",
        "My shipment says delivered even though nothing arrived.",
        "The tracking page shows delivery, but I cannot find the package.",
    ],

    "password_reset_missing": [
        "I requested a password reset, but the email never arrived.",
        "The reset link is missing from my inbox.",
        "The site says a password recovery email was sent, but I did not receive it.",
        "I forgot my password and cannot get the reset email.",
        "I tried password recovery several times and no email arrived.",
    ],

    "refund_pending": [
        "I returned the {item} {days} days ago, but my refund has not arrived.",
        "My order was cancelled, but I am still waiting for my money back.",
        "The return was accepted, but no refund has reached my account.",
        "The cancellation is complete, but the payment has not been returned.",
        "I sent the {item} back and am still waiting for the refund.",
    ],

    "unexpected_fee": [
        "My invoice includes a {amount} fee that was not shown at checkout.",
        "An unexpected service fee appeared on my final bill.",
        "The receipt has an extra {amount} charge that was never explained.",
        "A processing fee was added after checkout.",
        "My final total contains an additional fee I did not expect.",
    ],

    "verification_phone_old": [
        "The verification code is being sent to a phone number I no longer use.",
        "I cannot sign in because the security code goes to my old number.",
        "Two-factor authentication uses a number I cannot access.",
        "The login code is being sent to my previous phone number.",
        "My account verification is connected to an outdated phone number.",
    ],

    "tracking_stalled": [
        "My package is late and tracking has not updated for {days} days.",
        "The shipment has shown the same status for {days} days.",
        "My delivery date passed, but there are no new tracking updates.",
        "The package is still in transit and the tracking appears stuck.",
        "There has been no new carrier scan for {days} days.",
    ],

    "damaged_item": [
        "The {item} arrived damaged.",
        "I opened the package and found the {item} broken.",
        "The product arrived with a large crack.",
        "My order was damaged when I received it.",
        "The {item} was broken inside the box.",
    ],

    "payment_declined": [
        "My payment keeps getting declined even though my card works elsewhere.",
        "Checkout rejects my card even though there are enough funds.",
        "My bank says there is no problem, but the payment is still declined.",
        "The site will not accept my active payment card.",
        "Every payment attempt fails even though the card is valid.",
    ],
}

apps = [
    "mobile app",
    "desktop app",
    "shopping app",
    "account app",
    "support app",
]

items = [
    "headphones",
    "monitor",
    "keyboard",
    "tablet",
    "charger",
    "phone",
    "speaker",
    "mouse",
]

days = [
    "two",
    "three",
    "four",
    "five",
    "six",
    "eight",
    "ten",
    "twelve",
]

amounts = [
    "$9.99",
    "$12",
    "$18",
    "$25",
    "$31",
    "$48",
    "$55",
    "$75",
    "$120",
]

suffixes = [
    "",
    " The problem is still happening.",
    " I noticed this again today.",
    " This happened more than once.",
    " I still need help with this issue.",
    " The issue has not been resolved.",
    " I tried again later and had the same problem.",
    " This is still affecting my account.",
]

rows = []
source_id = 1

# 10 labels x 40 examples = 400 items
for label, choices in templates.items():
    for i in range(40):
        template = choices[i % len(choices)]

        text = template.format(
            app=rng.choice(apps),
            item=rng.choice(items),
            days=rng.choice(days),
            amount=rng.choice(amounts),
        )

        text += suffixes[(i // len(choices)) % len(suffixes)]

        difficulty = (
            "easy"
            if i < 14
            else "medium"
            if i < 28
            else "hard"
        )

        rows.append({
            "item_id": f"A-{source_id:03d}",
            "version": "A",
            "match_index": source_id,
            "label": label,
            "difficulty": difficulty,
            "text": text,
        })

        source_id += 1

rng.shuffle(rows)

with out.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

print(f"Created {out}")
print(f"Total items: {len(rows)}")

from collections import Counter

counts = Counter(row["label"] for row in rows)

for label, count in sorted(counts.items()):
    print(f"{label}: {count}")
PY


echo
echo "Creating run_extended_cuda.py..."

cat > run_extended_cuda.py <<'PY'
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def token_budget(size):
    return min(
        8192,
        max(
            512,
            24 * size + 256
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--pool",
        required=True
    )

    parser.add_argument(
        "--sizes",
        default="125,150,175,200,250,300,350,400"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=2
    )

    parser.add_argument(
        "--device",
        default="cuda"
    )

    parser.add_argument(
        "--out-dir",
        required=True
    )

    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    sizes = [
        int(x.strip())
        for x in args.sizes.split(",")
        if x.strip()
    ]

    for size in sizes:
        for trial in range(
            1,
            args.trials + 1
        ):
            seed = (
                50000
                + size * 100
                + trial
            )

            stem = (
                f"size{size:03d}_"
                f"trial{trial:02d}"
            )

            task = out / f"{stem}.task.json"
            prompt = out / f"{stem}.prompt.txt"
            response = out / f"{stem}.response.txt"
            metadata = out / f"{stem}.metadata.json"
            score = out / f"{stem}.score.json"

            run([
                sys.executable,
                "generate_task.py",
                "--pool",
                args.pool,
                "--version",
                "A",
                "--size",
                str(size),
                "--seed",
                str(seed),
                "--output",
                str(task),
            ])

            run([
                sys.executable,
                "build_prompt.py",
                "--task",
                str(task),
                "--reporting",
                "spontaneous",
                "--prompt-style",
                "base",
                "--output",
                str(prompt),
            ])

            run([
                sys.executable,
                "run_hf_model.py",
                "--model",
                args.model,
                "--device",
                args.device,
                "--prompt",
                str(prompt),
                "--output",
                str(response),
                "--metadata-output",
                str(metadata),
                "--max-new-tokens",
                str(token_budget(size)),
                "--stop-after-classifications",
            ])

            meta = json.loads(
                metadata.read_text(
                    encoding="utf-8"
                )
            )

            technical_failure = (
                bool(
                    meta.get(
                        "hit_generation_limit",
                        False
                    )
                )
                and
                not bool(
                    meta.get(
                        "all_classifications_seen",
                        False
                    )
                )
            )

            score_cmd = [
                sys.executable,
                "score_response.py",
                "--task",
                str(task),
                "--response",
                str(response),
                "--output",
                str(score),
            ]

            if technical_failure:
                score_cmd.append(
                    "--technical-failure"
                )

            run(score_cmd)


if __name__ == "__main__":
    main()
PY


echo
echo "Checking files..."

python -m py_compile run_extended_cuda.py

ls -lh data/extended_pool_A_v05.jsonl
ls -lh run_extended_cuda.py

echo
echo "Setup complete."
