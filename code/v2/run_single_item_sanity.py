from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from generate_task import make_item


VERDICTS = [
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT",
]


def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for verdict in VERDICTS:

        for trial in range(1, args.trials + 1):

            seed = (
                310000
                + VERDICTS.index(verdict) * 1000
                + trial
            )

            rng = random.Random(seed)

            item = make_item(
                index=1,
                version="A",
                verdict=verdict,
                clauses=1,
                distractors=0,
                rng=rng,
            )

            task_data = {
                "benchmark_version": "2.1-sanity",
                "task_family": "evidence_verification",
                "version": "A",
                "workload": "single_item",
                "claims": 1,
                "clauses_per_claim": 1,
                "distractors_per_claim": 0,
                "items": [item],
            }

            stem = (
                f"{verdict.lower()}_"
                f"trial{trial:02d}"
            )

            task = out / f"{stem}.task.json"
            prompt = out / f"{stem}.prompt.txt"
            response = out / f"{stem}.response.txt"
            metadata = out / f"{stem}.metadata.json"
            score = out / f"{stem}.score.json"

            task.write_text(
                json.dumps(task_data, indent=2),
                encoding="utf-8",
            )

            run([
                sys.executable,
                "v2/build_prompt.py",
                "--task", str(task),
                "--output", str(prompt),
            ])

            run([
                sys.executable,
                "v2/run_hf_model.py",
                "--model", args.model,
                "--device", args.device,
                "--prompt", str(prompt),
                "--output", str(response),
                "--metadata-output", str(metadata),
                "--max-new-tokens", "256",
                "--stop-after-claims",
            ])

            run([
                sys.executable,
                "v2/score_response.py",
                "--task", str(task),
                "--response", str(response),
                "--output", str(score),
            ])


if __name__ == "__main__":
    main()
