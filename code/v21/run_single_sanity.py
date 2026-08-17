from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


VERDICTS = [
    "SUPPORTED",
    "NOT_SUPPORTED",
]


def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for verdict_index, verdict in enumerate(VERDICTS):

        for trial in range(1, args.trials + 1):

            seed = (
                410000
                + verdict_index * 10000
                + trial
            )

            stem = (
                f"{verdict.lower()}_"
                f"trial{trial:02d}"
            )

            task = out / f"{stem}.task.json"
            prompt = out / f"{stem}.prompt.txt"
            response = out / f"{stem}.response.txt"
            metadata = out / f"{stem}.metadata.json"
            score = out / f"{stem}.score.json"

            run([
                sys.executable,
                "v21/generate_task.py",
                "--version", "A",
                "--claims", "1",
                "--facts", "1",
                "--distractors", "0",
                "--forced-verdict", verdict,
                "--seed", str(seed),
                "--output", str(task),
            ])

            run([
                sys.executable,
                "v21/build_prompt.py",
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
                "--max-new-tokens", "64",
                "--stop-after-claims",
            ])

            run([
                sys.executable,
                "v21/score_response.py",
                "--task", str(task),
                "--response", str(response),
                "--output", str(score),
            ])

            data = json.loads(
                score.read_text(encoding="utf-8")
            )

            data.update({
                "model": args.model,
                "gold_verdict": verdict,
                "trial": trial,
            })

            score.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
