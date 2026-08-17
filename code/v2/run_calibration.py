from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


WORKLOADS = [
    "low",
    "medium",
    "high",
    "near_limit",
]


def run(cmd):
    print(
        "+",
        " ".join(str(x) for x in cmd),
        flush=True,
    )

    subprocess.run(
        cmd,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--claims",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    args = parser.parse_args()

    out = Path(args.out_dir)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    for workload_index, workload in enumerate(
        WORKLOADS
    ):
        for trial in range(
            1,
            args.trials + 1,
        ):
            seed = (
                210000
                + workload_index * 10000
                + trial
            )

            stem = (
                f"{workload}_"
                f"A_trial{trial:02d}"
            )

            task = out / (
                f"{stem}.task.json"
            )

            prompt = out / (
                f"{stem}.prompt.txt"
            )

            response = out / (
                f"{stem}.response.txt"
            )

            metadata = out / (
                f"{stem}.metadata.json"
            )

            score = out / (
                f"{stem}.score.json"
            )

            run([
                sys.executable,
                "v2/generate_task.py",
                "--version",
                "A",
                "--workload",
                workload,
                "--claims",
                str(args.claims),
                "--seed",
                str(seed),
                "--output",
                str(task),
            ])

            run([
                sys.executable,
                "v2/build_prompt.py",
                "--task",
                str(task),
                "--output",
                str(prompt),
            ])

            run([
                sys.executable,
                "v2/run_hf_model.py",
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
                "2048",
                "--stop-after-claims",
            ])

            meta = json.loads(
                metadata.read_text(
                    encoding="utf-8"
                )
            )

            technical = (
                bool(
                    meta.get(
                        "hit_generation_limit",
                        False,
                    )
                )
                and not bool(
                    meta.get(
                        "all_claims_seen",
                        False,
                    )
                )
            )

            cmd = [
                sys.executable,
                "v2/score_response.py",
                "--task",
                str(task),
                "--response",
                str(response),
                "--output",
                str(score),
            ]

            if technical:
                cmd.append(
                    "--technical-failure"
                )

            run(cmd)

            data = json.loads(
                score.read_text(
                    encoding="utf-8"
                )
            )

            data.update({
                "model": args.model,
                "workload": workload,
                "trial": trial,
                "version": "A",
            })

            score.write_text(
                json.dumps(
                    data,
                    indent=2,
                ),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
