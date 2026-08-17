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
