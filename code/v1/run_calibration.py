from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def calibration_token_budget(size: int) -> int:
    # Each classification line is short. This leaves a generous safety margin
    # without giving a base model thousands of unnecessary continuation tokens.
    return max(160, 32 * size + 128)


def main():
    parser = argparse.ArgumentParser(description="Capability calibration for a Hugging Face model.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--sizes", default="1,5,10,20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="0 uses an automatic workload-based budget.",
    )
    parser.add_argument("--prompt-style", choices=["base", "instruct"], default="base")
    parser.add_argument("--out-dir", default="runs/calibration_v03")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    for size in sizes:
        for trial in range(1, args.trials + 1):
            seed = 575 + size * 100 + trial
            stem = f"size{size:03d}_trial{trial:02d}"
            task = out / f"{stem}.task.json"
            prompt = out / f"{stem}.prompt.txt"
            response = out / f"{stem}.response.txt"
            score = out / f"{stem}.score.json"
            meta = out / f"{stem}.metadata.json"
            max_tokens = args.max_new_tokens or calibration_token_budget(size)

            run([
                sys.executable,
                "generate_task.py",
                "--version", "A",
                "--size", str(size),
                "--seed", str(seed),
                "--output", str(task),
            ])
            run([
                sys.executable,
                "build_prompt.py",
                "--task", str(task),
                "--reporting", "spontaneous",
                "--prompt-style", args.prompt_style,
                "--output", str(prompt),
            ])
            run([
                sys.executable,
                "run_hf_model.py",
                "--model", args.model,
                "--device", args.device,
                "--prompt", str(prompt),
                "--output", str(response),
                "--metadata-output", str(meta),
                "--max-new-tokens", str(max_tokens),
                "--stop-after-classifications",
            ])

            metadata = json.loads(meta.read_text(encoding="utf-8"))
            extra = ["--technical-failure"] if metadata.get("hit_generation_limit") else []
            run([
                sys.executable,
                "score_response.py",
                "--task", str(task),
                "--response", str(response),
                *extra,
                "--output", str(score),
            ])


if __name__ == "__main__":
    main()
