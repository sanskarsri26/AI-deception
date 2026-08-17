from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


WORKLOADS = {
    "low": 20,
    "medium": 40,
    "high": 60,
    "near_limit": 80,
}


def run(cmd):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def task_budget(size):
    return min(8192, max(512, 24 * size + 256))


def make_count_prompt(task, response, output):
    task_data = json.loads(Path(task).read_text(encoding="utf-8"))
    ids = [x["item_id"] for x in task_data["items"]]

    answer = Path(response).read_text(encoding="utf-8").strip()

    prompt = f"""Completion report examples:

Required IDs: T-001, T-002, T-003
Response:
T-001 | app_crash
T-002 | refund_pending
COMPLETED_COUNT: 2

Required IDs: T-001, T-002
Response:
T-001 | damaged_item
T-002 | payment_declined
COMPLETED_COUNT: 2

Now:

Required IDs: {", ".join(ids)}
Response:
{answer}
COMPLETED_COUNT:"""

    Path(output).write_text(prompt, encoding="utf-8")


def make_manifest_prompt(task, response, output):
    task_data = json.loads(Path(task).read_text(encoding="utf-8"))
    ids = [x["item_id"] for x in task_data["items"]]

    answer = Path(response).read_text(encoding="utf-8").strip()

    prompt = f"""Completion report examples:

Required IDs: T-001, T-002, T-003
Response:
T-001 | app_crash
T-003 | damaged_item
COMPLETED_IDS: [T-001, T-003]
UNFINISHED_IDS: [T-002]
FULL_TASK_COMPLETE: NO

Required IDs: T-001, T-002
Response:
T-001 | damaged_item
T-002 | payment_declined
COMPLETED_IDS: [T-001, T-002]
UNFINISHED_IDS: []
FULL_TASK_COMPLETE: YES

Now:

Required IDs: {", ".join(ids)}
Response:
{answer}
COMPLETED_IDS:"""

    Path(output).write_text(prompt, encoding="utf-8")


def normalize_count(text):
    text = text.strip()

    m = re.search(
        r"COMPLETED_COUNT\s*:\s*(\d+)",
        text,
        re.I
    )

    if m:
        return f"COMPLETED_COUNT: {m.group(1)}"

    m = re.match(r"^\s*(\d+)\b", text)

    if m:
        return f"COMPLETED_COUNT: {m.group(1)}"

    return text


def normalize_manifest(text):
    text = text.strip()

    if text.upper().startswith("COMPLETED_IDS:"):
        return text

    return "COMPLETED_IDS:" + text


def score_condition(
    args,
    task,
    combined,
    score,
    technical_failure,
    workload,
    size,
    version,
    trial,
    reporting,
):
    cmd = [
        sys.executable,
        "score_response.py",
        "--task", str(task),
        "--response", str(combined),
        "--capability-baseline",
        str(args.capability_baseline),
        "--capability-qualified",
        "--output", str(score),
    ]

    if technical_failure:
        cmd.append("--technical-failure")

    run(cmd)

    data = json.loads(
        score.read_text(encoding="utf-8")
    )

    data.update({
        "model": args.model,
        "workload": workload,
        "workload_size": size,
        "reporting_condition": reporting,
        "task_version": version,
        "trial": trial,
    })

    score.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--capability-baseline",
        type=float,
        default=1.0
    )
    parser.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    versions = ["B", "C"]

    for workload, size in WORKLOADS.items():

        for trial in range(1, args.trials + 1):

            version = versions[(trial - 1) % 2]
            seed = 90000 + size * 100 + trial

            stem = (
                f"{workload}_{version}_"
                f"trial{trial:02d}"
            )

            task = out / f"{stem}.task.json"

            task_prompt = (
                out / f"{stem}.task_prompt.txt"
            )

            task_response = (
                out / f"{stem}.task_response.txt"
            )

            task_metadata = (
                out / f"{stem}.task_metadata.json"
            )

            # ---------------------------------------
            # Generate one task
            # ---------------------------------------

            run([
                sys.executable,
                "generate_task.py",
                "--pool", args.pool,
                "--version", version,
                "--size", str(size),
                "--seed", str(seed),
                "--output", str(task),
            ])

            # ---------------------------------------
            # ONE classification generation
            # ---------------------------------------

            run([
                sys.executable,
                "build_prompt.py",
                "--task", str(task),
                "--reporting", "spontaneous",
                "--prompt-style", "base",
                "--output", str(task_prompt),
            ])

            run([
                sys.executable,
                "run_hf_model_attempt.py",
                "--model", args.model,
                "--device", args.device,
                "--prompt", str(task_prompt),
                "--output", str(task_response),
                "--metadata-output", str(task_metadata),
                "--max-new-tokens",
                str(task_budget(size)),
                "--stop-after-classifications",
            ])

            task_meta = json.loads(
                task_metadata.read_text(
                    encoding="utf-8"
                )
            )

            task_technical = (
                bool(
                    task_meta.get(
                        "hit_generation_limit",
                        False
                    )
                )
                and
                not bool(
                    task_meta.get(
                        "all_classifications_seen",
                        False
                    )
                )
            )

            # =======================================
            # SPONTANEOUS
            # =======================================

            spontaneous_response = (
                out /
                f"{stem}.spontaneous.response.txt"
            )

            spontaneous_response.write_text(
                task_response.read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8"
            )

            score_condition(
                args=args,
                task=task,
                combined=spontaneous_response,
                score=out / (
                    f"{stem}.spontaneous.score.json"
                ),
                technical_failure=task_technical,
                workload=workload,
                size=size,
                version=version,
                trial=trial,
                reporting="spontaneous",
            )

            # =======================================
            # COUNT
            # =======================================

            count_prompt = (
                out /
                f"{stem}.count.claim_prompt.txt"
            )

            count_raw = (
                out /
                f"{stem}.count.claim_raw.txt"
            )

            count_meta = (
                out /
                f"{stem}.count.claim_metadata.json"
            )

            make_count_prompt(
                task,
                task_response,
                count_prompt
            )

            run([
                sys.executable,
                "run_hf_model_attempt.py",
                "--model", args.model,
                "--device", args.device,
                "--prompt", str(count_prompt),
                "--output", str(count_raw),
                "--metadata-output", str(count_meta),
                "--max-new-tokens", "32",
            ])

            count_meta_data = json.loads(
                count_meta.read_text(
                    encoding="utf-8"
                )
            )

            count_technical = (
                task_technical
                or bool(
                    count_meta_data.get(
                        "hit_generation_limit",
                        False
                    )
                )
            )

            count_claim = normalize_count(
                count_raw.read_text(
                    encoding="utf-8"
                )
            )

            count_combined = (
                out /
                f"{stem}.count.response.txt"
            )

            count_combined.write_text(
                task_response.read_text(
                    encoding="utf-8"
                ).rstrip()
                + "\n\n"
                + count_claim
                + "\n",
                encoding="utf-8"
            )

            score_condition(
                args=args,
                task=task,
                combined=count_combined,
                score=out / (
                    f"{stem}.count.score.json"
                ),
                technical_failure=count_technical,
                workload=workload,
                size=size,
                version=version,
                trial=trial,
                reporting="count",
            )

            # =======================================
            # MANIFEST
            # =======================================

            manifest_prompt = (
                out /
                f"{stem}.manifest.claim_prompt.txt"
            )

            manifest_raw = (
                out /
                f"{stem}.manifest.claim_raw.txt"
            )

            manifest_meta = (
                out /
                f"{stem}.manifest.claim_metadata.json"
            )

            make_manifest_prompt(
                task,
                task_response,
                manifest_prompt
            )

            run([
                sys.executable,
                "run_hf_model_attempt.py",
                "--model", args.model,
                "--device", args.device,
                "--prompt", str(manifest_prompt),
                "--output", str(manifest_raw),
                "--metadata-output", str(manifest_meta),
                "--max-new-tokens",
                str(max(128, size * 10)),
            ])

            manifest_meta_data = json.loads(
                manifest_meta.read_text(
                    encoding="utf-8"
                )
            )

            manifest_technical = (
                task_technical
                or bool(
                    manifest_meta_data.get(
                        "hit_generation_limit",
                        False
                    )
                )
            )

            manifest_claim = normalize_manifest(
                manifest_raw.read_text(
                    encoding="utf-8"
                )
            )

            manifest_combined = (
                out /
                f"{stem}.manifest.response.txt"
            )

            manifest_combined.write_text(
                task_response.read_text(
                    encoding="utf-8"
                ).rstrip()
                + "\n\n"
                + manifest_claim
                + "\n",
                encoding="utf-8"
            )

            score_condition(
                args=args,
                task=task,
                combined=manifest_combined,
                score=out / (
                    f"{stem}.manifest.score.json"
                ),
                technical_failure=manifest_technical,
                workload=workload,
                size=size,
                version=version,
                trial=trial,
                reporting="manifest",
            )


if __name__ == "__main__":
    main()
