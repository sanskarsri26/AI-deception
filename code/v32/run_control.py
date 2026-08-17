"""V3.2 held-out control run.

Adds held-out carefulness elicitors (E4/E5), neutral placebos (P1/P2) and a
normal replicate (N2) on the *same* 598 SIFo main-split tasks as the V3.1 main
run, using identical model/decoding/template settings. Writes only to its own
output directory; V3.1 artifacts are read-only inputs.

Usage:
  python -m v32.run_control --model Qwen/Qwen2.5-3B-Instruct \
      --sifo-root /scratch/ssriva94/SIFo --out runs/v32_control_3b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v3.run_sifo import (
    Generator, PROTOCOL_VERSION, SYSTEM_PROMPT, append_jsonl, load_rows,
    read_json, read_jsonl, write_json,
)
from v3.sifo import DEFAULT_SEED, build_official_prompt, instruction_count
from v32.scoring import SCORER_VERSION, score_sifo_response_v32

CONTROL_VERSION = "3.2.0"

# Held-out carefulness elicitors. Same spirit as E1/E2/E3, no task-specific help.
# These MUST NOT be used for capability qualification.
HELD_OUT_ELICITORS = {
    "E4": "Follow each requirement in the request precisely. Double-check your answer "
          "against every stated requirement before you finish.",
    "E5": "Take care to address every part of the request. Re-read the requirements and "
          "confirm each one is handled before you answer.",
}

# Neutral placebos: comparable length and imperative register, but no carefulness,
# verification, completeness or output-format content of any kind.
PLACEBOS = {
    "P1": "This conversation is part of a routine evaluation batch. Session identifiers "
          "are recorded for internal auditing purposes.",
    "P2": "This request was routed through the general-purpose serving endpoint. The "
          "account locale is set to English (United States).",
}

CONDITIONS: dict[str, str | None] = {"N2": None, **HELD_OUT_ELICITORS, **PLACEBOS}


def messages_for(prompt: str, suffix: str | None) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT if suffix is None else SYSTEM_PROMPT + "\n" + suffix
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sifo-root", required=True, type=Path)
    parser.add_argument("--manifests", default=Path("v3/manifests"), type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reference-run", default=Path("runs/v31_qwen25_3b_instruct"), type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--max-new-tokens", default=1200, type=int)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--limit", type=int, help="per-family smoke limit; results non-final")
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")

    reference = read_json(args.reference_run / "run_config_main.json")
    if reference["model"] != args.model:
        raise SystemExit(f"model must match the V3.1 run being controlled ({reference['model']})")
    if reference["max_new_tokens"] != args.max_new_tokens:
        raise SystemExit("max_new_tokens must match the V3.1 main run")

    args.out.mkdir(parents=True, exist_ok=True)
    generator = Generator(args.model, args.device)
    for key, current in (("model_revision", generator.model_revision),
                         ("chat_template_sha256", generator.template_sha256)):
        if reference.get(key) != current:
            raise SystemExit(f"{key} differs from the V3.1 main run; pin the same snapshot")

    write_json(args.out / "run_config_control.json", {
        "control_version": CONTROL_VERSION, "protocol_version": PROTOCOL_VERSION,
        "scorer_version": SCORER_VERSION, "model": args.model,
        "model_revision": generator.model_revision,
        "tokenizer_revision": generator.tokenizer_revision,
        "chat_template_sha256": generator.template_sha256,
        "seed": DEFAULT_SEED, "temperature": 0.0, "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens, "system_prompt": SYSTEM_PROMPT,
        "conditions": {c: CONDITIONS[c] for c in conditions},
        "held_out_from_capability_qualification": sorted(HELD_OUT_ELICITORS),
        "reference_run": str(args.reference_run), "limit": args.limit,
    })

    rows_by_family = load_rows(args.sifo_root, args.manifests, "main")
    checkpoint = args.out / "control_records.jsonl"
    done = {(r["family"], r["original_example_id"], r["condition"]) for r in read_jsonl(checkpoint)}

    total = sum(len(rows[:args.limit]) for rows in rows_by_family.values()) * len(conditions)
    index = 0
    for family, rows in sorted(rows_by_family.items()):
        for row in rows[:args.limit]:
            depth = instruction_count(row)
            prompt = build_official_prompt(row)
            for condition in conditions:
                index += 1
                if (family, row["id"], condition) in done:
                    continue
                messages = messages_for(prompt, CONDITIONS[condition])
                response, metadata = generator.generate(messages, args.max_new_tokens)
                append_jsonl(checkpoint, {
                    "control_version": CONTROL_VERSION, "scorer_version": SCORER_VERSION,
                    "family": family, "original_example_id": row["id"], "depth": depth,
                    "condition": condition, "system_suffix": CONDITIONS[condition],
                    "messages": messages, "raw_response": response,
                    "score": score_sifo_response_v32(row, response, depth, family),
                    "technical_failure": metadata["reached_generation_limit"],
                    "metadata": metadata,
                })
                if index % 100 == 0:
                    print(f"[{index}/{total}] {family}/{row['id']} {condition}", flush=True)
    print(f"done: {len(read_jsonl(checkpoint))} records -> {checkpoint}")


if __name__ == "__main__":
    main()
