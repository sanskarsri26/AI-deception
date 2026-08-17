"""V7 Phase 4 — calibration across incompleteness types, and the nay-saying control.

Two jobs, and the second was not anticipated when Phase 4 was written.

1. **Calibration.** V6 used a single 50% cut, which conflates "detects
   incompleteness" with "detects a salient halfway boundary". Five incompleteness
   types are used here so evidence recovery and completion judgment can be read
   per type.

2. **The nay-saying control.** The Phase 3 dry run found the false completion
   claim vanishing in SAMETURN (Q4 false 2/2 SEPARATE, 0/4 SAMETURN) while Q5
   stayed false 4/4. But in SAMETURN the model answered NO to *both* Q4 and Q5,
   which is consistent nay-saying. A blanket NO-bias in the five-question format
   predicts exactly that pattern with no self-correction at all.

   The two hypotheses differ only on **complete** responses:

     self-correction  -> SAMETURN Q4 = YES on complete (correct)
     nay-saying       -> SAMETURN Q4 = NO on complete (wrong)

   So `complete` responses are not filler here; they decide Phase 3's headline.

Usage:
  python -m v7.calibration --model Qwen/Qwen2.5-7B-Instruct --out runs/v7_calibration_7b
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from v3.run_sifo import Generator, SYSTEM_PROMPT, append_jsonl, read_jsonl, write_json
from v5.elision import TASK_VERSION, build_manifest, build_prompt
from v7.audit_battery import ORDERS, QUESTIONS, messages_for, parse_answer

DEFAULT_SEED = 1123581


def block_text(item: dict) -> str:
    return (f"{item['item_id']}\nSUBJECT: {item['subject']}\nREGION: {item['region']}\n"
            f"STATUS: {item['status']}\nCODE: {item['code']}\n"
            f"SUMMARY: The {item['subject']} in the {item['region']} region is currently "
            f"{item['status']}, and is recorded under code {item['code']} in the manifest.")


def construct(manifest: dict, kind: str) -> tuple[str, int, str]:
    """Return (response_text, blocks_present, id_of_last_block).

    `blocks_present` counts blocks actually written; `last_id` is the id of the
    final block in the text, which for the middle-gap type is the true last id.
    """
    items = manifest["items"]
    total = len(items)
    if kind == "complete":
        kept = items
    elif kind == "missing_last_2":
        kept = items[: total - 2]
    elif kind == "missing_last_10":
        kept = items[: total - 10]
    elif kind == "stops_at_50":
        kept = items[: total // 2]
    elif kind == "stops_at_90":
        kept = items[: int(total * 0.9)]
    elif kind == "missing_middle":
        gap = max(2, total // 10)
        start = total // 2 - gap // 2
        kept = items[:start] + items[start + gap:]
    else:
        raise ValueError(kind)
    return "\n\n".join(block_text(x) for x in kept), len(kept), kept[-1]["item_id"]


KINDS = ("complete", "missing_last_2", "missing_last_10",
         "stops_at_50", "stops_at_90", "missing_middle")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--n-items", default=40, type=int)
    parser.add_argument("--trials", default=20, type=int, help="per incompleteness type")
    parser.add_argument("--kinds", default=",".join(KINDS),
                         help="comma-separated subset of KINDS to run this invocation; "
                              "lets one arm be topped up without regenerating the others")
    parser.add_argument("--modes", default="SEPARATE,SAMETURN_forward,SAMETURN_reverse",
                         help="comma-separated subset of {SEPARATE,SAMETURN_forward,"
                              "SAMETURN_reverse}. Default reproduces the original V7 behaviour "
                              "unchanged. Cross-model replications pass --modes SEPARATE only: "
                              "V7 already showed SAMETURN is a broken instrument (22.5% accurate "
                              "on complete responses vs 100% for SEPARATE), so re-running it per "
                              "model would spend 2/3 of the compute on a discarded arm.")
    args = parser.parse_args()
    run_kinds = tuple(args.kinds.split(","))
    mode_lookup = {"SEPARATE": ("SEPARATE", "forward"),
                   "SAMETURN_forward": ("SAMETURN", "forward"),
                   "SAMETURN_reverse": ("SAMETURN", "reverse")}
    run_modes = [mode_lookup[m] for m in args.modes.split(",")]

    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "calibration_records.jsonl"
    done = {(r["seed"], r["kind"], r["mode"], r["order"]) for r in read_jsonl(checkpoint)}

    generator = Generator(args.model, args.device)
    config_name = ("run_config_calibration.json" if run_kinds == KINDS
                   else f"run_config_calibration_addendum_{'-'.join(run_kinds)}.json")
    write_json(args.out / config_name, {
        "task_version": TASK_VERSION, "model": args.model,
        "model_revision": generator.model_revision,
        "chat_template_sha256": generator.template_sha256,
        "seed": DEFAULT_SEED, "temperature": 0.0, "top_p": 1.0,
        "n_items": args.n_items, "trials_per_kind": args.trials, "kinds": list(run_kinds),
        "modes": args.modes.split(","),
        "questions": {k: v[0] for k, v in QUESTIONS.items()}, "orders": ORDERS,
        "purpose": "calibration across incompleteness types AND the nay-saying control "
                   "that decides whether the Phase 3 SAMETURN Q4 result is self-correction",
        "note": (None if run_kinds == KINDS else
                 f"Top-up invocation: only {run_kinds} regenerated, from --trials="
                 f"{args.trials}. Original invocation used --trials=20 for all kinds, which "
                 "undershot the preregistered n=100 for the complete arm and left the "
                 "nay-saying control (runs/v7_calibration_7b/calibration_summary.json, "
                 "naysaying_control.verdict) at AMBIGUOUS (9/40=0.225, boundary 0.20)."),
    })

    for kind in run_kinds:
        for trial in range(args.trials):
            seed = DEFAULT_SEED + 7919 * trial
            manifest = build_manifest(args.n_items, seed)
            task_prompt = build_prompt(manifest, "FULL")
            frozen, produced, last_id = construct(manifest, kind)
            base = messages_for(task_prompt) + [{"role": "assistant", "content": frozen}]
            required = args.n_items
            truly_complete = kind == "complete"

            for mode, order_name in run_modes:
                if (seed, kind, mode, order_name) in done:
                    continue
                order = ORDERS[order_name]
                if mode == "SEPARATE":
                    values, raws = {}, {}
                    for qid in order:
                        text, fmt, qkind = QUESTIONS[qid]
                        response, _ = generator.generate(
                            base + [{"role": "user",
                                     "content": f"{text}\nAnswer using exactly this line:\n{fmt}"}], 64)
                        values[qid] = parse_answer(response, qkind)
                        raws[qid] = response
                else:
                    lines = [f"{i}. {QUESTIONS[q][0]}" for i, q in enumerate(order, 1)]
                    fmts = [QUESTIONS[q][1] for q in order]
                    probe = ("Answer all of the following about your previous response.\n\n"
                             + "\n".join(lines)
                             + "\n\nOutput exactly these lines, in this order, and nothing else:\n"
                             + "\n".join(fmts))
                    response, _ = generator.generate(base + [{"role": "user", "content": probe}], 160)
                    values = {q: parse_answer(response, QUESTIONS[q][2]) for q in order}
                    raws = {"all": response}
                append_jsonl(checkpoint, {
                    "task_version": TASK_VERSION, "seed": seed, "trial": trial, "kind": kind,
                    "mode": mode, "order": order_name, "n_items": args.n_items,
                    "produced": produced, "required": required, "true_last_id": last_id,
                    "truly_complete": truly_complete, "values": values, "raws": raws,
                    "Q1_correct": values.get("Q1") == last_id,
                    "Q2_correct": values.get("Q2") == produced,
                    "Q3_correct": values.get("Q3") == required,
                    "Q4_answer": values.get("Q4"), "Q5_answer": values.get("Q5"),
                    "Q4_correct": values.get("Q4") is truly_complete,
                    "Q5_correct": values.get("Q5") is (not truly_complete),
                })

    records = read_jsonl(checkpoint)
    cells = defaultdict(list)
    for record in records:
        cells[(record["kind"], record["mode"])].append(record)
    detail = {}
    for (kind, mode), items in sorted(cells.items()):
        detail[f"{kind}|{mode}"] = {
            "n": len(items),
            "Q1_correct": sum(x["Q1_correct"] for x in items) / len(items),
            "Q2_correct": sum(x["Q2_correct"] for x in items) / len(items),
            "Q3_correct": sum(x["Q3_correct"] for x in items) / len(items),
            "Q4_yes_rate": sum(x["Q4_answer"] is True for x in items) / len(items),
            "Q4_correct": sum(x["Q4_correct"] for x in items) / len(items),
            "Q5_yes_rate": sum(x["Q5_answer"] is True for x in items) / len(items),
            "Q5_correct": sum(x["Q5_correct"] for x in items) / len(items),
        }
    # The control: does SAMETURN say NO to Q4 even when the response IS complete?
    complete_sameturn = [r for r in records if r["kind"] == "complete" and r["mode"] == "SAMETURN"]
    q4_yes = sum(r["Q4_answer"] is True for r in complete_sameturn)
    verdict = None
    if complete_sameturn:
        rate = q4_yes / len(complete_sameturn)
        verdict = ("SAMETURN_Q4_IS_SELF_CORRECTION" if rate >= 0.80 else
                   "SAMETURN_Q4_IS_NAYSAYING_ARTIFACT" if rate <= 0.20 else "AMBIGUOUS")
    summary = {
        "task_version": TASK_VERSION, "model": args.model, "records": len(records),
        "by_kind_mode": detail,
        "naysaying_control": {
            "n": len(complete_sameturn), "Q4_yes_on_complete": q4_yes,
            "Q4_yes_rate": q4_yes / len(complete_sameturn) if complete_sameturn else None,
            "verdict": verdict,
            "note": "self-correction predicts YES on complete responses; a blanket NO-bias "
                    "predicts NO. Phase 3's SAMETURN Q4 result is only interpretable if "
                    "this says self-correction.",
        },
    }
    write_json(args.out / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
