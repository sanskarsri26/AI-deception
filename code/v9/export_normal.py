"""Export the V9 NORMAL branch in the exact shape `v7.capability_pass` and
`v7.audit_battery` expect.

Both V7 tools read `<source>/pilot_records.jsonl` and audit every record whose
`full_score.missing_item_ids` is non-empty. V9's driver writes
`elicit_records.jsonl` containing all three conditions, so pointing those tools
straight at a V9 run would either find nothing (wrong filename) or audit the
PLACEBO and CAREFUL branches too (wrong population).

This writes a `pilot_records.jsonl` containing **only NORMAL, replicate 0** —
the frozen branch the whole chain is defined on — leaving the V7 tools
completely unmodified so their numbers stay directly comparable to V8 Stage 2.

Usage:
  python -m v9.export_normal --run runs/v9_elicit_gemma --out runs/v9_normal_gemma
"""

from __future__ import annotations

import argparse
from pathlib import Path

from v3.run_sifo import append_jsonl, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--condition", default="NORMAL")
    args = parser.parse_args()

    records = read_jsonl(args.run / "elicit_records.jsonl")
    selected = [r for r in records
                if r["condition"] == args.condition and r["replicate"] == 0]

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "pilot_records.jsonl"
    if target.exists():
        target.unlink()          # regenerated from source, never appended to
    for record in selected:
        append_jsonl(target, record)

    failures = [r for r in selected
                if r["full_score"]["missing_item_ids"] and not r["technical_failure"]]
    summary = {
        "source_run": str(args.run), "condition": args.condition,
        "exported": len(selected), "total_records": len(records),
        "with_missing_and_no_technical_failure": len(failures),
        "note": "shape-compatible with v7.capability_pass / v7.audit_battery, which "
                "read pilot_records.jsonl; only the frozen NORMAL replicate-0 branch "
                "is exported so those tools audit the correct population",
    }
    write_json(args.out / "export_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
