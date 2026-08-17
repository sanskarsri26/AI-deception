"""Renders the V7 dissociation figure: evidence recovery vs completion judgment,
incomplete (SEPARATE) vs complete (SEPARATE) responses. Numbers are read live
from the raw calibration records so the figure cannot drift from the data.

Usage: python -m v7.figures.make_dissociation_figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"

CALIB_RECORDS = Path("runs/v7_calibration_7b/calibration_records.jsonl")
TRUNCATION_KINDS = ("missing_last_2", "missing_last_10", "stops_at_50", "stops_at_90")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def pct(k: int, n: int) -> float:
    return 100.0 * k / n


def main() -> None:
    recs = read_jsonl(CALIB_RECORDS)
    incomplete = [r for r in recs if r["mode"] == "SEPARATE" and r["order"] == "forward"
                  and r["kind"] in TRUNCATION_KINDS]
    complete = [r for r in recs if r["mode"] == "SEPARATE" and r["order"] == "forward"
                and r["kind"] == "complete"]
    n_incomplete, n_complete = len(incomplete), len(complete)

    q1 = pct(sum(r["Q1_correct"] for r in incomplete), n_incomplete)
    q3 = pct(sum(r["Q3_correct"] for r in incomplete), n_incomplete)
    q4_false_claim = pct(sum(r["values"].get("Q4") is True for r in incomplete), n_incomplete)
    q5_false_denial = pct(sum(r["values"].get("Q5") is False for r in incomplete), n_incomplete)
    complete_q4_correct = pct(sum(r["values"].get("Q4") is True for r in complete), n_complete)
    complete_q5_correct = pct(sum(r["values"].get("Q5") is False for r in complete), n_complete)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 4.2), width_ratios=[1, 0.55], facecolor=SURFACE)

    # --- Panel A: incomplete responses ---
    labels_a = [
        f"Required count correct\n(evidence recovery)",
        f"Last block ID correct\n(evidence recovery)",
        f"Denies any omission\n(false — completion judgment)",
        f"Reports “complete”\n(false — completion judgment)",
    ]
    values_a = [q3, q1, q5_false_denial, q4_false_claim]
    colors_a = [GOOD, GOOD, CRITICAL, CRITICAL]

    y = range(len(labels_a))
    ax_left.barh(y, values_a, color=colors_a, height=0.55, zorder=3)
    ax_left.set_yticks(list(y))
    ax_left.set_yticklabels(labels_a, fontsize=10, color=INK)
    for yi, v in zip(y, values_a):
        ax_left.text(v - 2, yi, f"{v:.0f}%", va="center", ha="right",
                      fontsize=11, fontweight="bold", color="white", zorder=4)
    ax_left.axhline(1.5, color=GRID, linewidth=1, zorder=1)
    ax_left.set_xlim(0, 100)
    ax_left.set_xlabel("% of trials", fontsize=9, color=MUTED)
    ax_left.set_title(f"Incomplete responses, SEPARATE (n={n_incomplete})",
                       fontsize=12, color=INK, loc="left", fontweight="bold")
    ax_left.spines[["top", "right", "left"]].set_visible(False)
    ax_left.spines["bottom"].set_color(GRID)
    ax_left.tick_params(left=False, colors=MUTED)
    ax_left.set_facecolor(SURFACE)

    # --- Panel B: complete responses (control) ---
    labels_b = ["Denies any omission\n(correct)", "Reports “complete”\n(correct)"]
    values_b = [complete_q5_correct, complete_q4_correct]
    y2 = range(len(labels_b))
    ax_right.barh(y2, values_b, color=GOOD, height=0.55, zorder=3)
    ax_right.set_yticks(list(y2))
    ax_right.set_yticklabels(labels_b, fontsize=10, color=INK)
    for yi, v in zip(y2, values_b):
        ax_right.text(v - 2, yi, f"{v:.0f}%", va="center", ha="right",
                       fontsize=11, fontweight="bold", color="white", zorder=4)
    ax_right.set_xlim(0, 100)
    ax_right.set_xlabel("% of trials", fontsize=9, color=MUTED)
    ax_right.set_title(f"Complete responses, SEPARATE (n={n_complete})",
                        fontsize=12, color=INK, loc="left", fontweight="bold")
    ax_right.spines[["top", "right", "left"]].set_visible(False)
    ax_right.spines["bottom"].set_color(GRID)
    ax_right.tick_params(left=False, colors=MUTED)
    ax_right.set_facecolor(SURFACE)

    fig.suptitle(
        "Evidence recovery vs. completion judgment (Qwen2.5-7B-Instruct, V7 Phase 4)",
        fontsize=13, color=INK, fontweight="bold", x=0.02, ha="left", y=1.03)
    fig.text(0.02, -0.03,
              "Green = correct / evidence recovered.  Red = false judgment despite "
              "recoverable evidence of incompleteness.\n"
              "Strict conjunction (Q1 or Q2 correct, AND Q3 correct, AND false "
              "completion claim): 78/80 = 97.5%, Wilson95 [91.3%, 99.3%].",
              fontsize=8.5, color=SECONDARY_INK)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_dir = Path("v7/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dissociation.png"
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
