#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen2.5-7B"

CAPACITY_DIR="runs/extended_capacity_7b_v04"
PILOT_DIR="runs/main_pilot_7b_v04"
LOG_DIR="runs/logs_v04"

mkdir -p "$CAPACITY_DIR"
mkdir -p "$PILOT_DIR"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Qwen2.5-7B benchmark on one A100"
echo "=========================================="
echo


# ============================================================
# 1. RUN MAIN BENCHMARK PILOT FIRST
# ============================================================

echo "Starting MAIN BENCHMARK PILOT..."
echo "Results: $PILOT_DIR"
echo

python run_main_benchmark.py \
    --model "$MODEL" \
    --capacity 100 \
    --capability-baseline 1.0 \
    --device auto \
    --trials 3 \
    --out-dir "$PILOT_DIR" \
    2>&1 | tee "$LOG_DIR/main_pilot.log"


echo
echo "=========================================="
echo "Main pilot finished."
echo "=========================================="
echo


# ============================================================
# 2. RUN EXTENDED CAPABILITY TEST
# ============================================================

echo "Starting EXTENDED CAPACITY TEST..."
echo "Results: $CAPACITY_DIR"
echo

python run_extended_capacity_v04.py \
    --model "$MODEL" \
    --device auto \
    --pool data/pilot_pool_extended_v04.jsonl \
    --sizes 125,150,175,200,250,300 \
    --trials 5 \
    --out-dir "$CAPACITY_DIR" \
    2>&1 | tee "$LOG_DIR/extended_capacity.log"


echo
echo "=========================================="
echo "Extended capacity test finished."
echo "=========================================="
echo


# ============================================================
# 3. SUMMARIZE CAPACITY
# ============================================================

python summarize_calibration.py \
    --dir "$CAPACITY_DIR"


echo
echo "=========================================="
echo "ALL RUNS COMPLETE"
echo "=========================================="
echo
echo "Main pilot:"
echo "  $PILOT_DIR"
echo
echo "Extended capacity:"
echo "  $CAPACITY_DIR"
echo
echo "Logs:"
echo "  $LOG_DIR/main_pilot.log"
echo "  $LOG_DIR/extended_capacity.log"
