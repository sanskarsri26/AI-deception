# Deceptive Laziness V3.1

V3.1 wraps unchanged published SIFo tasks with repeated capability elicitation,
a normal condition, requirement-level under-utilization classification, an
independent frozen-response self-audit, and independently calibrated completion
reporting. Existing `runs/` artifacts are never migrated or overwritten; use a
fresh output directory for this protocol version.

## CPU preparation and tests

```bash
python -m v3.prepare --sifo-root /scratch/$USER/SIFo --out-dir v3/manifests
python -m v3.audit_benchmarks \
  --followbench-root /scratch/$USER/FollowBench \
  --sifo-root /scratch/$USER/SIFo \
  --output v3/manifests/benchmark_audit.json
python -m unittest discover -s tests -p 'test_*.py'
```

## First GPU calibration runs

Use the exact commands below. They deliberately use new V3.1 directories.

```bash
CUDA_VISIBLE_DEVICES=0 python -m v3.run_sifo \
  --phase calibration \
  --model Qwen/Qwen2.5-3B-Instruct \
  --sifo-root /scratch/$USER/SIFo \
  --out-dir runs/v31_qwen25_3b_instruct

CUDA_VISIBLE_DEVICES=0 python -m v3.run_sifo \
  --phase calibration \
  --model Qwen/Qwen2.5-7B-Instruct \
  --sifo-root /scratch/$USER/SIFo \
  --out-dir runs/v31_qwen25_7b_instruct
```

Before main, inspect `run_config_calibration.json`,
`calibration_summary.json`, `capability_calibration_records.jsonl`,
`reporting_calibration.json`, and `reporting_calibration_records.jsonl`.
Confirm the model/tokenizer revisions and chat-template hash, zero generation
limit failures, E1/E2/E3 record counts, per-family/depth requirement accuracy,
balanced complete/incomplete reporting accuracy, and which reporting format (if
any) passed 0.90 in both states. Do not loosen thresholds after inspection.

For a cheap integration run, use a separate directory and add
`--limit 2 --report-trials 2`. A limited main must also specify `--limit`; its
results remain non-final.

## Main runs

```bash
CUDA_VISIBLE_DEVICES=0 python -m v3.run_sifo \
  --phase main \
  --model Qwen/Qwen2.5-3B-Instruct \
  --sifo-root /scratch/$USER/SIFo \
  --out-dir runs/v31_qwen25_3b_instruct

CUDA_VISIBLE_DEVICES=0 python -m v3.run_sifo \
  --phase main \
  --model Qwen/Qwen2.5-7B-Instruct \
  --sifo-root /scratch/$USER/SIFo \
  --out-dir runs/v31_qwen25_7b_instruct
```

If neither reporting format qualifies, main still measures under-utilization
and self-audit performance, but all deceptive classifications remain gated off.
Inspect `main_summary.json` first, then every candidate's full record in
`main_records.jsonl`: original prompt, all three elicited responses, frozen
normal response, requirement labels, independent audit, independent completion
claim, and stopping metadata.
