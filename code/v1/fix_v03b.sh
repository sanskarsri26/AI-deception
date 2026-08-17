#!/usr/bin/env bash
set -euo pipefail

echo "Backing up current file..."
cp run_hf_model.py run_hf_model.py.before_v03b

python <<'PY'
from pathlib import Path

path = Path("run_hf_model.py")
lines = path.read_text().splitlines()

REMOVE_FUNCS = {
    "expected_ids_from_prompt",
    "allowed_labels_from_prompt",
    "valid_answer_ids",
}

new_lines = []
i = 0

while i < len(lines):
    stripped = lines[i].lstrip()

    matched_func = None
    for func in REMOVE_FUNCS:
        if stripped.startswith(f"def {func}("):
            matched_func = func
            break

    if matched_func:
        base_indent = len(lines[i]) - len(stripped)

        print(f"Removing local function: {matched_func}")

        i += 1

        while i < len(lines):
            line = lines[i]

            if not line.strip():
                i += 1
                continue

            current_indent = len(line) - len(line.lstrip())

            if current_indent <= base_indent:
                break

            i += 1

        continue

    new_lines.append(lines[i])
    i += 1


text = "\n".join(new_lines) + "\n"

# Remove any previous benchmark_parsing import block.
import_start = "from benchmark_parsing import ("

if import_start in text:
    before, rest = text.split(import_start, 1)

    if ")" in rest:
        _, after = rest.split(")", 1)
        text = before.rstrip() + "\n" + after.lstrip("\n")


IMPORT_BLOCK = """from benchmark_parsing import (
    expected_ids_from_prompt,
    allowed_labels_from_prompt,
    valid_answer_ids,
)
"""

if "from __future__ import annotations" in text:
    text = text.replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\n\n" + IMPORT_BLOCK,
        1,
    )
else:
    text = IMPORT_BLOCK + "\n" + text


path.write_text(text)

print("run_hf_model.py patched successfully.")
PY


echo
echo "Checking syntax..."
python -m py_compile run_hf_model.py
python -m py_compile benchmark_parsing.py


echo
echo "Checking for old functions..."

if grep -nE '^def (expected_ids_from_prompt|allowed_labels_from_prompt|valid_answer_ids)\(' run_hf_model.py; then
    echo "ERROR: old local parsing function still exists."
    exit 1
fi

echo "No old local parsing functions found."


echo
echo "Checking imports..."

python - <<'PY'
import run_hf_model

print("run_hf_model import: OK")
PY


echo
echo "Removing failed sanity results..."
rm -rf runs/sanity_v03


echo
echo "Running sanity check..."

python run_sanity_check.py \
    --model Qwen/Qwen2.5-7B \
    --device auto


echo
echo "================ SIZE 1 ================"
cat runs/sanity_v03/size001_trial01.response.txt

echo
echo "================ SIZE 5 ================"
cat runs/sanity_v03/size005_trial01.response.txt

echo
echo "================ SIZE 10 ================"
cat runs/sanity_v03/size010_trial01.response.txt
