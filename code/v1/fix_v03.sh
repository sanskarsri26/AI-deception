#!/usr/bin/env bash
set -euo pipefail

echo "Backing up run_hf_model.py..."
cp run_hf_model.py run_hf_model.py.bak

echo "Creating benchmark_parsing.py..."

cat > benchmark_parsing.py <<'PY'
import re

ITEM_ID_RE = r"T-\d{3}"

TICKET_LINE_RE = re.compile(
    rf"^\s*({ITEM_ID_RE})\s*:",
    re.MULTILINE
)

ALLOWED_LABEL_RE = re.compile(
    r"^\s*-\s*([a-z_]+)\s*:",
    re.MULTILINE
)


def expected_ids_from_prompt(prompt):
    return {
        x.upper()
        for x in TICKET_LINE_RE.findall(prompt)
    }


def allowed_labels_from_prompt(prompt):
    return {
        x.lower()
        for x in ALLOWED_LABEL_RE.findall(prompt)
    }


def valid_answer_ids(text, allowed_labels):
    if not allowed_labels:
        return set()

    labels = "|".join(
        re.escape(x)
        for x in sorted(
            allowed_labels,
            key=len,
            reverse=True,
        )
    )

    pattern = re.compile(
        rf"^\s*({ITEM_ID_RE})\s*\|\s*({labels})\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    return {
        match.group(1).upper()
        for match in pattern.finditer(text)
    }
PY


echo "Patching run_hf_model.py..."

python <<'PY'
from pathlib import Path
import re

path = Path("run_hf_model.py")
text = path.read_text()

# Remove any old benchmark_parsing import first.
text = re.sub(
    r'\nfrom benchmark_parsing import \([\s\S]*?\)\n',
    '\n',
    text
)

# Remove old local parsing functions.
for func in [
    "expected_ids_from_prompt",
    "allowed_labels_from_prompt",
    "valid_answer_ids",
]:
    pattern = (
        rf'\ndef {func}\([^)]*\):'
        rf'[\s\S]*?'
        rf'(?=\n(?:def|class) |\nif __name__|\Z)'
    )

    text = re.sub(
        pattern,
        '\n',
        text
    )

# Add the correct import after __future__ import if present.
import_block = """
from benchmark_parsing import (
    expected_ids_from_prompt,
    allowed_labels_from_prompt,
    valid_answer_ids,
)
"""

if "from __future__ import annotations" in text:
    text = text.replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\n" + import_block,
        1
    )
else:
    text = import_block.lstrip() + "\n" + text

path.write_text(text)

print("run_hf_model.py patched.")
PY


echo "Checking Python syntax..."

python -m py_compile benchmark_parsing.py
python -m py_compile run_hf_model.py


echo "Checking imports..."

python - <<'PY'
from benchmark_parsing import (
    expected_ids_from_prompt,
    allowed_labels_from_prompt,
    valid_answer_ids,
)

print("benchmark_parsing import: OK")
PY


echo "Checking that old local functions are gone..."

grep -n "def expected_ids_from_prompt" run_hf_model.py && {
    echo "ERROR: old expected_ids_from_prompt still exists"
    exit 1
} || true

grep -n "def allowed_labels_from_prompt" run_hf_model.py && {
    echo "ERROR: old allowed_labels_from_prompt still exists"
    exit 1
} || true


echo "Removing old failed sanity run..."

rm -rf runs/sanity_v03


echo "Running sanity test..."

python run_sanity_check.py \
    --model Qwen/Qwen2.5-7B \
    --device auto


echo
echo "===== SIZE 1 ====="
cat runs/sanity_v03/size001_trial01.response.txt

echo
echo "===== SIZE 5 ====="
cat runs/sanity_v03/size005_trial01.response.txt

echo
echo "===== SIZE 10 ====="
cat runs/sanity_v03/size010_trial01.response.txt
