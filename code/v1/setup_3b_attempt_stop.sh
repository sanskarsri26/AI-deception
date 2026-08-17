#!/usr/bin/env bash
set -euo pipefail

echo "Creating 3B-only attempt-based runner..."

cp run_hf_model.py run_hf_model_attempt.py

python <<'PY'
from pathlib import Path
import re

path = Path("run_hf_model_attempt.py")
text = path.read_text(encoding="utf-8")

class_marker = "class AllValidClassificationsSeen"

if class_marker not in text:
    raise SystemExit(
        "Could not find AllValidClassificationsSeen "
        "in run_hf_model.py"
    )

# Add an attempt parser immediately before the existing stopper.
helper = r'''
import re as _attempt_re

_ATTEMPTED_ANSWER_RE = _attempt_re.compile(
    r"^\s*(T-\d{3})\s*\|",
    _attempt_re.I | _attempt_re.M,
)

def attempted_answer_ids(text):
    """
    Return required-style IDs that the model attempted,
    regardless of whether the label is valid.
    """
    return {
        m.group(1).upper()
        for m in _ATTEMPTED_ANSWER_RE.finditer(text)
    }


'''

idx = text.index(class_marker)
text = text[:idx] + helper + text[idx:]

# Isolate the stopping-criteria class.
start = text.index(class_marker)

next_top = re.search(
    r"\n(?=(?:class|def)\s+[A-Za-z_])",
    text[start + len(class_marker):]
)

if next_top:
    end = (
        start
        + len(class_marker)
        + next_top.start()
        + 1
    )
else:
    end = len(text)

block = text[start:end]

# Replace ONLY the validity-based lookup inside the stopper.
pattern = re.compile(
    r"valid_answer_ids\(\s*([^,]+?)\s*,\s*[^)]+\)",
    re.S,
)

new_block, count = pattern.subn(
    r"attempted_answer_ids(\1)",
    block,
    count=1,
)

if count != 1:
    raise SystemExit(
        "Could not locate valid_answer_ids() "
        "inside stopping criterion."
    )

text = text[:start] + new_block + text[end:]

path.write_text(text, encoding="utf-8")

print("Created:", path)
print(
    "Stopping rule now uses attempted IDs "
    "instead of valid-label IDs."
)
PY


echo
echo "Creating 3B-only confirmation runner..."

cp run_extended_confirm.py run_extended_confirm_3b.py

python <<'PY'
from pathlib import Path
import re

path = Path("run_extended_confirm_3b.py")
text = path.read_text(encoding="utf-8")

# Use the 3B-specific model runner.
text = text.replace(
    '"run_hf_model.py"',
    '"run_hf_model_attempt.py"'
)

# Give capability runs generous output space.
pattern = re.compile(
    r"def token_budget\(size\):.*?"
    r"(?=\n\ndef |\nclass |\n\ndef main)",
    re.S,
)

replacement = '''def token_budget(size):
    return min(
        8192,
        max(1024, 48 * size + 512)
    )
'''

text, count = pattern.subn(
    replacement,
    text,
    count=1
)

if count != 1:
    print(
        "WARNING: token_budget() was not automatically "
        "replaced. Existing budget will be used."
    )

path.write_text(text, encoding="utf-8")

print("Created:", path)
PY


python -m py_compile run_hf_model_attempt.py
python -m py_compile run_extended_confirm_3b.py

echo
echo "Verification:"
grep -n -A20 "class AllValidClassificationsSeen" \
    run_hf_model_attempt.py | head -25

echo
echo "3B setup complete."
