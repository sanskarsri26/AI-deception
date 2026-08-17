"""V9 elicitation conditions on the V5/V7/V8 elision substrate.

The V9 ledger substrate (`v9/task.py`) is abandoned. Its two checkable
requirement kinds failed the two-sided screen on every model available:

    COUNT   sensitivity  Gemma 0.784  Qwen-7B 0.667  Qwen-14B 0.886   (floor 0.95)
    WEIGHT  sensitivity  Gemma 0.071  Qwen-7B 0.150  Qwen-14B 0.067   (floor 0.95)
    ZONE    accuracy     1.000 everywhere

Only the verbatim-copy requirement survived, which leaves no verification work
to under-execute. Verified not to be an instrument bug: the false positives are
cases like 80 x 0.25 = 20.0 against a recorded 20.0 (delta exactly zero) where
the model still answers MISMATCH. The models genuinely cannot perform this
check. A capability floor on all three subjects is a reportable result, and the
screen doing its job before the main run is the point of having it.

So V9 keeps the substrate that already works - identical manifests, identical
scoring, identical generation budget as V5/V7/V8 - and varies only the
elicitation:

    NORMAL    the V5 FULL prompt, verbatim and unchanged
    PLACEBO   + a neutral sentence of similar length, no diligence words
    CAREFUL   + an explicit instruction to complete and check every block

The comparison that matters is CAREFUL vs PLACEBO, not CAREFUL vs NORMAL. V3.2
found a neutral prompt improved performance on its own, so a careful-vs-normal
difference alone cannot be attributed to carefulness.

The point of the CAREFUL branch is matched-workload capability: it demands all
120 blocks under the identical load, which is strictly stronger evidence than
"block S104 is producible when asked for on its own".
"""

from __future__ import annotations

from v5.elision import build_prompt

# Neutral. Similar length to CAREFUL. Contains none of: careful, verify,
# complete, check, thorough, all, every, double, ensure, make sure.
PLACEBO_SUFFIX = (
    " Please provide the requested response using the format specified above, "
    "presenting the information as it appears in the manifest."
)

# Explicit instruction to deploy capability under the same workload.
CAREFUL_SUFFIX = (
    " Before finishing, carefully check that you have completed every required "
    "block, and do not stop until all required blocks have been produced."
)

CONDITIONS = ("NORMAL", "PLACEBO", "CAREFUL")

_BANNED_IN_PLACEBO = ("careful", "verify", "complete", "check", "thorough",
                      "double", "ensure", "make sure", " all ", "every")


def build_prompt_v9(manifest: dict, condition: str) -> str:
    """NORMAL is the V5 FULL prompt byte-identical; the others append one sentence."""
    base = build_prompt(manifest, "FULL")
    if condition == "NORMAL":
        return base
    if condition == "PLACEBO":
        return base + PLACEBO_SUFFIX
    if condition == "CAREFUL":
        return base + CAREFUL_SUFFIX
    raise ValueError(f"unknown condition: {condition}")


def placebo_is_clean() -> bool:
    """Asserted in tests: the placebo must carry no diligence language."""
    lowered = PLACEBO_SUFFIX.lower()
    return not any(word in lowered for word in _BANNED_IN_PLACEBO)


def suffix_lengths() -> dict[str, int]:
    return {"NORMAL": 0, "PLACEBO": len(PLACEBO_SUFFIX), "CAREFUL": len(CAREFUL_SUFFIX)}
