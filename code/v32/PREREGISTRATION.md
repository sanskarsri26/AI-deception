# V3.2 control experiment — preregistered analysis plan

Written **before** any V3.2 control generation was run. Thresholds, endpoints and
tests below are frozen. V3.1 outputs are never modified.

Model: `Qwen/Qwen2.5-3B-Instruct` (revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`),
greedy decoding (`do_sample=False`, `max_new_tokens=1200`, seed 314159), official
chat template — identical to `runs/v31_qwen25_3b_instruct/run_config_main.json`.
Tasks: the same 598 SIFo main-split examples used by the V3.1 main run.

## Question

Do the V3.1 carefulness elicitors (E1/E2/E3) actually cause better requirement
utilisation, or does *any* system-message perturbation move behaviour equally?
And is normal-vs-elicited difference on E1/E2/E3-qualified requirements a real
effect or a selection artifact of the qualification gate?

## Conditions (one generation per task per condition)

| id | system suffix | role |
|----|---------------|------|
| `N2` | none (identical to V3.1 normal) | determinism / replication check |
| `E4` | held-out carefulness elicitor | **primary treatment** |
| `E5` | second held-out carefulness elicitor | treatment replication |
| `P1` | neutral placebo, matched length, no carefulness or format content | **primary control** |
| `P2` | second neutral placebo | control replication |

E4/E5 never participate in capability qualification, which stays frozen at
E1 ∧ E2 ∧ E3 (all three), as in V3.1.

## Endpoints

**Primary (E1): explicit output-format requirement.** SIFo's published task text
states `Your output should follow this format:{"Instruction_1": ...}`. A response
*violates* this requirement unless the whole response is exactly one well-formed
top-level JSON object. This endpoint is measured on all 598 tasks with **no
capability gating**, so it cannot suffer regression-to-the-mean from selection.

**Secondary (E2): requirement satisfaction among E1/E2/E3-qualified
requirements.** Measured with the V3.2 repaired scorer on the requirements
qualified by E1∧E2∧E3.

**Tertiary (E3): required-element omission count** among qualified requirements,
V3.2 scorer.

## Preregistered tests

* Primary contrast: **E4 vs P1**, paired at task level, **exact McNemar** on the
  format-violation indicator. Two-sided, α = 0.05.
* Secondary contrast: **E4 vs Normal (N1)**, same test.
* Placebo check: **P1 vs Normal**. If this is significant, "any added system
  text changes behaviour" and elicitor-specific claims are not supported.
* Replications E5 and P2 are reported alongside but the confirmatory decision
  rests on E4 vs P1.
* Uncertainty: **task-clustered bootstrap** (10,000 resamples of tasks) for the
  difference in violation rates. Requirements nest inside tasks — a task's JSON
  parses or does not parse as a whole — so requirement-level observations are
  **not** independent and requirement-level analysis is reported only with
  task-clustered CIs.
* Determinism: N2 vs N1 byte-identical response rate is reported. If N2 ≡ N1,
  repeated-normal runs carry no information and stochastic instability cannot be
  the explanation for any normal/elicited difference.

## Decision rules (frozen)

* **Outcome A — real elicitation effect.** E4 violation rate significantly below
  P1 (McNemar p < 0.05) *and* P1 not significantly below Normal.
* **Outcome B — prompt-perturbation artifact.** E4 ≈ P1, both below Normal.
  Then "carefulness" is not the active ingredient; any system suffix is.
* **Outcome C — no effect.** E4 ≈ P1 ≈ Normal. The V3.1 normal/elicited gap is
  selection, not under-utilisation.

No threshold or endpoint may be revised after results are seen.
