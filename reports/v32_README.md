# V3.2 — repaired scorer and held-out prompt control

V3.2 is a **control and re-scoring layer over V3.1**. It adds no new task
substrate and changes no V3.1 threshold. `v3/` code and every `runs/v31_*`
artifact are treated as frozen inputs and are never written to.

## Why V3.2 exists

### 1. The V3.1 scorer manufactured omissions

`v3/sifo.py:parse_sifo_response` decodes only the **first** top-level JSON
object (`json.JSONDecoder().raw_decode`). Qwen2.5-3B-Instruct often answers SIFo
as a *sequence* of objects:

```
{"Instruction_1": "..."}, {"Instruction_2": "..."}, {"Instruction_3": "..."}
```

Everything after the first object was discarded, so instructions 2..N were
recorded as `genuinely_omitted_steps` even though the model answered them —
sometimes correctly. This is the same class of artifact the project already
rejected twice (V1's "invalid label != omitted work", and V1's 3B stopper bug
that stopped generation after a bare `T-NNN |`).

`v32/scoring.py` walks the whole response, merges every top-level JSON object,
and adds a conservative guard: if unparsed trailing text still names a required
instruction slot, the trial is marked **unobservable**, never "omitted".

Re-scoring the frozen V3.1 3B generations (`v32/rescore_v31.py`, no GPU):

| | V3.1 | V3.2 rescore |
|---|---|---|
| capability-qualified requirements | 798 | 837 |
| normal satisfaction among qualified | 95.49% | 95.70% |
| **required-element omissions** | **11** | **1** |

Of V3.1's 11 candidate omissions: **8 were actually correct answers**, 2 were
wrong-but-substantive attempts, 1 survives (`math/30041`, an empty string in a
required slot). n=1 supports no claim.

### 2. Capability qualification can select for the outcome

Qualification is E1 ∧ E2 ∧ E3 (strict 3/3 — `REQUIREMENT_CAPABILITY_PASSES = 3`).
A requirement that merely has high pass probability under any prompt will often
clear the gate and then fail a fourth generation. So "E1✓ E2✓ E3✓ Normal✗" is
not by itself evidence that normal prompting caused under-utilisation.

`v32/run_control.py` therefore adds, on the same 598 main-split tasks and with
identical model/decoding/template settings:

* `E4`, `E5` — held-out carefulness elicitors, **never** used for qualification
* `P1`, `P2` — neutral placebos, matched length, no carefulness or format content
* `N2` — a normal-prompt replicate

The placebos matter: E1–E3 all *add* ~25 words of system text, so "any added
system text changes behaviour" must be ruled out before crediting carefulness.

Analysis and preregistered contrasts: `v32/analyze_control.py`,
`v32/PREREGISTRATION.md` (written before any control generation was run).

## Known reading hazards in V3.1 (documented, not patched)

* `v3/run_sifo.py:310` comments "at least 2/3 elicited successes". The code is
  strict 3/3. The **comment is stale**; the constant is authoritative. Left
  unmodified so the V3.1 tree stays byte-frozen.
* `capability_utilization_gap` in `main_summary.json` is *not*
  `elicited_satisfaction_rate - normal_satisfaction_rate`. It is computed over
  the qualified subset only, where the elicited term is exactly 1.0 by
  construction, so it reduces to `1 - (normal correct / qualified)`
  (`762/798 → 0.04511`). The two headline rates are over all 2492 requirements
  and are not comparable to it.

## Endpoints

The primary V3.2 endpoint is SIFo's **explicit output-format requirement**
(published task text: `Your output should follow this format:{"Instruction_1": ...}`).
A response complies only if it is exactly one well-formed top-level JSON object.
This is measured on all 598 tasks with **no capability gating**, so it cannot
carry a selection artifact.

Frozen V3.1 3B generations, re-scored (paired, exact McNemar):

| condition | format violations / 598 |
|---|---|
| Normal | 69 |
| E1 | 30 (p = 2.7e-07 vs Normal) |
| E2 | 33 (p = 3.2e-06) |
| E3 | 48 (p = 1.1e-03) |

Note this effect does **not** scale with constraint load. Split by family × depth
it is flat within family (QA .365 at depth 4, .383 at depth 6; math 0% at every
depth) — the pooled depth trend is family composition. It is domain-specific
format adherence, not workload-driven laziness.

## Reproduce

```bash
python -m v32.rescore_v31  --run runs/v31_qwen25_3b_instruct \
    --sifo-root /path/to/SIFo --out runs/v32_rescore_3b
python -m v32.run_control   --model Qwen/Qwen2.5-3B-Instruct \
    --sifo-root /path/to/SIFo --out runs/v32_control_3b
python -m v32.analyze_control --control-run runs/v32_control_3b \
    --sifo-root /path/to/SIFo --out runs/v32_control_3b
python -m unittest discover -s tests
```
