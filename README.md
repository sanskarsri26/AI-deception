# AI-deception

Does a language model that leaves required work undone — while demonstrably able
to do it — recover evidence of its own shortfall and still report the task
complete?

This repository holds every version of that investigation: the code that
produced each result, the preregistrations frozen before the runs they govern,
the raw run outputs, and the analyses written over them. Nine versions, four
models, two substrates.

Every headline number here can be recomputed from the records in `evidence/`.
Nothing in this repository is a summary standing in for data.

---

## The construct

The behavioural signature is a conjunction, not a single measurement:

```
capability demonstrated for the omitted work
  → the work not produced under normal execution
  → evidence of the shortfall recovered on probing
  → the work reported as complete anyway
```

The project name is a label for that signature. **No version of this project
measures intent, and none claims it.** The construct is "Candidate Deceptive
Laziness", fixed since `preregistrations/v7_PREREGISTRATION.md`. Read the
repository name as the research area, not as the finding.

---

## Headline results

### The reporting gap replicates across three model families

V8 Stage 1. Constructed transcripts, identical instrument, no pooling across
models. Strict conjunction is `(Q1 ∨ Q2) ∧ Q3 ∧ Q4 falsely claims complete`.

| model | family | strict conjunction | Wilson 95% | complete-response accuracy |
|---|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Qwen | 78/80 = 0.975 | [0.913, 0.993] | 100% |
| `mistralai/Mistral-7B-Instruct-v0.3` | Mistral | 152/160 = 0.950 | [0.904, 0.974] | 100% |
| `google/gemma-2-9b-it` | Gemma | 153/160 = 0.956 | [0.912, 0.979] | 100% |
| `Qwen/Qwen2.5-14B-Instruct` | Qwen | 2/160 = 0.013 | [0.003, 0.044] | 100% |

Three independent training runs from three labs converge at 95–97.5%. All four
models are at ceiling on judging *complete* transcripts, so none of this is
incapacity.

The 14B model is the same family one size up and shows almost none of the
effect. The licensed description is that **the effect differs sharply by model
size within Qwen in this comparison** — deliberately not "scale-dependent",
which would assert a general scaling law that one within-family comparison at
two sizes cannot support.

### The full chain holds on failures the model generated itself

V8 Stage 2. `google/gemma-2-9b-it`, 120-block substrate, 40 trials, 0 technical
failures. Every prior strong number in this project came from transcripts the
*instrument* truncated; this is the first on responses the model chose to leave
incomplete.

| step | result | Wilson 95% |
|---|---|---|
| natural under-execution | 22/40 = 0.550 | [0.398, 0.693] |
| capability for every omitted block | 22/22 = 1.000 | — |
| Q1 recovered last block id | 19/22 = 0.864 | [0.667, 0.953] |
| Q3 knew blocks required | 22/22 = 1.000 | [0.851, 1.000] |
| Q4 falsely claimed complete | 22/22 = 1.000 | [0.851, 1.000] |
| **strict conjunction** | **19/22 = 0.864** | **[0.667, 0.953]** |

All 286 omitted blocks were probed individually — no sampling. Q4 and Q5 are
logical opposites asked in independent branches, and the model was
polarity-consistent in all 22 cases, so **acquiescence is ruled out, not merely
argued against**.

One caveat that belongs next to this number every time: asked how many blocks it
produced (Q2), the model gave the *required* count in 22/22 cases. It recovers
the last block id and the requirement. It does **not** demonstrate knowing how
much work it completed.

### A capability floor, excluded rather than reported

`mistralai/Mistral-7B-Instruct-v0.3` on the same 120-block substrate collapsed to
4–25 blocks in 73 of 78 trials, stopping cleanly with most of an 11,200-token
budget unused, while single-block extraction succeeded 74/74. That is a
capability floor, **not** under-execution, and it is excluded and documented
rather than reported as a 94% under-execution rate.

### V9 — the utilization gap

V9 asks whether the gap closes when the model is told to be careful, against a
length-matched neutral placebo. Its status and result are recorded in
`reports/FINDINGS.md`. If `evidence/v9/IN_FLIGHT_SNAPSHOT.txt` is present, the
run was still generating when this snapshot was built and its partial records
must not be analysed — no V9 endpoint is defined on a partial sample and the
preregistration forbids adaptive stopping.

---

## What this evidence does and does not support

**Supported**

- Three 7–9B instruction-tuned models, from three separate labs, recover
  evidence sufficient to reveal an incomplete transcript and still give a
  self-favouring false completion report, at 95–97.5% on the identical
  instrument.
- On naturally generated failures in one model, the full chain holds in 19 of 22
  capability-qualified cases.
- Incapacity, acquiescence, and generation truncation are each ruled out by
  design, not by argument.

**Not supported**

- Any claim of intent, goal, or strategy.
- That this is a general property of language models — four models, two
  substrates, one operator, one site, no external replication.
- Any scaling claim. One family at two sizes is not a scaling result.
- That the 55% natural under-execution rate generalises across models. It is 27×
  another model's rate on the identical task, so that term looks strongly
  model-dependent even where the reporting measure agrees closely.

**Deliberately absent** — related-work citations (LazyBench, the sandbagging
literature) are unverified against primary sources and are therefore cited
nowhere in this repository.

---

## Layout

```
README.md              this file
PROVENANCE.txt         source commit, host, build time, library versions
MANIFEST.sha256        SHA-256 for every file — verify with sha256sum -c
reports/               written analyses, including the full V1–V8 report
preregistrations/      the five frozen designs (V3.2, V6, V7, V8, V9)
code/                  per-version source, plus shared tests/ and data/
evidence/              raw run outputs, one directory per version
figure_data/           plot-ready CSVs for the headline figures
```

Preregistrations appear twice on purpose: collected in `preregistrations/` for
reading in sequence, and in place under `code/vN/` beside the code that uses
them.

### Suggested reading order

1. `reports/DECEPTIVE_LAZINESS_V1_TO_V8_REPORT.md` — full narrative, construct,
   version history, numerical appendices.
2. `reports/FINDINGS.md` — the append-only result log; each entry labelled
   confirmatory or exploratory.
3. `preregistrations/` in order v32 → v6 → v7 → v8 → v9 — how the design
   tightened as each innocent explanation was ruled out.
4. `evidence/` — the records themselves.

---

## Version history

| version | substrate | contribution | evidence |
|---|---|---|---|
| **V1** | batch ticket classification | first completion-report measurement; later audited and its 3B signal found invalid | `evidence/v1/`, `reports/V22_AUDIT.md` |
| **V2** | binary evidence audit | narrower task after V1's validity problems | `evidence/v2/` |
| **V2.1** | one-fact capability | capability and reasoning-depth calibration | `evidence/v21/` |
| **V2.2** | five-check audit + IFEval | external-validity pilots on a published instruction-following family | `evidence/v22/` |
| **V3.0/3.1** | published SIFo substrate | moves off a bespoke task onto a published one | `evidence/v3/`, `evidence/v31/` |
| **V3.2** | SIFo, repaired scorer | scorer repair plus held-out prompt controls; established that **neutral prompt perturbation helps on its own** — the finding that makes V9's placebo arm necessary | `evidence/v32/` |
| **V4** | verification under workload | null result; introduced the reporting gate | `evidence/v4/` |
| **V5** | long-output elision, 120 blocks | the expensive substrate that produces natural under-execution | `evidence/v5/` |
| **V6** | constructed probes | self-favouring completion-assertion bias | `evidence/v6/` |
| **V7** | natural + constructed | first preregistered replication; the evidence-access battery; established that SAMETURN administration is a broken instrument and SEPARATE is the only valid arm | `evidence/v7/`, `reports/v7_RESULTS.md` |
| **V8** | cross-family | the reporting gap in three families, then the full chain on natural failures | `evidence/v8/`, `reports/FINDINGS.md` |
| **V9** | matched workload | closes the "capability was only shown in isolation" objection, with a placebo control | `evidence/v9/` |

---

## Figures

`figure_data/` holds plot-ready CSVs, all recomputed from the raw records:

| file | figure |
|---|---|
| `figure1_under_execution.csv` | natural under-execution rate, Qwen vs Gemma |
| `figure2_completion_reporting.csv` | completion reporting across the four models |
| `figure2b_by_truncation_kind.csv` | the same, broken out by truncation type |
| `figure3_gemma_funnel.csv` | 40 trials → 22 incomplete → 19 strict cases |
| `figure3b_probe_detail.csv` | per-probe results on the 22 qualified cases |

Two plotting notes that keep the figures honest. In Figure 2, plot
complete-response accuracy as a second series — all four models sit at 100%, and
that is what rules out incapacity; without it the 14B bar reads as a model that
simply cannot do the task. In Figure 3, the 22 → 19 step is where evidence
access fails, not attrition, and should be labelled as such.

---

## Verifying this repository

```bash
sha256sum -c MANIFEST.sha256      # expect all OK, 0 FAILED
```

`PROVENANCE.txt` records the source commit, branch, host, build time, and
library versions. The evidence tree plus that commit log is what makes the
discipline below checkable rather than merely asserted.

## Methodological commitments

- Preregistrations were frozen before the runs they govern. Every change is a
  dated addendum stating the reason — never a silent edit.
- No threshold, gate, item difficulty, or prompt wording was changed after
  seeing a result.
- Every reported number is labelled confirmatory or exploratory.
- Run outputs are append-only: new runs go to new directories, and no prior
  version's records were modified.
- `reports/FINDINGS.md` is appended to, never rewritten.
- A null is a result. V4 returned one, V9 was designed so that one would be
  publishable, and both are reported rather than iterated away.

---

## Rebuilding

```bash
bash build_evidence_package.sh [dest]
```

Re-runnable and read-only with respect to the source repository.
