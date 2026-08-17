# Published benchmark selection audit

## Decision

V3 uses original SIFo tasks as its primary substrate. FollowBench is the better
benchmark for graded constraint load, but SIFo is the cleaner substrate for
this experiment's decisive distinction: an absent/empty required output unit
versus a present, substantive, but wrong output unit.

| Property | FollowBench | IFEval | SIFo |
|---|---|---|---|
| Published task retained | Yes | Yes | Yes |
| Requirement-level scoring | Yes; rule checks for a subset and GPT-4 judgments otherwise | Yes; deterministic strict/loose checkers | Yes; normalized expected-answer matching |
| Natural load progression | Five evolved levels per path | One or more independently described constraints | Two to six sequential instructions |
| Objective omission signal | Usually no | Usually no | Yes, for valid JSON: one required field per instruction |
| Wrong attempt separable from omission | No general rule | No general rule | Yes, if a required field is present and nonempty |
| Main scorer limitation | Much of the suite requires an external LLM judge | Some checker heuristics are brittle and constraints can interact | Gold-in-prediction scoring is permissive; malformed output makes omission unobservable |

FollowBench's official LLM evaluation asks a judge for one YES/NO decision per
added constraint, so it does support requirement violations. A NO does not say
whether the model ignored the constraint, attempted it incorrectly, or made an
ordinary content error. Several sources have official rule checks, but these
also return constraint satisfaction rather than attempt evidence. Treating
every failed FollowBench constraint as under-execution would collapse the very
distinction V3 is intended to study.

IFEval is the strongest secondary family for objective requirement violations:
its original prompts carry `instruction_id_list` and checker arguments, and its
strict/loose scores are deterministic. Its responses do not generally contain
one independently required answer slot per constraint. For example, a response
that misses a keyword-count constraint may still be a substantive attempt.
IFEval therefore distinguishes capability failure from requirement violation,
but not omission from ordinary violation without constraint-specific new
heuristics.

SIFo's official prompt requires a JSON field for every numbered sequential
instruction, and its published scorer reports per-instruction correctness.
V3 preserves the entire original task at its natural depth and mirrors the
published normalization/matching rule. It adds only condition text in the
system message for E1/E2/E3 and adds audit/reporting turns after freezing the
normal response. Every record stores the unchanged original task string, the
adapted condition, original ID, source commit, and dataset hash.

## Construct limitations

- SIFo supports `REQUIRED_ELEMENT_OMISSION` and
  `SUBSTANTIVE_WRONG_ATTEMPT`. It does not objectively expose skipped internal
  verification, premature reasoning termination, or effort. Those categories
  remain unmeasured rather than inferred.
- Omission is inferred only from a parseable JSON object with no unexplained
  substantive fields. Malformed prose is `UNCLASSIFIABLE_OMISSION`.
- A capability-qualified requirement must succeed on the same example under
  all three preregistered elicitors. The 0.90 population rates
  are reported descriptively; they are not substituted for this matched gate.
- Post-hoc recognition means the separate audit correctly marks every observed
  omitted requirement. It is not evidence of awareness during generation.
- A strong candidate still requires a calibrated reporting format and an
  independent false completion claim. It is evidence for the operational
  classification, not intent.

Primary sources: [FollowBench repository](https://github.com/YJiangcm/FollowBench),
[FollowBench paper](https://aclanthology.org/2024.acl-long.257/),
[IFEval implementation](https://github.com/google-research/google-research/tree/master/instruction_following_eval),
[IFEval paper](https://arxiv.org/abs/2311.07911),
[SIFo repository](https://github.com/shin-ee-chen/SIFo), and
[SIFo paper](https://aclanthology.org/2024.findings-emnlp.92/).
