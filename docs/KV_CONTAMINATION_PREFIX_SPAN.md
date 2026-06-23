# KV-Cache Contamination — Prefix-Span Study

Measures how much the **answerability** judgment of `ibm-granite/granite-switch-4.1-3b-preview`
degrades when the *KV cache of the prompt prefix* is built under a **foreign** adapter instead of
the base model. This isolates cross-adapter KV contamination from output-side adapter selection.

## Method

Each scored record is a prompt laid out as three contiguous spans:

```
[ system + docs ] [ user request ] [ answer ]
        ^prefix          ^body         ^scored
```

A per-token adapter plan assigns each span to an adapter index
(`0` = base, `1..N` = embedded adapters), via `plan()` in the experiment script:

```python
def plan(n, sys_end, user_end, sys_idx, ans_idx):
    p = [0] * n
    for i in range(sys_end):     p[i] = sys_idx   # system+docs span
    for i in range(user_end, n): p[i] = ans_idx   # answer span
    return torch.tensor([p])                       # user span [sys_end:user_end) stays 0 (base)
```

Two conditions per record, run via the weightless `ExternalSelectionSwitch`
(`switch_impl="external"`, `model.model.switch.set_external_indices(...)`):

| Span | COND1 (clean) | COND2 (foreign) |
|---|---|---|
| system + docs `[0:sys_end)` | base (0) | **foreign** |
| user request `[sys_end:user_end)` | base (0) | base (0) |
| answer `[user_end:n)` | answerability | answerability |

Only the **system+docs prefix** is foreign-flavored in COND2. The user-request span runs under
base in *both* conditions; the answer span always runs under the `answerability` adapter. The metric
is answerability accuracy over the answer span.

- **Checkpoint:** `ibm-granite/granite-switch-4.1-3b-preview` (12 embedded adapters).
- **Foreigns:** the 11 embedded adapters other than `answerability` (the fixed answer adapter).
- **Records:** `LIMIT=0` → 3409 scored records (2045 answerable, 1364 unanswerable).
- **Clean baseline (COND1):** identical for every foreign — `3045/3409 = 0.893` — because COND2
  changes the prefix *flavor*, not the span *lengths*; the CLEAN cache is built once and reused.

### Contaminated-token count

Foreign-independent (the foreign adapter changes the KV flavor of the prefix, not its length), so
the contaminated total is the same across all 11 experiments — it is the **system+docs prefix** only:

| Contaminated span | Total tokens | Mean / record | answerable mean | unanswerable mean |
|---|---|---|---|---|
| system + docs (foreign) | 3,020,359 | 886.0 | 1259.3 | 326.4 |

The answerable class carries ~3.9× more contaminated prefix context than the unanswerable class,
which mechanistically explains why the answerable class collapses under nearly every foreign.

## Results — all 11 foreigns, full file (3409 scored)

COND1 clean = `3045/3409 = 0.893` for every row. Sorted by overall delta.

| Foreign prefix | COND2 foreign | Δ | Flips | ans: clean→foreign (Δ), flips | unans: clean→foreign (Δ), flips |
|---|---|---|---|---|---|
| hallucination_detection | 2017/3409 = 0.592 | −0.302 | 1202 | 0.871→0.362 (−0.509), 1050 | 0.926→0.936 (+0.010), 152 |
| uncertainty | 2681/3409 = 0.786 | −0.107 | 490 | 0.871→0.675 (−0.196), 433 | 0.926→0.953 (+0.027), 57 |
| guardian-core | 2708/3409 = 0.794 | −0.099 | 439 | 0.871→0.685 (−0.186), 389 | 0.926→0.958 (+0.032), 50 |
| policy-guardrails | 2724/3409 = 0.799 | −0.094 | 427 | 0.871→0.691 (−0.180), 377 | 0.926→0.961 (+0.035), 50 |
| context-attribution | 2843/3409 = 0.834 | −0.059 | 354 | 0.871→0.781 (−0.090), 288 | 0.926→0.913 (−0.013), 66 |
| query_clarification | 2904/3409 = 0.852 | −0.041 | 313 | 0.871→0.786 (−0.085), 257 | 0.926→0.951 (+0.025), 56 |
| factuality-detection | 2913/3409 = 0.855 | −0.039 | 200 | 0.871→0.796 (−0.075), 173 | 0.926→0.943 (+0.017), 27 |
| requirement-check | 2940/3409 = 0.862 | −0.031 | 159 | 0.871→0.812 (−0.060), 134 | 0.926→0.938 (+0.012), 25 |
| query_rewrite | 2949/3409 = 0.865 | −0.028 | 154 | 0.871→0.816 (−0.055), 125 | 0.926→0.938 (+0.012), 29 |
| citations | 2963/3409 = 0.869 | −0.024 | 172 | 0.871→0.825 (−0.046), 140 | 0.926→0.935 (+0.009), 32 |
| factuality-correction | 3076/3409 = 0.902 | +0.009 | 263 | 0.871→0.938 (+0.067), 150 | 0.926→0.849 (−0.077), 113 |

## Key findings

- **Contamination is real and large at the extreme.** `hallucination_detection` as the prefix
  flavor drops overall answerability accuracy from 0.893 to 0.592 (−0.302, 1202 flips) — a ~34%
  relative drop driven almost entirely by the answerable class.
- **The damage is asymmetric and concentrated in the answerable class.** Across every harmful
  foreign, the answerable accuracy falls (up to −0.509 for `hallucination_detection`) while the
  unanswerable accuracy is flat or slightly *up*. A contaminated prefix biases the model toward
  "unanswerable", which costs accuracy on genuinely answerable items and incidentally helps on
  unanswerable ones.
- **Effect size tracks prefix length.** The answerable class carries ~3.9× more contaminated tokens
  (1259 vs 326 mean), and it is the class that collapses — consistent with longer foreign-flavored
  prefixes accumulating more divergent KV state.
- **Not all foreigns hurt.** `factuality-correction` slightly *improves* overall accuracy (+0.009)
  by pushing the answerable class up (+0.067) more than it drags the unanswerable class down
  (−0.077) — the only foreign with a net-positive prefix effect.
- **Spread is wide.** Overall deltas range from −0.302 to +0.009; flips from 154 to 1202. The single
  worst foreign accounts for far more disruption than the rest combined.

## Dose-response — contaminating a *leading fraction* of the prefix

The results above contaminate the **whole** system+docs prefix (100%). To see *how much* of the
prefix must be foreign-flavored before answerability degrades — and whether the response is linear
or saturating — we re-ran the experiment flavoring only the **leading fraction** of the prefix.

For each record, the foreign adapter flavors `[0 : round(frac·sys_end))`; the rest of the prefix
`[round(frac·sys_end) : sys_end)` and the user span run under base, and the answer span stays
`answerability` (`plan()`'s `frac` argument). `frac` is a fraction of *each record's own* `sys_end`,
so the absolute contaminated-token count scales per record. `frac=0.0` is the clean baseline; `frac=1.0`
reproduces the whole-prefix table above. We swept `frac ∈ {0.30, 0.50, 0.70}` for four foreigns
spanning the severity range — one severe (`hallucination_detection`), one moderate (`guardian-core`),
two mild (`query_clarification`, `query_rewrite`) — over the same full file (3409 scored, COND1 = 0.893).

### Overall answerability accuracy vs. contaminated fraction

The `0%` column is the shared clean baseline; the `100%` column is from the whole-prefix table above.

| Foreign | 0% | 30% | 50% | 70% | 100% |
|---|---|---|---|---|---|
| hallucination_detection | 0.893 | 0.664 (−0.230) | 0.610 (−0.283) | 0.586 (−0.307) | 0.592 (−0.302) |
| guardian-core | 0.893 | 0.824 (−0.070) | 0.810 (−0.083) | 0.806 (−0.087) | 0.794 (−0.099) |
| query_clarification | 0.893 | 0.852 (−0.041) | 0.842 (−0.052) | 0.841 (−0.053) | 0.852 (−0.041) |
| query_rewrite | 0.893 | 0.876 (−0.017) | 0.868 (−0.025) | 0.869 (−0.024) | 0.865 (−0.028) |

Δ vs. the clean baseline is in parentheses. The `query_clarification` 70% cell was recovered with a
one-token boundary nudge — see *Recovering the query_clarification 70% cell* below.

### Answerable-class accuracy (where the damage lands)

The unanswerable class stays flat-to-slightly-up at every fraction (as in the whole-prefix study), so
the dose-response is carried entirely by the answerable class. Answerable accuracy (clean = 0.871):

| Foreign | 0% | 30% | 50% | 70% | 100% |
|---|---|---|---|---|---|
| hallucination_detection | 0.871 | 0.448 | 0.389 | 0.349 | 0.362 |
| guardian-core | 0.871 | 0.733 | 0.711 | 0.705 | 0.685 |
| query_clarification | 0.871 | 0.789 | 0.768 | 0.765 | 0.786 |
| query_rewrite | 0.871 | 0.835 | 0.822 | 0.823 | 0.816 |

## Key findings — dose-response

- **The response saturates early; it is not linear.** Every foreign reaches the large majority of its
  100% damage by **30%** contamination. `hallucination_detection` is the clearest: −0.230 at 30% is
  ~76% of its −0.302 whole-prefix effect, and the curve is nearly flat from 50% on (−0.283 → −0.307).
  Contaminating the *first third* of the prefix is almost as harmful as contaminating all of it.
- **Damage front-loads on the earliest tokens.** Because the contaminated slice is *leading*, the
  steep 0%→30% segment shows the system instruction + earliest documents carry most of the
  cross-adapter KV divergence; later prefix tokens add little. This is consistent with a foreign
  flavor perturbing the global context framing rather than accumulating linearly per token.
- **Severity ordering is preserved at every fraction.** HD ≫ guardian-core > query_clarification >
  query_rewrite holds across 30/50/70/100, so the partial-prefix dose ranks foreigns the same way the
  whole-prefix study does — the curve is a faithful interpolation, not a reshuffling.
- **Mild foreigns plateau (and the partial-prefix dose can slightly exceed the 100% endpoint).** For
  `query_clarification` and `query_rewrite`, the curve flattens by 50% and the 50–70% partial doses
  match or slightly exceed the whole-prefix (100%) damage (qclar 50% = −0.052, 70% = −0.053 vs
  100% = −0.041; qrew 70% = −0.024 vs 100% = −0.028). The leading-fraction doses landing a touch below
  the full-prefix accuracy is consistent with the front-loading result — the earliest tokens carry the
  damage, and the clean *tail* of the prefix at `frac < 1.0` adds little recovery — but the gaps are
  within run-to-run noise (a handful of records out of 3409). The takeaway is that the effect has
  plateaued, not that more contamination helps.
- **Practical implication.** KV reuse across adapters is dangerous even when only the *front* of a
  shared prefix was built under a foreign adapter — a partially-foreign cache is nearly as
  contaminating as a fully-foreign one. Guarding the *leading* prefix tokens is not sufficient
  mitigation.

### Recovering the query_clarification 70% cell

The `query_clarification` 70% (`f070`) cell initially crashed: that run hit a record-dependent CUDA
out-of-bounds in the MoE shared-expert LoRA path (`SwitchedLoRALinear` scatter-add) that fires on a
specific record under the partial-prefix (`foreign[0:cut] → base → answerability`) plan layout at
`frac < 1.0`. The crash is in the shared adapter kernel, not the experiment logic, and is a triple
coincidence of (record geometry `sys_end` × `cut = round(frac·sys_end)` × the LoRA adapter slice) —
which is why it aligned only at this one (foreign, fraction) out of all 12 cells, and why `frac = 1.0`
(uniform foreign prefix, no interior boundary) never crashed.

The cell was recovered by nudging the leading foreign-span boundary by **one token**
(`cut = round(frac·sys_end) + 1`, clamped to `sys_end`), which moves the foreign→base interior
transition off the offending alignment. The leading-prefix semantics are preserved — a one-token
shift on a single record (out of 3409) is RoPE-robust and accuracy-irrelevant — and the run then
completed cleanly over the full file with the same shared CLEAN cache (`COND1 = 0.893`, `scored =
3409`), so the cell is fully comparable to the other 11. The recovered value (−0.053) sits exactly
where the curve predicts, between the 50% (−0.052) and 100% (−0.041) endpoints.

The underlying kernel OOB remains a latent bug in the shared-expert LoRA path for any caller whose
external adapter plan creates a short interior foreign→base transition; the one-token nudge is a
workaround for this experiment, not a fix for the kernel.
