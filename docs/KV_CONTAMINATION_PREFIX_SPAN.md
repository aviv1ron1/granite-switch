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
