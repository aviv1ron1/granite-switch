# Concepts & Glossary

[← Quickstart](../quickstart.md)

## Adapter function

A **LoRA adapter trained to a specific input/output contract** — a score, a
decision, a rewritten query — with the output schema enforced at the token level
(by [Mellea](https://mellea.ai)). The "function" framing matters: each has a
known signature (typed inputs/outputs), not just free-form text. This is what
makes them *composable as software*. Contracts ship as `io.yaml` files, embedded
into the checkpoint under `io_configs/<name>/`.

## LoRA vs Activated LoRA (aLoRA)

| | Standard LoRA | Activated LoRA (aLoRA) |
|---|---|---|
| KV distribution | trained against its **own** KV cache | trained against a **common normalized** KV cache |
| Switching adapters mid-flow | must discard + recompute KV cache | reuses shared base prefill |
| Multiple in one checkpoint | interfere / need joint training | coexist, activate on demand |
| Latency in multi-step flows | high (recompute) | low (prefill reuse) |

aLoRA is IBM's technology that makes many adapters share one KV cache and
activate on demand — the reason Granite Switch can serve many capabilities from
one deployment with no memory/latency overhead. The live-race telemetry in the
README quantifies this (74% vs 29% KV hit rate on a multi-step RAG pipeline).

## Control token

A dedicated special token per adapter (e.g. `<guardian>`, `<query_rewrite>`).
**Placing it in the input activates that adapter** from its position forward. The
switch detects it and emits per-token `adapter_indices`. All control tokens are
freely generatable — there is no runtime suppression (CLAUDE.md gotcha #2).

## Substitute token / token exchange

Because a control token isn't part of the base model's normal distribution,
storing its key/value would pollute the shared KV cache. The switch **rewrites**
each control token to a benign *substitute* id before the decoder embeds it, so
the cache stays clean while `adapter_indices` (from the original ids) still
drives LoRA selection. Full detail:
[architecture/token-exchange.md](../architecture/token-exchange.md).

## Adapter index convention

- `0` = base / no adapter; `1+` = adapter N (control-token & switch convention).
- vLLM Punica kernels internally use `-1` = no adapter, 0-indexed adapters.
- The switch cannot return to base mid-sequence — once up, an adapter stays
  active for the rest of the sequence.

## The Switch (`SingleSwitch`)

A single-head attention module with one active dimension that latches the most
recent control token's adapter id forward via causal attention. Tiny and
learning-free. See [architecture/switch.md](../architecture/switch.md).

## Skinning

Producing a Granite Switch checkpoint from a base model with **zero adapters**
(or just structure) — the base model re-expressed in the Switch architecture
(fused projections, switch layer machinery). Used to verify the architecture
itself is equivalent to the base model. See the skinning-equivalence tests
(`tests/composer/test_skinning_equivalence.py`,
`tests/composer/test_arch_skinning.py`).

## Adapter library

A Hugging Face repo containing multiple adapters (e.g.
`ibm-granite/granitelib-rag-r1.0`), each in a subdirectory with its own
`adapter_config.json` + `io.yaml`. The composer discovers, filters, and
selectively downloads them.

## Composition

Combining a base model + adapter functions into one checkpoint — analogous to
statically linking object code. Adapters can be developed and benchmarked
independently, then composed and swapped/upgraded without retraining.

## Fused projections

The HF and vLLM backends both fuse Q/K/V into `qkv_proj` and gate+up into
`shared_input_linear`. This changes float reduction order vs upstream HF's
separate projections — so HF is not bit-exact with upstream; vLLM equivalence is
authoritative (CLAUDE.md gotcha #9).

## The Granite multipliers

Granite differs from Llama via `attention_multiplier`, `logits_scaling` (~8.0),
`residual_multiplier`, and `embedding_multiplier`. Always read from config, never
hardcode. See [architecture/configuration.md](../architecture/configuration.md).
