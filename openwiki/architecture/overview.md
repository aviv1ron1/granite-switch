# Architecture Overview

[← Back to Quickstart](../quickstart.md)

Granite Switch has three cooperating subsystems that share one weight format:

```
                    ┌─────────────────────────────────────┐
                    │            composer/                 │
   base model  +    │  discover → load → transfer weights  │  →  checkpoint
   adapters         │  → tokenizer/chat template → validate│     (safetensors +
                    └─────────────────────────────────────┘      config.json +
                                     │                            adapter_index.json)
                                     │  same checkpoint, no conversion
                     ┌───────────────┴───────────────┐
                     ▼                                ▼
             ┌───────────────┐               ┌───────────────┐
             │   hf/ backend │               │  vllm/ backend│
             │ (prototyping, │               │  (production, │
             │   training)   │               │ Punica kernels)│
             └───────────────┘               └───────────────┘
```

- **[`composer/`](composer.md)** builds checkpoints (requires the `compose` extra).
- **[`hf/`](hf-backend.md)** is a `transformers`-native model for prototyping and
  training (requires `hf` extra).
- **[`vllm/`](vllm-backend.md)** is a vLLM plugin for fast serving (requires `vllm`).

The single source of truth for the model shape is
[`GraniteSwitchConfig`](configuration.md) in `src/granite_switch/config.py`,
shared by all three.

---

## Runtime data flow (inference)

Both backends implement the same logical forward pass. In the HuggingFace
backend (`src/granite_switch/hf/modeling_granite_switch.py`) it looks like this:

```
input_ids ──► SingleSwitch ──► (adapter_indices, modified_input_ids)
                  │                       │
                  │  detect control       │  control tokens rewritten
                  │  tokens, emit          │  to substitute ids
                  │  per-token index       ▼
                  │                   embed_tokens(modified_input_ids)
                  │                        │
                  └──── adapter_indices ───┤
                                           ▼
                              decoder layers (attention + shared_mlp)
                                           │
                            SwitchedLoRALinear applies adapter i's
                            LoRA weights to tokens where index == i
                                           ▼
                                       logits
```

Key files:
- Switch: `src/granite_switch/hf/switch/single.py` → [switch.md](switch.md)
- Token exchange logic: same file + config → [token-exchange.md](token-exchange.md)
- LoRA application: `src/granite_switch/hf/core/lora.py` → [lora.md](lora.md)
- Model wiring: `src/granite_switch/hf/modeling_granite_switch.py` (see
  `GraniteSwitchModel.forward`, around line 232)

---

## Why this design (the "why")

Standard LoRA trains each adapter against *its own* KV distribution. Switching
adapters mid-flow (as in a multi-step RAG pipeline) forces the KV cache to be
discarded and recomputed at each step — the dominant latency cost.

Granite Switch's adapters are all trained against a **common normalized KV
cache**, so:

1. They coexist in one checkpoint without cross-contamination.
2. The base prefill is reused across adapter switches (the aLoRA speedup —
   see the live-race telemetry linked from the README).
3. Adapter functions can be developed, benchmarked, and swapped independently,
   like software libraries.

The **token-exchange** mechanism (replacing a commit `e7b9330`, "Replace
control-token KV hiding with token-exchange") is what keeps the KV cache clean:
control tokens are rewritten to benign substitutes before the decoder embeds
them, so the decoder never stores a foreign token's key/value.

---

## Index-numbering convention (important)

There are two different conventions in the codebase — a frequent source of bugs:

| Context | "no adapter" | adapter N |
|---|---|---|
| Control-token / `adapter_indices` (HF & switch) | `0` | `N` (1-indexed) |
| vLLM Punica kernels (internal) | `-1` | `N-1` (0-indexed) |

The vLLM backend converts via `adapter_indices - 1` internally. See
[vllm-backend.md](vllm-backend.md).

`SingleSwitch` cannot transition *back* to base mid-sequence — once an adapter
activates it stays active for the rest of the sequence (documented in
`config.py` and `switch/single.py`).

---

Continue to: [The Switch](switch.md) · [Token exchange](token-exchange.md) ·
[LoRA layers](lora.md) · [Composer](composer.md)
