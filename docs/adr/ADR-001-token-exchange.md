# ADR-001: Replace control-token KV hiding with token exchange

**Status:** Accepted ([PR #34](https://github.com/generative-computing/granite-switch/pull/34), merged 2026-05-19)

**Closes:** #8

## Context

The legacy adapter-control mechanism padded every Q/K/V tensor in every decoder layer
with `control_dims=32` extra dimensions and branded each control token's K with
`finfo.min` so attention couldn't read it. The padding was needed because the base
model had never seen control tokens during training — without hiding, those tokens'
K vectors would distort attention from real tokens.

Costs of that approach on Granite 4.1-8b:

- KV cache `head_dim` expanded from native 128 → 160, padded to 192.
- FlashAttention's fast path required native head dims, so the padded vectors fell
  back to a slower kernel.
- ~20% extra KV memory and ~33% extra attention compute per layer.
- A `hidden_count` position-correction pass (see legacy gotcha in CLAUDE.md history)
  to keep RoPE aligned despite the hidden tokens.
- Config carried `control_dims`, `hiding_groups`, `hiding_policy`,
  `adapter_third_party`, `expanded_head_dim`, `num_hiding_groups`, and several
  helper accessors to manage the scheme.

The dimensions only existed to hide the control tokens; they cost real compute on
every forward pass for every request, regardless of whether any control tokens were
present.

## Decision

Replace KV hiding with **token exchange**: at the switch layer, rewrite each control
token's id in `input_ids` to a substitute id whose embedding the model was already
trained to handle. The decoder embeds the rewritten ids once and runs natively — no
Q/K/V padding, no KV cache expansion, no hidden-count correction.

**Two paired layers:**

1. **Chat template** (compose-time, `composer/tokenizer_setup.py`): a skip-once Jinja
   flag (`ns.skip_next_start_of_role`) suppresses the role-marker token that would
   normally follow each control token. `alora_pass2` drops the first character of
   in-message ALoRA invocation text (BPE-equivalent to dropping one tokenized piece).
   The rendered sequence is one token shorter than before.

2. **Switch** (runtime, `hf/switch/single.py` + `vllm/switch/single.py`): when emitting
   `adapter_indices`, also rewrite each control token's id via a
   `control_to_substitute_lut` buffer. Returns `(adapter_indices, modified_input_ids)`.
   The decoder unpacks and embeds `modified_input_ids`.

**Substitute derivation:**

- **ALoRA** → first token of `alora_invocation_tokens` (read from the adapter's
  `adapter_config.json`).
- **LoRA / built-in** → whatever the tokenizer's chat template emits at the start of
  a no-adapter user turn. Derived at compose time by
  `_probe_lora_substitute_token_id(tokenizer)` — render a minimal probe chat,
  tokenize, take `input_ids[0]`. On Granite 4.x this resolves to `<|start_of_role|>`
  (id 100264).

The legacy KV-hiding path was removed entirely, not gated. `_expand_with_control_dimensions`
deleted from both backends. `control_dims`, `hiding_groups`, `hiding_policy`,
`adapter_third_party`, `expanded_head_dim`, `num_hiding_groups`,
`get_hiding_group_token_ids`, `get_third_party_adapter_mask`, and
`get_adapter_hiding_policy_matrix` removed from config. `adapter_substitute_token_ids`
is now required when `num_adapters > 0`.

## Consequences

**Positive:**

- KV cache `head_dim` returns to native 128 (from 192). FlashAttention runs on the
  native vectors.
- ~20% less KV memory, ~33% less attention compute per layer on Granite 4.1-8b.
- No retraining required — substitute ids are already embeddings the model handles.
- ~3000 LoC deleted net (config helpers, dual-path expand/contract logic, hiding
  reports, position-correction tests).
- `hidden_count` RoPE-offset gotcha is gone; positions are simply correct.

**Negative / breaking:**

- Checkpoints composed under the legacy scheme cannot load. `from_pretrained` raises
  `ValueError` from the new `adapter_substitute_token_ids is required` validator.
  Users must recompose with the current `compose_granite_switch.py` against the same
  base + adapter sources (minutes per checkpoint).
- The chat template now contains compose-time logic (skip-once flag, alora_pass2)
  that downstream tooling rendering the template by hand needs to respect.
- Substitute-id derivation depends on the tokenizer's chat template behavior at
  compose time. A future tokenizer change that alters the no-adapter turn opener
  would shift the LoRA substitute id and require recomposing.

## Alternatives considered

**Minimize `control_dims` from 32 to 1** (Solution B in [issue #8](https://github.com/generative-computing/granite-switch/issues/8)).
The legacy scheme allocated 32 extra Q/K/V dimensions, but only `num_hiding_groups`
(typically 1) were ever non-zero. Setting `control_dims = num_hiding_groups` would
have cut the padding overhead substantially without any chat-template or switch-layer
changes. Rejected: it reduces the cost but doesn't eliminate it — KV cache still grows
(`head_dim + 1` instead of native), FlashAttention still falls off its fast path on
non-native head dims, and the `hidden_count` RoPE-correction gotcha stays. Token
exchange achieves zero overhead at the same engineering cost, so the fallback wasn't
worth shipping.
