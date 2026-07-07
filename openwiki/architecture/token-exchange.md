# Token Exchange

[← Overview](overview.md) · [The Switch](switch.md)

Token exchange is the mechanism that keeps the shared KV cache clean when
control tokens fire adapters. It **replaced** an earlier "control-token KV
hiding" approach (commit `e7b9330`, PR #34) with something simpler and more
robust.

## The problem

Each adapter function is activated by inserting a special **control token**
(e.g. `<guardian>`) into the input. But that token is *not* part of the base
model's normal vocabulary distribution — if the decoder embeds it and stores its
key/value in the KV cache, it pollutes the shared cache and can leak across
requests or interfere with other adapters. Adapters are trained against a
*normalized* KV cache that never contains these control tokens.

## The solution

The switch does two things in one pass (`SingleSwitch.forward`):

1. **Selection** — detect control tokens, emit `adapter_indices`
   (which adapter is active per token). See [switch.md](switch.md).
2. **Rewrite** — replace each control token id with a benign **substitute id**
   *before the decoder embeds the sequence*.

```
input_ids:          [ ..., <guardian>=100265, "text", ... ]
                                 │  switch rewrites via LUT
modified_input_ids: [ ..., <|start_of_role|>=100264, "text", ... ]
                                 │
                    embed_tokens(modified_input_ids)  ← decoder is oblivious
```

The decoder embeds `modified_input_ids` and stores *substitute* keys/values in
the cache — indistinguishable from a no-adapter render. Meanwhile
`adapter_indices` (computed from the *original* ids) still tells the LoRA layers
which adapter to apply. Selection and cache state are decoupled.

## The lookup table

Built once at switch construction from config
(`src/granite_switch/hf/switch/single.py`):

```python
lut = torch.full((vocab_size,), -1)          # -1 = not a control token
for ctrl_id, sub_id in zip(adapter_token_ids,
                           adapter_substitute_token_ids):
    lut[ctrl_id] = sub_id
self.register_buffer("control_to_substitute_lut", lut)
```

There is **no decoder-side LUT**, no per-forward scatter, no clone guard — the
switch hands the decoder finished `input_ids`. This simplicity is the whole
point of the token-exchange rewrite.

## Choosing the substitute id

The substitute must be a token that would *naturally* occupy that position in a
no-adapter render, so the post-swap sequence looks normal to the model.

- **ALoRA adapters**: the control token sits mid-sequence (before the invocation
  sequence), so its substitute is chosen accordingly.
- **LoRA adapters**: the control token is prepended at sequence position 0. The
  composer *probes the tokenizer* to find what token normally appears at
  position 0 of a rendered chat — for Granite 4.x this is `<|start_of_role|>`
  (id `100264`). See `_probe_lora_substitute_token_id` in
  `composer/compose_granite_switch.py` (well-documented there).

This probe avoids hard-coding any Granite-specific token string; it reads the
constant out of the tokenizer's own chat template at compose time.
`tests/composer/test_lora_substitute_probe.py` pins this Granite 4.x behavior.

## Config contract

`GraniteSwitchConfig` enforces (see `config.py`):

- `adapter_substitute_token_ids` is **required** when `num_adapters > 0`.
- Its length must equal `num_adapters`.
- All ids must be `>= 0` (real token ids).
- `adapter_token_ids` must be present and unique (duplicates would collapse LUT
  slots).

## Known limitation

When position 0 is a control token in a hiding group (a LoRA prefix token with
`add_bos_token=False`), the hidden count is off by 1, causing a 1-position RoPE
offset. This is acceptable — adapter detection stays exact and RoPE is robust to
small positional shifts (CLAUDE.md gotcha #7).

## Tests

- `tests/unit/test_token_exchange.py`
- `tests/hf/test_token_exchange.py`
- `tests/vllm/test_token_exchange.py`
