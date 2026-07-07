# The Switch (`SingleSwitch`)

[← Overview](overview.md)

**Source:** `src/granite_switch/hf/switch/single.py` (HF),
`src/granite_switch/vllm/switch/single.py` (vLLM).

The switch is the component that turns a token sequence into a per-token
**adapter selection**. It is intentionally tiny (a single attention head with a
single active dimension) so it adds negligible cost.

## What it computes

Given `input_ids` and the configured `adapter_token_ids`, the switch returns:

```python
(adapter_indices, modified_input_ids)
#  adapter_indices:    [batch, seq_len]  0 = base, 1+ = adapter N
#  modified_input_ids: [batch, seq_len]  control tokens rewritten to substitutes
```

## How it works — attention as a cumulative "latch"

The clever trick: instead of a learned module, the switch encodes adapter
selection into hand-built Q/K/V vectors and lets a standard causal attention
backend (SDPA, FlashAttention, eager) do the work. Only **dimension 0** of each
`head_dim`-wide vector carries signal; the rest is zero padding to satisfy the
backend's shape constraints.

| Token type | key[0] | query[0] | value[0] |
|---|---|---|---|
| Control token for adapter N | `+gain` | `1` | `N` (adapter_id) |
| Any other token | `-gain` | `1` | `0` |

Because `Q·K = 1 × (±gain) = ±gain` regardless of `head_dim`, causal softmax
attention makes each position attend overwhelmingly to the **most recent control
token at or before it**. The attended `value[0]` is that control token's
adapter id — so the output is effectively "which adapter is active at this
position", latched forward until the next control token.

`control_token_gain` (default `15.0`, in `GraniteSwitchConfig`) controls how
sharp this selection is. Higher gain → crisper one-hot attention. The
`switch_head_dim` / `projection_head_dim` matches the decoder's native head dim
so the switch shares the same KV cache layout.

> The gain value is validated for numerical sharpness — see
> `tests/unit/test_sharpness_equivalence.py` and
> `tests/shared/single_switch_cases.py`.

## Why "no return to base"

The latch only flips *up* when it sees a control token; there is no token that
resets to base. This is a deliberate simplification documented in both
`config.py` and the switch docstring:

> "SingleSwitch has no mechanism to transition back to base mid-sequence."

For the composed-model use case (activate an adapter, generate its structured
output) this is exactly what's wanted.

## KV caching

The switch participates in the model's `Cache` object like a real layer. It is
assigned `layer_idx` (0 in HF, with decoder layers following) so its keys/values
are stored and reused during incremental decoding. In vLLM the switch is a real
`Attention` module discovered separately for KV allocation — which is why
`register()` in `vllm/__init__.py` subtracts 1 from the layer count for pipeline
parallelism slicing (`_GraniteSwitchArchConfigConvertor.get_num_hidden_layers`).

## Token exchange half

Besides selection, the switch rewrites control tokens to substitutes using a
lookup table (`control_to_substitute_lut`) built at construction from
`adapter_token_ids` → `adapter_substitute_token_ids`. This is covered in detail
in [token-exchange.md](token-exchange.md).

## Tests

- `tests/hf/test_single_switch.py` — stress tests across attention backends
  (auto-detects which backends work on the platform; see CLAUDE.md gotcha #6).
- `tests/vllm/test_single_switch.py` — vLLM equivalence.
- `tests/shared/single_switch_cases.py` — parametrized shared cases.
