# Configuration: `GraniteSwitchConfig`

[← Overview](overview.md)

**Source:** `src/granite_switch/config.py`.

`GraniteSwitchConfig` is the single source of truth for model shape, shared by
all three subsystems. It **extends** `transformers.GraniteMoeHybridConfig`, so
every base-Granite parameter is available plus the switch/adapter fields below.

```python
from granite_switch import GraniteSwitchConfig   # or granite_switch.config
```

`model_type = "granite_switch"` — this is what triggers auto-registration and
vLLM plugin dispatch.

## Switch / adapter parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `num_adapters` | int | `0` | Number of *real* LoRA adapters (index 0 = base). `0` = plain skinning |
| `adapter_token_ids` | list[int] | `None` | Control token id per adapter; `[i]` activates adapter `i+1`. Must be unique |
| `adapter_substitute_token_ids` | list[int] | `None` | Substitute id per adapter for [token exchange](token-exchange.md). Required when `num_adapters > 0` |
| `adapter_names` | list[str] | `None` | Ordered names for name→index mapping |
| `adapter_ranks` | list[int] | `None` | Per-adapter LoRA rank; length must equal `num_adapters` |
| `max_lora_rank` | int | `8` | Max rank across adapters (allocation). Must equal `max(adapter_ranks)` |
| `lora_target_modules` | list[str] | auto | Module *groups* to apply LoRA to (see below) |
| `control_token_gain` | float | `15.0` | Attention gain for control/non-control separation in the switch |
| `switch_head_dim` | int | `32` | Q/K/V dim in switch attention (aligned to decoder head dim at runtime) |
| `fused_add_norm` | bool | `False` | vLLM residual-norm convention (for bit-exact skinning equivalence) |

`projection_head_dim` is computed (`head_dim` or `hidden_size //
num_attention_heads`) and used by the switch and vLLM head-size convertor. It is
*not* set as `head_dim` because HF's RoPE also reads that field.

## `lora_target_modules` groups

Auto-derived when `None` and `num_adapters > 0`:

| Group | Wraps |
|---|---|
| `qkv_proj` | fused Q/K/V |
| `o_proj` | attention output |
| `shared_input_linear` | shared_mlp fused gate+up |
| `shared_output_linear` | shared_mlp down |

## Validation rules (enforced in `__init__`)

- `num_adapters >= 0`.
- If `num_adapters > 0`:
  - `adapter_token_ids` length == `num_adapters`, all unique.
  - `adapter_substitute_token_ids` required, length == `num_adapters`, all `>= 0`.
  - `adapter_token_ids` required whenever substitutes are given.
  - `adapter_ranks` required, length == `num_adapters`,
    `max(adapter_ranks) == max_lora_rank`.
- `layer_types` defaults to all `"attention"`, length == `num_hidden_layers`
  (which includes the switch's index-0 slot when adapters are present, so
  `DynamicCache` pre-allocation matches).

## Example (from CLAUDE.md)

```json
{
  "model_type": "granite_switch",
  "architectures": ["GraniteSwitchForCausalLM"],
  "num_adapters": 4,
  "adapter_token_ids": [100, 101, 102, 103],
  "adapter_substitute_token_ids": [100264, 100264, 100264, 100264],
  "adapter_names": ["adapter_0", "adapter_1", "adapter_2", "adapter_3"],
  "max_lora_rank": 8,
  "adapter_ranks": [8, 8, 8, 8],
  "switch_head_dim": 32,
  "control_token_gain": 15.0
}
```

> Note: the CLAUDE.md example predates the token-exchange rewrite and still
> mentions `hiding_groups`/`hiding_policy`/`control_dims`. Those KV-hiding fields
> were removed in commit `e7b9330`; the current config uses
> `adapter_substitute_token_ids` + `control_token_gain` instead.

## Granite-specific multipliers — never hardcode

Inherited from the base config; always load from config:

- `attention_multiplier` — attention score scaling (instead of `1/sqrt(head_dim)`)
- `logits_scaling` — applied to final logits (~8.0; main diff vs Llama)
- `residual_multiplier` — applied to residual connections
- `embedding_multiplier` — applied to input embeddings (used in
  `GraniteSwitchModel.forward`: `inputs_embeds * self.embedding_multiplier`)

## Tests

`tests/unit/test_config.py`, `tests/unit/test_config_edge_cases.py`.
