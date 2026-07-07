# LoRA Layers

[← Overview](overview.md)

**Source:** `src/granite_switch/hf/core/lora.py` (HF),
`src/granite_switch/vllm/core/lora.py` (vLLM, Punica kernels).

LoRA layers are where the selected adapter's low-rank weights are actually
applied. Both backends implement the same math; only the kernel differs.

## The two classes

- **`SwitchedLoRALinear`** — wraps a single base `nn.Linear`. Stacks all
  adapters' `lora_A` / `lora_B` in tensors and applies the right one per token.
- **`MergedSwitchedLoRALinear`** — for **fused** projections (QKV, gate+up). The
  base weight is one fused matrix, but each fused slice has its own LoRA. Stores
  a `lora_A_slices` / `lora_B_slices` `ParameterList`.

> The HF backend uses **fused** QKV and gate-up projections to match the vLLM
> architecture. This means it is *not* bit-exact with upstream HuggingFace
> `GraniteMoeHybridForCausalLM` (which uses separate projections) — the fused
> reduction order changes floating point results. The vLLM skinning-equivalence
> tests are the authoritative check (CLAUDE.md gotcha #9).

## Per-token application (the core of `forward`)

```python
output = self.base_layer(x)          # dense base output, always computed
if adapter_indices is None or not torch.any(adapter_indices > 0):
    return output                    # fast path: pure base, no LoRA

for adapter_idx in active_adapters:  # unique adapters present in this batch
    token_mask   = adapter_indices_flat == adapter_idx
    tensor_idx   = adapter_idx - 1   # 1-indexed selection → 0-indexed tensor
    lora_a       = self.lora_A[tensor_idx, 0]   # [rank, in_features]
    lora_b       = self.lora_B[tensor_idx, 0]   # [out_features, rank]
    x_adapter    = x_flat[token_indices]
    delta        = (x_adapter @ lora_a.t()) @ lora_b.t()
    output_flat[token_indices] += delta
```

Key points:

- **Base output is always dense-computed**; LoRA is an *additive delta* applied
  only to tokens whose `adapter_indices > 0`.
- The loop is over the *unique adapters actually present* in the batch, so a
  batch that only touches one adapter pays for one.
- `lora_B` is **pre-scaled by `alpha/rank`** during weight loading (in the
  composer), not at runtime — so no scaling factor appears here. This is a
  deliberate optimization noted in the code (`lora.py` line ~58).
- `adapter_idx - 1` maps the 1-indexed selection convention to the 0-indexed
  weight tensor. (Contrast with vLLM Punica, which uses `-1` for "no adapter" —
  see [overview.md](overview.md#index-numbering-convention-important).)

## Context-passing pattern

Some call sites (like `shared_mlp` inside a `GraniteMoeHybrid` block) can't
thread `adapter_indices` explicitly through their forward signature. For those,
the decoder layer stashes indices on the module via `_adapter_indices`
(see `GraniteSwitchAttentionDecoderLayer._set_shared_mlp_context` in
`hf/modeling_granite_switch.py`), and `forward` falls back to that stored value
when its argument is `None`.

## Which modules get LoRA

Driven by `config.lora_target_modules` (module *group* names), auto-derived in
`GraniteSwitchConfig`:

| Group | What it wraps |
|---|---|
| `qkv_proj` | fused Q/K/V projection |
| `o_proj` | attention output projection |
| `shared_input_linear` | fused gate+up of `shared_mlp` |
| `shared_output_linear` | `shared_mlp` down projection |

See [domain/supported-models.md](../domain/supported-models.md) for the full
base-model → Switch parameter-path mapping.

## vLLM variant

`src/granite_switch/vllm/core/lora.py` implements the same selection but uses
**Punica kernels** (`lora_kernel_meta.py`) for batched low-rank matmuls and
supports tensor parallelism. See [vllm-backend.md](vllm-backend.md).

## Tests

- `tests/hf/test_lora.py`, `tests/vllm/test_lora.py`
- `tests/shared/lora_cases.py` (parametrized shared cases)
- `tests/vllm/test_tp_lora.py` (tensor-parallel LoRA)
