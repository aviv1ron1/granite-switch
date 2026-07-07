# vLLM Backend

[← Overview](overview.md)

**Source:** `src/granite_switch/vllm/` (requires the `vllm` or `vllm20` extra).
**Purpose:** production inference — Punica LoRA kernels, PagedAttention,
continuous batching, tensor/pipeline parallelism.

## Layout

```
vllm/
├── __init__.py             # register() — vLLM plugin entry point
├── granite_switch_model.py # GraniteSwitchForCausalLM / GraniteSwitchModel (vLLM)
├── core/
│   ├── lora.py             # SwitchedLoRALinear (Punica kernels)
│   ├── lora_kernel_meta.py # kernel metadata / batching
│   └── decoder.py          # decoder layers
└── switch/single.py        # SingleSwitch (vLLM Attention)
```

## Plugin registration

Declared in `pyproject.toml`:

```toml
[project.entry-points."vllm.general_plugins"]
register_granite_switch = "granite_switch.vllm:register"
```

vLLM calls `register()` on startup. It is **re-entrant** and:

1. Registers `GraniteSwitchConfig` with transformers `AutoConfig`.
2. Registers a custom `_GraniteSwitchArchConfigConvertor` so vLLM sees:
   - `get_num_hidden_layers()` → **layer count minus 1** when adapters are
     present (the `SingleSwitch` occupies a KV-cache placeholder slot that vLLM
     discovers separately; PP slicing must count only physical decoder layers).
   - `get_head_size()` → `projection_head_dim` (token exchange does not expand
     the head dim, so it's just the base model's head dim).
3. Registers `GraniteSwitchForCausalLM` with `ModelRegistry` (only if not
   already present).

Verify registration:

```bash
python -c "from vllm.plugins import load_general_plugins; \
           from vllm import ModelRegistry; load_general_plugins(); \
           print('GraniteSwitchForCausalLM' in ModelRegistry.get_supported_archs())"
```

## Index convention (Punica)

vLLM's Punica kernels use `-1` for "no adapter" and 0-indexed adapters
internally. The backend converts the switch's 1-indexed `adapter_indices` via
`adapter_indices - 1`. See
[overview.md](overview.md#index-numbering-convention-important).

## Known limitations (from CLAUDE.md)

- **TP row-parallel bias doubling** (gotcha #8): `SwitchedLoRALinear`'s
  row-parallel bypass passes bias to all TP ranks; after all-reduce this doubles
  the bias. *Not triggered* by any Granite arch — Granite 4.0/4.1 use
  `attention_bias=False` and `mlp_bias=False`.
- Single-GPU inference is the primary path; multi-GPU (TP/PP) is tested but the
  supported-models doc notes models must fit constraints.

## vLLM version story

Two conflicting extras (see `pyproject.toml` `[tool.uv].conflicts`):

- `[vllm]` → vLLM 0.19.1 (**default**) — works with CUDA 12.x.
- `[vllm20]` → vLLM 0.20+ — requires CUDA 13.0+ (via PyTorch 2.11).

Pick based on your CUDA driver. Details in
[workflows/development.md](../workflows/development.md).

## Weight compatibility

A checkpoint saved by the HF backend (`model.save_pretrained(...)`) loads
directly into vLLM (`LLM(model="./checkpoint")`) — no conversion. Cross-backend
weight equivalence is guarded by `tests/integration/test_hf_to_vllm_weights.py`.

## Tests

`tests/vllm/` (all require a GPU): `test_model_forward.py`,
`test_generation_equivalence.py`, `test_single_switch.py`, `test_lora.py`,
`test_tp_lora.py`, `test_tp_integration.py`,
`test_pipeline_parallelism_generation.py`, `test_upstream_equivalence.py`,
`test_token_exchange.py`, `test_granite4_mini.py`, `test_granite4_fullsize.py`.
