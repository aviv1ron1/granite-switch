# HuggingFace Backend

[← Overview](overview.md)

**Source:** `src/granite_switch/hf/` (requires the `hf` extra).
**Purpose:** prototyping, optional router training, and CPU/GPU inference with
full `transformers` integration.

## Layout

```
hf/
├── __init__.py                  # registers with AutoConfig / AutoModelForCausalLM
├── modeling_granite_switch.py   # GraniteSwitchForCausalLM, GraniteSwitchModel
├── core/lora.py                 # SwitchedLoRALinear, MergedSwitchedLoRALinear
└── switch/single.py             # SingleSwitch (attention-based selection)
```

## Auto-registration

Importing `granite_switch.hf` registers the model with transformers:

```python
AutoConfig.register("granite_switch", GraniteSwitchConfig)
AutoModelForCausalLM.register(GraniteSwitchConfig, GraniteSwitchForCausalLM)
```

So `AutoModelForCausalLM.from_pretrained("./my-model")` works once the package is
imported. Registration is wrapped in a try/except (safe if already registered).

## Model classes

- **`GraniteSwitchModel`** (base, extends `GraniteMoeHybridPreTrainedModel`):
  owns `embed_tokens`, the `switch` (`SingleSwitch`), and the decoder `layers`.
  Its `forward` (around line 232) runs the switch → embed → decoder flow from
  [overview.md](overview.md#runtime-data-flow-inference).
- **`GraniteSwitchForCausalLM`** adds the LM head and `GenerationMixin`.
- **`GraniteSwitchAttentionDecoderLayer`** wraps a Granite attention block +
  `shared_mlp`, threading `adapter_indices` into LoRA layers (and stashing them
  on `shared_mlp` via the context-passing pattern — see [lora.md](lora.md)).

`GraniteSwitchModel.forward` exposes `self._last_adapter_indices` for tests and
debugging.

## Fused projections & equivalence

The HF backend deliberately uses **fused** QKV and gate-up projections to mirror
the vLLM architecture. Consequence: it is **not bit-exact** with upstream
`GraniteMoeHybridForCausalLM` (separate projections → different float reduction
order). The HF skinning-equivalence tests in
`tests/composer/test_skinning_equivalence.py` are therefore skipped; the vLLM
equivalence tests are authoritative (CLAUDE.md gotcha #9).

## Attention-backend quirks

- The **eager** backend does NOT treat `attention_mask=None` as causal (it means
  "full attention"). SDPA and FlashAttention handle `None` correctly via the
  `is_causal` attribute (CLAUDE.md gotcha #6).
- `tests/hf/test_single_switch.py` auto-detects which backends work on the
  current platform (probes each with a `k=-inf` GQA call at import) and skips the
  rest.

## transformers version handling

`modeling_granite_switch.py` branches on `_TRANSFORMERS_GE_5_9` for the
`create_causal_mask` kwargs (`inputs_embeds` vs `input_embeds` +
`cache_position`). Supported range is transformers 5.5–5.9 (see
[configuration.md](configuration.md) / `pyproject.toml`).

## Tests

`tests/hf/` — `test_model_forward.py`, `test_generation.py`,
`test_single_switch.py`, `test_single_switch_e2e.py`, `test_lora.py`,
`test_token_exchange.py`, `test_qk_norm.py`, `test_quantization.py`,
`test_granite4_mini.py`, `test_granite4_fullsize.py`. All run on CPU.
