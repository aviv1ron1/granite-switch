# The Composer

[← Overview](overview.md)

**Source:** `src/granite_switch/composer/` (requires the `compose` extra).
**CLI entry point:** `python -m granite_switch.composer.compose_granite_switch`.

The composer combines a base Granite model with one or more LoRA/aLoRA adapters
into a single deployable checkpoint that works with both backends unchanged.

## Module responsibilities

| Module | Role |
|---|---|
| `compose_granite_switch.py` | CLI, `build()` / `save_and_validate_model_artifacts()`, orchestration |
| `compose_utils.py` | `GraniteSwitchComposer` — the programmatic API (`from_base_and_adapters`) |
| `adapter_discovery.py` | Find adapters in HF repos / local dirs / YAML manifests; filter; selective download |
| `adapter_loader.py` | Detect LoRA rank/alpha and which module groups are present |
| `arch.py` | Architecture descriptors (`granite_dense_arch`, `granite_moe_hybrid_arch`); resolve base model → module layout |
| `weight_transfer.py` | Transfer base weights (with fusion) and stack adapter weights |
| `weight_remapper.py` | `AdapterRemapper` — map upstream PEFT names → Switch parameter names |
| `tokenizer_setup.py` | Add control tokens, build the adapter-aware chat template |
| `validator.py` | Post-build parameter validation |
| `reporting/` | Model card, compose report, population table, adapter analysis |

## The compose pipeline

`GraniteSwitchComposer.from_base_and_adapters` (in `compose_utils.py`) runs these
steps — read the docstring there for the authoritative list:

1. **Resolve architecture** from the base model config (`resolve_arch`). Detects
   dense vs MoE-hybrid layout.
2. **Detect LoRA config** — per-adapter rank/alpha (`detect_lora_config`) and
   which module groups each adapter targets (`detect_present_modules`).
3. **Build `GraniteSwitchConfig`** — copy arch-required fields from base config,
   prepend one **switch layer** at index 0 (bumps `num_hidden_layers` by 1 when
   adapters are present), set switch/adapter params.
4. **Create model** (`GraniteSwitchForCausalLM`), cast to config dtype.
5. **Transfer base weights** (`transfer_base_weights`) with arch-driven fusion of
   Q/K/V and gate/up projections.
6. **Transfer adapter weights** (`transfer_adapter_weights`) — stack each
   adapter's `lora_A`/`lora_B` into the switched tensors, pre-scaling `lora_B` by
   `alpha/rank`.
7. **Validate** all parameters (`validate_all_parameters`).

The CLI (`compose_granite_switch.py`) additionally:
- Downloads base + adapters from the Hub (with **selective download** —
  `_build_allow_patterns` fetches only the needed adapter subdirs).
- Adds control tokens and configures the chat template (`tokenizer_setup.py`).
- Picks **substitute token ids** for token exchange (probing the tokenizer for
  LoRA prefix substitutes — see [token-exchange.md](token-exchange.md)).
- Copies each adapter's `io.yaml` into `io_configs/<name>/` and writes
  `adapter_index.json` (name → token id → io schema).
- Records the HF snapshot commit SHA per adapter for provenance.
- Emits a build report / model card via `reporting/`.

## CLI reference (key flags)

From `_compose_argparser()`:

| Flag | Purpose | Default |
|---|---|---|
| `--base-model` | Base Granite model (HF id or path) | `ibm-granite/granite-4.1-3b` |
| `--adapters` | Adapter repo ids / paths / YAML manifests | `[]` |
| `--output` | Output directory | `./granite-with-all-aloras` |
| `--include-adapters` | Only include names matching fnmatch globs | all |
| `--exclude-adapters` | Exclude names (applied after include) | none |
| `--technology-filter` | Keep only `alora` or `lora` adapters | both |
| `--technology` | Override the technology *label* (vs filter) | auto-detect |
| `--built-in-adapters` | Names for empty-LoRA placeholder slots | `[]` |
| `--lora-rank` / `--lora-alpha` | Rank/alpha for built-in slots | `8` / `=rank` |
| `--switch-head-dim` | Override switch attention head dim | from config |
| `--list-adapters` | List available adapters and exit (no build) | off |
| `--debug-fields` | Include `original_path` in `adapter_index.json` | off |

## Adapter discovery

`adapter_discovery.py` supports three adapter sources:

1. **HF repo id** — e.g. `ibm-granite/granitelib-rag-r1.0`. May be an *adapter
   library* (multiple adapters in subdirs) or a single adapter.
2. **Local path** — a directory with `adapter_config.json` + weights.
3. **YAML manifest** — declares adapters pointing to arbitrary locations.

`filter_adapters` applies include/exclude globs and technology filters. Selective
download (`_build_allow_patterns` + `resolve_repo_path`) fetches only the needed
files.

## Constraints from CLAUDE.md

- All end-to-end tests must build models through `GraniteSwitchComposer` — never
  hand-assemble `GraniteSwitchConfig` or call `transfer_base_weights` directly
  (gotcha #5). Extend the composer if it can't handle a case (e.g. zero-adapter
  skinning is supported — `adapter_paths=None`).

## Tests

`tests/composer/` — e.g. `test_compose_e2e.py`, `test_adapter_discovery`-related
files, `test_weight_remapper.py`, `test_tokenizer_setup.py`,
`test_chat_template.py`, `test_validator.py`, `test_skinning_equivalence.py`.
