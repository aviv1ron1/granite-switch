# Workflow: Compose a Model

[← Quickstart](../quickstart.md) · Architecture: [Composer](../architecture/composer.md)

Composing produces a single checkpoint (base + embedded adapters + control tokens
+ chat template) that runs on both backends.

## Prerequisites

```bash
pip install "granite-switch[compose]"    # or: uv sync --extra compose
```

Optionally speed up downloads:

```bash
pip install "huggingface_hub[hf_transfer]"
huggingface-cli login
export HF_HUB_ENABLE_HF_TRANSFER=1
```

## Basic compose

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-core-r1.0 \
             ibm-granite/granitelib-rag-r1.0 \
             ibm-granite/granitelib-guardian-r1.0 \
  --output ./my-model
```

Use the [adapter function catalog / composer UI](https://generative-computing.github.io/granite-switch/adapter_catalog.html)
to browse adapters and generate a ready-to-run command.

## Common variations

```bash
# List what a library contains, don't build
--adapters ibm-granite/granitelib-rag-r1.0 --list-adapters

# Include only specific adapters (fnmatch globs)
--adapters ibm-granite/granitelib-rag-r1.0 --include-adapters answerability 'query_*'

# Exclude one adapter
--adapters ibm-granite/granitelib-guardian-r1.0 --exclude-adapters factuality-detection

# Only aLoRA adapters
--adapters ibm-granite/granitelib-rag-r1.0 --technology-filter alora

# Built-in empty-LoRA slots (placeholders you can train later)
--built-in-adapters base --lora-rank 8
```

Full flag table: [architecture/composer.md](../architecture/composer.md#cli-reference-key-flags).

## What lands in the output directory

- `config.json` — `GraniteSwitchConfig` (num_adapters, token ids, ranks, …)
- `model.safetensors*` — base weights (fused) + stacked adapter LoRA weights
- `tokenizer*` — with added control tokens and an adapter-aware chat template
- `adapter_index.json` — name → control token id → io schema mapping
- `io_configs/<adapter>/io.yaml` — per-adapter input/output contracts
- build report / model card (from `composer/reporting/`)

## Programmatic API

```python
from granite_switch.composer import GraniteSwitchComposer

model = GraniteSwitchComposer.from_base_and_adapters(
    base_model_name_or_path="ibm-granite/granite-4.1-3b",
    adapter_paths=["/path/to/adapter_a", "/path/to/adapter_b"],
    adapter_token_ids=[100265, 100266],
    adapter_substitute_token_ids=[100264, 100264],
    adapter_names=["adapter_a", "adapter_b"],
)
model.save_pretrained("./my-model")
```

The pipeline steps are documented in
[architecture/composer.md](../architecture/composer.md#the-compose-pipeline).

## Skip composition

Use a pre-composed checkpoint from Hugging Face:

- `ibm-granite/granite-switch-4.1-3b-preview`
- `ibm-granite/granite-switch-4.1-8b-preview`
- `ibm-granite/granite-switch-4.1-30b-preview`

## Bring your own adapter

See the tutorial guide
[`tutorials/guides/build_your_own_adapter.md`](../../tutorials/guides/build_your_own_adapter.md)
— train a LoRA/aLoRA against a Granite base, then compose it like any other.

## Validation

Every compose run finishes with `validate_all_parameters` (in
`composer/validator.py`). End-to-end tests always build through the composer
(never hand-assemble config) — see CLAUDE.md gotcha #5 and
`tests/composer/test_compose_e2e.py`.
