# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**granite-switch** is a single Python package (`granite_switch`) for building and deploying Granite models with embedded LoRA adapters. Two backends share the same weight format: `granite_switch.hf` (HuggingFace, training) and `granite_switch.vllm` (production inference, 10-20x speedup via Punica kernels + PagedAttention).

## Project Structure

Key layout rules — full tree via `find src/` or `find tests/`:

- `src/granite_switch/` — unified package; `composer/`, `hf/`, `vllm/` match the optional extras
- `tests/` — official test suite only; subdirs: `unit/`, `hf/`, `vllm/`, `composer/`, `integration/`, `regression/`, `shared/`
- `scratch/` — gitignored; use this for throwaway diagnostic scripts (not `tests/`)
- `tutorials/` — notebooks and guides; see `tutorials/CLAUDE.md` for conventions

## Installation (local/dev)

```bash
pip install -e ".[dev]"         # everything (recommended for development)
pip install -e ".[hf,compose]"  # HF + composer only (no vLLM)
```

## File Organization Convention

**IMPORTANT:** Keep the repository organized by placing files in their designated directories.

### Documentation Files (Markdown)

**All `.md` documentation files MUST go in a `docs/` directory:**

- **Root-level docs (`docs/`)**: Cross-implementation documentation, guides, and architecture docs
- **Exceptions**: Only `CLAUDE.md` and `README.md` may be at the repository root

### Test Files (Python)

**All `test_*.py` test files MUST go in a `tests/` directory:**

- **`tests/unit/`**: Unit tests (fastest, CPU-only)
- **`tests/hf/`**: HuggingFace implementation tests
- **`tests/vllm/`**: vLLM implementation tests
- **`tests/composer/`**: Compose system tests
- **`tests/integration/`**: Cross-implementation and end-to-end integration tests
- **`tests/regression/`**: Regression tests (hf/, vllm/, integration/, shared/, tools/)
- **`tests/shared/`**: Shared test utilities and parametrized cases

**IMPORTANT: `tests/` is for official regression tests ONLY.** Do NOT place throwaway diagnostic,
debugging, or exploratory scripts in `tests/`. Use `scratch/` instead (it is gitignored). Running
`pytest tests/` should only execute curated, maintained tests — never one-off investigations.

### Naming Conventions

- **Test files**: `test_*.py`
- **Documentation**: `UPPER_CASE.md`
- **Scripts**: `snake_case.py`

## Development Commands

### Composing Models

```bash
python -m granite_switch.composer.compose_granite_switch \
  --adapters ibm-granite/granitelib-rag-r1.0
```

### Testing

**Always use `-v -s --tb=short`** when running tests. `-x` (fail fast) stops on the first failure —
no point running 200 more tests after something breaks.

**Check GPU availability first** — the underlying hardware can change between sessions:

```bash
python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU only')"
```

**Run tests incrementally by directory**, in order of speed — don't run the full suite as a
single command:

```bash
# 1. Unit tests first (fastest, CPU)
pytest tests/unit/ -v -s --tb=short -x

# 2. HF tests by file (CPU)
pytest tests/hf/test_single_switch.py -v -s --tb=short -x
pytest tests/hf/test_model_forward.py -v -s --tb=short -x

# 3. vLLM tests by file (GPU required)
pytest tests/vllm/test_single_switch.py -v -s --tb=short -x
pytest tests/vllm/test_model_forward.py -v -s --tb=short -x

# 4. Integration tests last (slowest, GPU required)
pytest tests/integration/ -v -s --tb=short -x
```

### vLLM Deployment

```bash
# Verify plugin registration
python -c "from vllm.plugins import load_general_plugins; \
           from vllm import ModelRegistry; \
           load_general_plugins(); \
           print('OK' if 'GraniteSwitchForCausalLM' in ModelRegistry.get_supported_archs() else 'FAIL')"

# Start API server
python -m vllm.entrypoints.openai.api_server \
  --model ./granite-with-all-aloras \
  --port 8000
```

## Key Configuration Parameters

- **`attention_multiplier`**: Attention score scaling (instead of `1/sqrt(head_dim)`)
- **`logits_scaling`**: Applied to final logits (main architectural difference with Llama)
- **`residual_multiplier`**: Applied to residual connections
- **`embedding_multiplier`**: Applied to input embeddings

Always use config values — never hardcode these parameters.

## Common Gotchas

### 1. Adapter Index Convention

**Control tokens**: `0` = no adapter, `1+` = adapter indices

**vLLM Punica kernels**: `-1` = no adapter (internal conversion: `adapter_indices - 1`)

### 2. Control Token Generatability

All control tokens are freely generatable — there is no runtime suppression. The
model can produce any control token during generation.

### 3. Chat Template Token Placement

- **ALORA adapters**: Token placed either in user message by matching invocation sequence or right before generation prompt
- **LORA adapters**: Token placed at sequence beginning

### 4. Granite vs Llama Differences

- Granite uses `logits_scaling` (typically 8.0)
- Custom attention scaling via `attention_multiplier`
- Different residual and embedding multipliers

Always load from config, never hardcode.

### 5. End-to-End Tests Must Use Compose Infrastructure

No test should manually assemble `GraniteSwitchConfig` or call `transfer_base_weights`
directly. All model construction must go through `GraniteSwitchComposer` so that the
compose pipeline itself is what's being tested. If the composer can't handle a use case
(e.g., zero-adapter skinning), extend the composer — don't work around it in tests.

### 6. HF Attention Backends and Causal Masking

The eager backend does NOT handle `attention_mask=None` as causal — it treats `None` as no mask
(full attention). SDPA and FlashAttention handle `attention_mask=None` correctly via `is_causal`
attribute on the module.

The HF stress tests (`tests/hf/test_single_switch.py`) auto-detect which attention backends work on the
current platform by probing each with a k=-inf GQA call at import time. Unavailable backends are skipped.

### 7. Known Limitation: Hidden Count Offset When Position 0 is in a Hiding Group

When position 0 is a control token in a hiding group (e.g., a LoRA prefix token with
`add_bos_token=False`), `hidden_count` is off by 1, causing a 1-position RoPE offset. This is
acceptable because adapter detection is exact and RoPE is robust to small positional shifts.

### 8. Known Limitation: TP Row-Parallel Bias Doubling

`SwitchedLoRALinear`'s row-parallel bypass path passes bias to all TP ranks instead of
suppressing it for rank > 0. After all-reduce this doubles the bias. Not affected: all Granite
architectures (4.0, 4.1) use `attention_bias=False` and `mlp_bias=False`.

### 9. HF Backend Uses Fused Projections (Not Bit-Exact with Upstream HF)

The GraniteSwitch HF backend uses fused QKV and gate-up projections, symmetric with the vLLM
backend architecture. Upstream HuggingFace `GraniteMoeHybridForCausalLM` uses separate projections.
Fused projections change the floating-point reduction order, so bit-exact skinning equivalence
with the upstream HF model is not achievable. The vLLM skinning equivalence tests are the
authoritative check — both the upstream and skinned models use the same fused-projection
architecture there. The HF skinning tests in `tests/composer/test_skinning_equivalence.py` are
skipped for this reason.

## Documentation

- `docs/ARCHITECTURE.md` - Architecture overview (control tokens, backends, SingleSwitch)
- `docs/GIT_WORKFLOW.md` - Git branching strategy and commit guidelines
- `docs/SUPPORTED_MODELS.md` - Model compatibility

## Git Workflow

**See [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) for complete git workflow guidelines.**

- **Branch naming**: `feature/ticket-ID-description` or `bugfix/ticket-ID-description`
- **Workflow**: Branch from `main` → develop → rebase → PR → merge → delete branch
- **Critical**: Always verify comments match code before committing (see GIT_WORKFLOW.md)
- **Commit format**: Clear summary + explanation of WHAT changed and WHY

When committing, **never sign as Claude** (per project instructions)

## License

Apache-2.0 (as indicated by SPDX headers in source files)
