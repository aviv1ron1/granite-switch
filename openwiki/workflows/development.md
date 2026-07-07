# Workflow: Development & Testing

[← Quickstart](../quickstart.md)

## Setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # core only (config)
uv sync --extra hf         # HuggingFace backend
uv sync --extra vllm       # vLLM backend (0.19.1 default)
uv sync --extra compose    # composer tools
uv sync --extra dev        # everything
```

Requires Python 3.11–3.13, PyTorch 2.10+, transformers 5.5–5.9.

### vLLM version conflict (important)

`pyproject.toml` defines two mutually exclusive vLLM tracks
(`[tool.uv].conflicts`):

| Extra / group | vLLM | Requires |
|---|---|---|
| `vllm` / `vllm19` (default) | 0.19.1 | CUDA 12.x-compatible |
| `vllm20` / `dev-vllm20` | 0.20+ | CUDA 13.0+ (PyTorch 2.11) |

Default is vLLM 0.19.1 because 0.20 pulls CUDA 13 which breaks many CUDA 12.x
environments. Use the `vllm20` variants only if your driver supports CUDA 13+.

## Package import paths

```python
from granite_switch import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM, SingleSwitch
from granite_switch.hf.core.lora import SwitchedLoRALinear
from granite_switch.vllm import register
from granite_switch.composer import GraniteSwitchComposer
```

## Running tests

Always use `-v -s --tb=short` (and `-x` to fail fast). Check the GPU first —
hardware can change between sessions:

```bash
python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU only')"
```

Run **incrementally by directory**, fastest first — never the whole suite as one
command:

```bash
# 1. Unit (fastest, CPU)
pytest tests/unit/ -v -s --tb=short -x

# 2. HF (CPU), by file
pytest tests/hf/test_single_switch.py -v -s --tb=short -x
pytest tests/hf/test_model_forward.py -v -s --tb=short -x

# 3. vLLM (GPU required), by file
pytest tests/vllm/test_single_switch.py -v -s --tb=short -x
pytest tests/vllm/test_model_forward.py -v -s --tb=short -x

# 4. Composer (CPU)
pytest tests/composer/ -v -s --tb=short -x

# 5. Integration (slowest, GPU)
pytest tests/integration/ -v -s --tb=short -x

# Debug a pattern
pytest tests/ -k "token_exchange" -v -s --tb=short -x
```

### pytest markers (`pyproject.toml`)

| Marker | Meaning |
|---|---|
| `gpu` | requires CUDA GPU |
| `vllm` | requires vLLM installed |
| `slow` | > 30s |
| `deep` | expensive code-theory tests (m=8 / 256-dim); run with `pytest -m deep` |
| `requires_model` | needs a real model checkpoint |

Default `addopts` ignore `tests/_legacy` and exclude `deep` (`-m "not deep"`).

### Test directory map

| Dir | Scope | Hardware |
|---|---|---|
| `tests/unit/` | config, token exchange, sharpness | CPU |
| `tests/hf/` | HF backend forward/generation/lora | CPU |
| `tests/vllm/` | vLLM forward, TP/PP, kernels, equivalence | GPU |
| `tests/composer/` | discovery, weight transfer, chat template, skinning | CPU |
| `tests/integration/` | cross-backend weights, e2e compose | GPU |
| `tests/shared/` | parametrized cases + utilities (imported, not run alone) | — |

Files prefixed `_` (e.g. `_lora_tests.py`, `_tp_integration_worker.py`) are
subprocess workers / shared test bodies, invoked by the matching `test_*.py`.

## Where to put throwaway scripts

Use `scratch/` (gitignored) for debug/diagnostic scripts — **never** `tests/`.
`pytest tests/` must only run curated, maintained tests (CLAUDE.md).

## File organization rules

- All `.md` docs go under `docs/` (exceptions: root `README.md`, `CLAUDE.md`, and
  this `openwiki/`).
- All `test_*.py` go under `tests/` in the right subdir.
- Scripts: `snake_case.py`; docs: `UPPER_CASE.md`.

## Git workflow

See [`docs/GIT_WORKFLOW.md`](../../docs/GIT_WORKFLOW.md). Quick reference:

- Branch: `feature/ticket-ID-description` or `bugfix/ticket-ID-description`.
- Flow: branch from `main` → develop → rebase → PR → merge → delete branch.
- Verify comments match code before committing.
- **Never sign commits as Claude** (project instruction).

License: Apache-2.0 (SPDX headers on every source file).
