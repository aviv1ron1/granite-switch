# Granite Switch — OpenWiki

> Build AI models like you build software: pick adapter functions, compose them
> into one checkpoint, deploy with one command.

Granite Switch is a model architecture + toolchain for embedding **multiple LoRA
adapters** (called *adapter functions*) into a single Granite base model. The
result is one deployable checkpoint that serves many capabilities — RAG, safety,
factuality, query rewriting — with no per-adapter memory or latency overhead,
thanks to IBM's **Activated LoRA (aLoRA)** technology and a shared KV cache.

This wiki explains **what** the system does, **why** it is built the way it is,
and **where** to look in the code.

---

## What this repository is

`granite-switch` is a single Python package (`granite_switch`) with optional
extras for different backends. It does three things:

1. **Compose** — Combine a base Granite model + several LoRA/aLoRA adapters into
   one checkpoint that has control tokens, a chat template, and stacked adapter
   weights. → [`composer/`](architecture/composer.md)
2. **Run (HuggingFace)** — A `transformers`-native model class for prototyping,
   training, and CPU/GPU inference. → [`hf/`](architecture/hf-backend.md)
3. **Run (vLLM)** — A production inference backend using Punica LoRA kernels and
   PagedAttention, registered as a vLLM plugin. → [`vllm/`](architecture/vllm-backend.md)

Both backends read the **same checkpoint** with no conversion step.

---

## The core idea in 60 seconds

- Each adapter function has a dedicated **control token** (e.g. `<guardian>`).
  Placing that token in the input sequence activates the adapter from that
  position forward.
- A tiny attention-based **switch** (see [`SingleSwitch`](architecture/switch.md))
  reads `input_ids`, detects control tokens, and emits a per-token
  `adapter_indices` tensor (`0` = base, `1+` = adapter N).
- The switch also performs **token exchange**: it rewrites each control token id
  to a benign *substitute* id before the decoder embeds the sequence, so the
  decoder never sees a foreign token and the KV cache stays clean.
- LoRA layers ([`SwitchedLoRALinear`](architecture/lora.md)) apply the right
  adapter's weights per token based on `adapter_indices`.
- Because all adapters are trained against the **same normalized KV cache**, the
  base prefill is reused across adapter switches instead of recomputed — the key
  aLoRA speedup.

Read the full mechanism in [architecture/token-exchange.md](architecture/token-exchange.md).

---

## Install

Uses [uv](https://docs.astral.sh/uv/) for local dev; `pip` for end users.

```bash
# End users
pip install "granite-switch[vllm]"      # production inference
pip install "granite-switch[hf]"        # HuggingFace prototyping
pip install "granite-switch[compose]"   # build/compose checkpoints
pip install "granite-switch[dev]"       # everything

# Local development (this repo)
uv sync --extra dev
```

Requires Python 3.11–3.13, PyTorch 2.10+, transformers 5.5–5.9.
See [workflows/development.md](workflows/development.md) for the vLLM 0.19 vs 0.20
version story (CUDA compatibility).

---

## Three common tasks

### 1. Compose a checkpoint

```bash
python -m granite_switch.composer.compose_granite_switch \
  --base-model ibm-granite/granite-4.1-3b \
  --adapters ibm-granite/granitelib-core-r1.0 \
             ibm-granite/granitelib-rag-r1.0 \
             ibm-granite/granitelib-guardian-r1.0 \
  --output ./my-model
```

Full CLI + pipeline: [workflows/compose-a-model.md](workflows/compose-a-model.md).

### 2. Serve with vLLM + Mellea (recommended)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ibm-granite/granite-switch-4.1-3b-preview --port 8000
```

Then call typed adapter functions through [Mellea](https://mellea.ai). See
[workflows/inference.md](workflows/inference.md).

### 3. Prototype with HuggingFace

```python
from granite_switch.hf import GraniteSwitchForCausalLM
model = GraniteSwitchForCausalLM.from_pretrained("./my-model")
```

---

## Documentation map

### Architecture
- [Overview & data flow](architecture/overview.md) — how the pieces connect
- [The Switch (`SingleSwitch`)](architecture/switch.md) — attention-based adapter selection
- [Token exchange](architecture/token-exchange.md) — the KV-cache-safe control mechanism
- [LoRA layers](architecture/lora.md) — `SwitchedLoRALinear` and per-token weight application
- [Composer](architecture/composer.md) — how checkpoints are built
- [HuggingFace backend](architecture/hf-backend.md)
- [vLLM backend](architecture/vllm-backend.md)
- [Configuration](architecture/configuration.md) — `GraniteSwitchConfig` reference

### Workflows
- [Compose a model](workflows/compose-a-model.md)
- [Run inference](workflows/inference.md)
- [Development & testing](workflows/development.md)

### Domain
- [Concepts & glossary](domain/concepts.md) — adapter functions, aLoRA vs LoRA, control tokens
- [Supported models](domain/supported-models.md)

---

## Ecosystem context

Granite Switch is one layer of a coordinated stack:

| Component | Role |
|---|---|
| [Granite models](https://huggingface.co/ibm-granite) | Base models (3B/8B/30B) |
| [Granite Libraries](https://huggingface.co/collections/ibm-granite/granite-libraries) | Pre-trained adapter functions to compose |
| **Granite Switch** (this repo) | Architecture + composer that embeds adapters |
| [Mellea](https://mellea.ai) | Orchestrates adapter functions as typed calls with constrained decoding |

Sources: [`README.md`](../README.md), [`pyproject.toml`](../pyproject.toml),
[`CLAUDE.md`](../CLAUDE.md).
