# Supported Models

[← Quickstart](../quickstart.md)

Source of truth: [`docs/SUPPORTED_MODELS.md`](../../docs/SUPPORTED_MODELS.md).
Architecture is detected automatically from the base model's HF
`config.model_type` field (`composer/arch.py` → `resolve_arch`).

## Feature support

| Family | `model_type` | Support | KV cache handling |
|---|---|---|:---:|
| Granite 4.x Dense | `granite` | **Full** — primary target | Token exchange |

The composer resolves architecture via descriptors in `composer/arch.py`:

- `granite_dense_arch` → `model_type: granite`
- `granite_moe_hybrid_arch` → `model_type: granitemoehybrid`

Any Granite model whose config reports `model_type: granite` can serve as a base.

## Example base models

| Model tag | Size | Variant |
|---|---|---|
| `ibm-granite/granite-4.1-3b` | 3B | Dense, instruct (**default base**) |
| `ibm-granite/granite-4.1-8b` | 8B | Dense, instruct |
| `ibm-granite/granite-4.0-micro` | 3B | Dense, instruct |

Base (non-instruct) variants like `granite-4.1-3b-base` are also supported.

> Currently **single-GPU inference** is the primary supported path. Models that
> don't fit in one GPU's memory are not yet supported.

## Pre-composed checkpoints

| Checkpoint | Size |
|---|---|
| `ibm-granite/granite-switch-4.1-3b-preview` | 3B |
| `ibm-granite/granite-switch-4.1-8b-preview` | 8B |
| `ibm-granite/granite-switch-4.1-30b-preview` | 30B |

## Target layers (base → Switch parameter paths)

### Attention

| Base PEFT modules | Switch group | Parameter path |
|---|---|---|
| `q_proj`/`k_proj`/`v_proj` (fused) | `qkv_proj` | `model.layers.{i}.self_attn.qkv_proj.lora_{A,B}_slices.{0,1,2}` |
| `o_proj` | `o_proj` | `model.layers.{i}.self_attn.o_proj.lora_{A,B}` |

### MLP (remapped to `shared_mlp` namespace)

| Base PEFT modules | Switch group | Parameter path |
|---|---|---|
| `gate_proj`+`up_proj` (fused) | `shared_input_linear` | `model.layers.{i}.shared_mlp.input_linear.lora_{A,B}_slices.{0,1}` |
| `down_proj` | `shared_output_linear` | `model.layers.{i}.shared_mlp.output_linear.lora_{A,B}` |

The remapping from upstream PEFT names to these paths is done by
`AdapterRemapper` (`composer/weight_remapper.py`).

## Adapter libraries to compose

Published under
[ibm-granite/granite-libraries](https://huggingface.co/collections/ibm-granite/granite-libraries):

- `ibm-granite/granitelib-core-r1.0` — core capabilities
- `ibm-granite/granitelib-rag-r1.0` — RAG (answerability, citations, query rewrite, …)
- `ibm-granite/granitelib-guardian-r1.0` — safety / guardian checks

Browse all with benchmarks in the
[adapter function catalog](https://generative-computing.github.io/granite-switch/adapter_catalog.html).
