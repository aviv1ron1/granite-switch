# Running Granite Switch in the browser (ONNX / transformers.js)

This documents the ONNX export backend (`granite_switch.onnx`) and browser
runtime (`web/`) that let a composed Granite Switch model run client-side via
ONNX Runtime Web.

## Why a new backend

Granite Switch is a custom architecture (`model_type: granite_switch`) with
per-token LoRA adapter switching. transformers.js dispatches on `model_type`
against a fixed built-in list and has **no custom-architecture / `trust_remote_code`
mechanism** — `AutoModelForCausalLM.from_pretrained` on a `granite_switch` config
fails outright with `Unsupported model type: granite_switch`. The fix: export the
entire forward into self-contained ONNX graphs. This is a third backend alongside
`hf/` and `vllm/`.

There are two ways to run the exported graphs in JS, both shipped in `web/`:

1. **On transformers.js** (`web/src/granite-switch-tfjs.js`). The export writes a
   `tfjs/` layout whose `config.json` declares a *supported* DecoderOnly
   `model_type` (`gpt2`) and places the decode graph at `onnx/model.onnx`.
   `@huggingface/transformers`'s `AutoModelForCausalLM.from_pretrained` then loads
   the config and **creates/owns the ONNX `InferenceSession` through its own
   backend**, returning a transformers.js `PreTrainedModel`. Our decode loop runs
   over that transformers.js-owned session, threading the switch state.
2. **On raw onnxruntime-web** (`web/src/granite-switch.js`), reusing
   `@huggingface/transformers` only for `Tensor` / tokenization. Useful when you
   don't want to present a surrogate `model_type`.

Both produce identical output (verified against the Python golden).

## The two hard problems and how they were solved

### 1. Data-dependent control flow → branchless dataflow

`SwitchedLoRALinear` / `MergedSwitchedLoRALinear` (`hf/core/lora.py`) select
adapters with a Python loop over `adapter_indices.unique()` and a `torch.any`
early-exit — untraceable by `torch.onnx.export`. The `Onnx*` variants in
`onnx/export_modules.py` rewrite this branchlessly:

1. Prepend a **zero adapter row** at index 0, so `adapter_index == 0` (base)
   gathers a zero matrix → exact-zero delta, no branch needed.
2. `index_select` (Gather) the per-token A/B matrices by adapter index.
3. Batched matmul `(x · Aᵀ) · Bᵀ`.

The exported graph uses only Gather / MatMul / Where / Softmax / CumSum — all
supported by onnxruntime-web on WASM and WebGPU. (GATE 1: 436-node graph, no
`If`/`Loop`/custom ops.)

### 2. Cumulative switch state across KV-cached decode

The switch selects adapters via causal attention over **all prior** control
tokens, so single-token decode must carry that history. The switch's K/V only
carry signal in one channel, so `onnx/decode.py` threads a compact
`switch_key0` / `switch_val0` (`[batch, len]`) as explicit graph state, grown
each step exactly like the per-layer KV cache. Decode is then numerically
identical to a full prefill across a control token (GATE 3).

## Two non-obvious export bugs (worth remembering)

- **SDPA GQA path.** The HF SDPA backend lowers to
  `aten.scaled_dot_product_attention(enable_gqa=True)`; the ONNX converter
  mis-handles that when `q_heads == kv_heads` (no real grouping) — wrong
  numerics, or an outright assert with a `None` mask. Fix: `reskin_for_export`
  forces `_attn_implementation = "eager"`.
- **Attribute side-effects.** The shared MLP receives `adapter_indices` via a
  Python attribute (`_set_shared_mlp_context`), which `torch.export` drops. Fix:
  `OnnxGraniteSwitchAttentionDecoderLayer` inlines the MLP and threads
  `adapter_indices` as a real argument.

## Validation gates

| Gate | Check | Where |
|------|-------|-------|
| 1 | Full forward exports to a clean ONNX graph (no If/Loop/custom ops) | export succeeds |
| 2 | Exported prefill logits match HF backend (allclose) | `tests/onnx/test_onnx_parity.py::test_onnx_prefill_matches_hf` |
| 3 | KV-cached step-by-step decode matches full HF forward across a control token | `tests/onnx/test_onnx_parity.py::test_onnx_decode_matches_prefill` |
| 4 | Raw onnxruntime-web greedy decode matches the Python golden | `web/test/validate_node.mjs` + `web/example/index.html` |
| 4b | Loaded + run **ON transformers.js**, output matches the Python golden | `web/test/validate_tfjs.mjs` + `web/example/index-tfjs.html` |
| 5 | **Real 4 B model** (`granite-switch-4.1-3b-preview`) export matches HF backend | manual: export + `onnxruntime` vs HF logits |
| 5b | Real 4 B model **runs ON transformers.js** (Node), greedy decode matches HF golden | `web/test/validate_tfjs_real.mjs` (`GS_REAL_DIR=...`) |

Parity is `allclose`, not bit-exact: the branchless gather+matmul reorders the
LoRA reduction vs the HF per-adapter loop (consistent with CLAUDE.md §9, which
claims bit-exact only for the fused/vLLM path).

**GATE 5 (real model).** The tiny fixture proves the *shape* of the export; the
production check is the real 4.15 B `ibm-granite/granite-switch-4.1-3b-preview`
(12 real adapters, ranks 16/32, dense — no MoE/mamba layers active). Exported
fp32 prefill vs the HF backend on a prompt containing a control token:
`max|diff| = 9.5e-5`, `argmax agreement = 100%`. The branchless switch+LoRA
export is numerically correct at production scale, not just on the fixture.

**GATE 5b (real model on transformers.js).** The real 4 B model is then loaded
**on transformers.js** in Node exactly as GATE 4b — `AutoModelForCausalLM.from_
pretrained` returns a `GPT2LMHeadModel` and transformers.js creates + owns the
ONNX `InferenceSession`, resolving the 16.6 GB external `model.onnx.data`
sidecar through its onnxruntime-node backend. Greedy decode over that
transformers.js-owned session (with our switch-state threading) matches the HF
golden token-for-token, including the citations control token. So the *same*
loading path proven on the fixture scales to the production checkpoint.

## Large models and quantization

A 4 B model in fp32 is ~16.6 GB of weights — far over ONNX's 2 GiB protobuf
cap, so it cannot embed weights inline. `export_model_dir` auto-detects this
(by parameter byte size) and keeps an **external `<name>.onnx.data` sidecar**
instead (`--no-embed` forces it; `--embed` forces inline and errors past the
cap). `onnxruntime` (Python/Node) resolves the adjacent sidecar automatically;
`onnxruntime-web` needs the sidecar filename registered with its WASM FS.

For browser delivery, `granite_switch.onnx.quantize` shrinks the weights
(activations stay float, so the switch's adapter-selection arithmetic is
untouched):

- **`int8`** — `quantize_dynamic` (per-channel). ~4x smaller; robust default,
  broadly supported by onnxruntime-web on WASM.
- **`q4`** — `MatMulNBitsQuantizer` (4-bit block). ~8x smaller; the scheme
  transformers.js ships as `model_q4.onnx`.

Both are post-training and weight-only (no calibration data). Emit them via the
export CLI: `--quantize int8 --quantize q4` writes `*_int8.onnx` / `*_q4.onnx`
alongside the fp32 graphs and mirrors the quantized decode graph into
`tfjs/onnx/model_<scheme>.onnx` (selectable from transformers.js via `dtype`).

## Usage

See [`web/README.md`](../web/README.md) for export and run instructions.

## Current limitations

- Dense Granite only (MoE export not yet wired — the export decoder layer
  asserts `not has_experts`). The real `granite-switch-4.1-3b-preview` is dense
  (`num_local_experts = 0`), so this is not a blocker for it.
- Greedy decode only; sampling + tokenizer/chat-template integration in JS are
  the remaining productionization steps.
- Quantization is validated numerically; end-to-end browser load of the
  quantized 4 B model (WASM FS sidecar registration + WebGPU) is the final
  delivery step.
