# Granite Switch — browser runtime (ONNX Runtime Web)

Run a composed **Granite Switch** model — base Granite + per-token LoRA adapter
switching — entirely client-side in the browser, via raw
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). No inference
server; no transformers.js `pipeline()` (it has no custom-architecture API for
`model_type: granite_switch`).

## How it works

The full Granite Switch forward — adapter selection from control tokens, the
control-token rewrite, branchless stacked-LoRA, the base decoder, and the LM
head — is exported into two self-contained ONNX graphs:

- `prefill.onnx` — processes the whole prompt.
- `decode.onnx` — one token at a time, with KV cache **plus** the switch's
  cumulative adapter-selection state threaded across steps
  (`past_switch_key0` / `past_switch_val0`). This is what keeps adapter
  selection correct after the prompt.

`src/granite-switch.js` drives those graphs with a greedy decode loop. It is
engine-agnostic: pass `onnxruntime-web` in the browser, or `onnxruntime-node`
headlessly.

## Export a model

```bash
pip install -e ".[onnx]"
python -m granite_switch.onnx.export <composed-checkpoint> --output web/example/model
```

This writes `prefill.onnx`, `decode.onnx`, `gs_onnx.json` (runtime metadata),
and a `tfjs/` layout (for loading **on** transformers.js).

**Weight layout is size-aware.** Small models embed their weights inline (a
single `.onnx`, simplest for the browser). Models past ONNX's 2 GiB protobuf
cap — e.g. the real 4 B `granite-switch-4.1-3b-preview` (~16.6 GB fp32) — keep
weights in an external `<name>.onnx.data` sidecar. `--embed` / `--no-embed`
force the choice.

**Quantize for the browser** (weight-only, no calibration data):

```bash
python -m granite_switch.onnx.export <ckpt> --output out --no-embed \
  --quantize int8 --quantize q4
```

`int8` (~4x smaller, onnxruntime-web friendly) and `q4` (~8x, transformers.js's
`model_q4` scheme) are written alongside the fp32 graphs and mirrored into
`tfjs/onnx/model_<scheme>.onnx`. The branchless switch/LoRA dataflow stays
float, so adapter selection is unaffected.

**Which dtype?** On the real 4 B `granite-switch-4.1-3b-preview`, **`int8`
reproduces the fp32 greedy decode token-for-token** (~6 GB), while **`q4`
diverges** after a few tokens (its weight perturbation flips argmax — plausible
but not golden). So `int8` is the verified-correct browser default here; `q4`
trades correctness for size and should be spot-checked per model.

## Package for the Hugging Face Hub

`granite_switch.onnx.package` turns an export dir into an upload-ready repo
(`config.json` + `gs_onnx.json` + `onnx/model[_int8|_q4].onnx` + sidecars +
tokenizer + a model card):

```bash
python -m granite_switch.onnx.package out ./hf-repo \
  --tokenizer-src <composed-checkpoint> \
  --repo-id your-org/granite-switch-onnx-web
# then:
huggingface-cli upload your-org/granite-switch-onnx-web ./hf-repo . --repo-type model
```

The repo loads **from the Hub on transformers.js** — see
`example/index-hub.html`:

```js
const meta = await (await fetch(
  "https://huggingface.co/your-org/REPO/resolve/main/gs_onnx.json")).json();
const gs = await GraniteSwitchTfjs.load({ modelId: "your-org/REPO", meta, dtype: "int8" });
const tokens = await gs.generate([10, 20, 30, /*control*/ 100352, 40], 16);
```

**Browser size reality (tested).** A 4 B model does **not** load in a standard
browser tab. Verified in Chrome: the graph, config, metadata, and WASM runtime
fetch fine, but loading the **6.4 GB int8 weight sidecar aborts** with an
onnxruntime-web WASM out-of-memory error — it exceeds the **wasm32 ~4 GB
address-space limit**. Requesting `device: "webgpu"` does not rescue it, because
onnxruntime-web still stages external-data weights through WASM memory. fp32
(16.6 GB) is even further out of reach.

So:
- **Tiny fixture** → runs in any tab (verified; both onnxruntime-web and
  transformers.js paths).
- **4 B model** → loads + runs correctly in **Node** (onnxruntime-node, no 4 GB
  ceiling) and is upload-ready for the Hub, but is **too large for an in-tab
  WASM session**. A browser deployment needs either a **smaller composed
  Granite Switch** (fewer / lower-rank adapters, smaller base) or a genuinely
  sub-4 GB quantization (note: `q4` here already trades away decode
  correctness, so it is not a free win).

## Run

**Browser:** serve `web/` over HTTP and open `example/index.html`:

```bash
python3 -m http.server 8731 --directory web
# open http://localhost:8731/example/index.html
```

**Headless (Node, same ORT engine):**

```bash
cd web && npm install && npm run validate
```

`npm run validate` runs `test/validate_node.mjs`, which greedy-decodes a prompt
containing an adapter control token and asserts the tokens match the Python
golden continuation (this is the GATE 4 check).

## Status / limitations

- Numeric parity with the HuggingFace backend is proven in `tests/onnx/` (tiny
  CPU fixture) **and on the real 4 B `granite-switch-4.1-3b-preview`**
  (max|diff| 9.5e-5, argmax 100% — GATE 5).
- Dense Granite only — MoE export is not yet wired (the export decoder layer
  asserts `not has_experts`). The real 4 B model is dense, so this is not a
  blocker for it.
- Large-model external-data layout and int8/q4 quantization are implemented;
  end-to-end browser load of the quantized 4 B model (WASM-FS sidecar
  registration + WebGPU) is the remaining delivery step.
- Greedy decode only; sampling (temperature/top-k/top-p via
  `@huggingface/transformers` `LogitsProcessor`) and tokenizer/chat-template
  integration are the remaining productionization steps.
