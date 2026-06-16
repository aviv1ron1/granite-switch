# Granite Switch — browser runtime (ONNX Runtime Web)

Run a composed **Granite Switch** model — base Granite + per-token LoRA adapter
switching — entirely client-side in the browser, via raw
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). No inference
server.

## Two ways to load

There are two runtimes here, both proven against the Python golden:

1. **Native transformers.js** (`src/granite-switch-register.js` — recommended for
   apps). A self-registering shim teaches transformers.js the `granite_switch`
   architecture, so the model loads with the standard
   `AutoModelForCausalLM.from_pretrained` and transformers.js owns the ONNX
   session (and fetches the external-data sidecar in the browser). Requires a
   **bundler** (the shim deep-imports transformers.js internals the package
   `exports` map hides). See [`example-demo/`](example-demo/) for a shippable
   Vite app — verified loading the real 350 M model in a Chrome tab.

   ```js
   import { loadGraniteSwitch } from "./src/granite-switch-register.js";
   const gs = await loadGraniteSwitch("org/repo", { dtype: "fp32" });
   const seq = await gs.generate({ inputs, max_new_tokens: 16, do_sample: false });
   ```

   Node parity test: `npm run validate:native` (tiny fixture + real 350 M).

2. **Raw onnxruntime-web** (`src/granite-switch.js` — no bundler needed). Drives
   the exported graphs directly with a hand-written decode loop; this is the
   original path, documented in detail below.

The rest of this README covers the raw onnxruntime-web runtime. (transformers.js's
`pipeline()` still has no custom-architecture API; the shim works at the
`AutoModelForCausalLM` layer, not `pipeline()`.)

## How it works

The full Granite Switch forward — adapter selection from control tokens, the
control-token rewrite, branchless stacked-LoRA, the base decoder, and the LM
head — is exported into a **single** self-contained ONNX graph:

- `decode.onnx` — KV-cached, **plus** the switch's cumulative adapter-selection
  state threaded across steps (`past_switch_key0` / `past_switch_val0`), which is
  what keeps adapter selection correct after the prompt. Its `input_ids` seq axis
  is dynamic, so the SAME graph serves both the batched first pass over the whole
  prompt (`input_ids [1, N]` + an empty past) and each subsequent single-token
  decode step (`input_ids [1, 1]` + a non-empty past). There is no separate
  `prefill.onnx` — that would ship a duplicate copy of the same weights.

`src/granite-switch.js` drives this graph with a greedy decode loop. It is
engine-agnostic: pass `onnxruntime-web` in the browser, or `onnxruntime-node`
headlessly. (The browser runtime feeds the whole prompt in one pass with an empty
past; the Node validator instead replays the prompt token-by-token through the
same graph to build state — same graph, fewer `session.run` calls in the browser.)

`src/granite-switch-tokenizer.js` wraps `@huggingface/transformers`'
`AutoTokenizer` + the model's `chat_template.jinja` so you can go from text to
token ids (firing the adapter control token) and back — see
[Tokenizer / chat template](#tokenizer--chat-template).

## External data (the browser weight-loading requirement)

A large graph keeps its weights in a `<name>.onnx.data` sidecar next to the
`.onnx`. **`onnxruntime-node` resolves that sidecar automatically from disk, but
`onnxruntime-web` does NOT** — in the browser you must fetch the sidecar bytes
and hand them to the session via the `externalData` option. `GraniteSwitch.load`
does this for you:

```js
const gs = await GraniteSwitch.load(ort, {
  decodePath: `${repoBase}/onnx/model.onnx`,
  meta,
  executionProviders: ["wasm"],
  fetchExternalData: true,            // fetch onnx/model.onnx.data and mount it
  externalDataBaseUrl: `${repoBase}/onnx`,
});
```

The `path` in the `externalData` entry must equal the sidecar basename baked
into the graph (`model.onnx.data`); `load()` derives it from the `.onnx`
filename. You can also pass pre-fetched bytes directly (`decodeData`) — that is
how the Node validator exercises this same path.

**Do not load a large *embedded* `.onnx` in the browser.** onnxruntime-web
aborts when it has to parse a multi-hundred-MB inline protobuf in the WASM heap
(a numeric WASM abort, surfaced as `error: undefined` since the thrown value is a
number). The exporter therefore always writes the browser (`tfjs/`) artifacts
external, regardless of the embed setting for the top-level files.

## Export a model

```bash
pip install -e ".[onnx]"
python -m granite_switch.onnx.export <composed-checkpoint> --output web/example/model
```

This writes `decode.onnx`, `gs_onnx.json` (runtime metadata), and a `tfjs/`
layout (the browser-loadable artifacts, also loadable **on** transformers.js).

**Weight layout.** The top-level `decode.onnx` embeds its weights inline when
small (a single `.onnx`, convenient for `onnxruntime` / `onnxruntime-node`), or
keeps an external `<name>.onnx.data` sidecar past ONNX's 2 GiB protobuf cap —
e.g. the real 4 B `granite-switch-4.1-3b-preview` (~16.6 GB fp32). `--embed` /
`--no-embed` force that choice.

The **`tfjs/` artifacts are always external** (`model.onnx` + `model.onnx.data`)
regardless of `--embed`, because that is the only form the browser can load (see
[External data](#external-data-the-browser-weight-loading-requirement)).
`gs_onnx.json` records which files to fetch (`browser_decode`,
`browser_decode_external_data`, …).

**Quantize for the browser** (weight-only, no calibration data):

```bash
python -m granite_switch.onnx.export <ckpt> --output out --no-embed \
  --quantize int8 --quantize q4
```

`int8` (~4x smaller, onnxruntime-web friendly) and `q4` (~8x, transformers.js's
`model_q4` scheme) are written alongside the fp32 graphs and mirrored into
`tfjs/onnx/model_<scheme>.onnx`. The branchless switch/LoRA dataflow stays
float, so adapter selection is unaffected.

**Which dtype? Quantization quality is per-model — verify it.** Two opposite
real cases:

- **4 B `granite-switch-4.1-3b-preview`:** `int8` reproduces the fp32 greedy
  decode token-for-token (~6 GB); `q4` diverges after a few tokens.
- **350 M `granite-switch-4.0-350m-cti`:** `int8` is **broken** — the first-pass
  logit error vs HF is ~36.6 and argmax flips on the *first* generated token. A
  smaller model has less weight redundancy, so dynamic int8 destroys its
  numerics. Here **fp32 is the browser deliverable**, and at ~1.3 GB it fits the
  WASM heap comfortably (no quantization needed to fit).

The takeaway: never assume a dtype is correct — run the parity check
(`npm run validate:browser-path`) on each model and pick the smallest dtype that
still matches the golden. For the switch math itself, the branchless LoRA
dataflow always stays float, so adapter *selection* is never the thing that
breaks.

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
- **350 M `granite-switch-4.0-350m-cti`** → **runs in a real browser tab**
  (verified in Chrome). fp32 weights (~1.3 GB sidecar) load via `externalData`
  and greedy decode matches the Python golden token-for-token, including across
  the adapter control token. This is the "smaller composed Granite Switch" the
  4 B case calls for. See `example/index-ort.html` (raw onnxruntime-web + the JS
  tokenizer, full text-in / text-out).
- **4 B model** → loads + runs correctly in **Node** (onnxruntime-node, no 4 GB
  ceiling) and is upload-ready for the Hub, but is **too large for an in-tab
  WASM session**. A browser deployment needs a **smaller composed Granite
  Switch** (as above) or a genuinely sub-4 GB quantization (note: `q4` here
  already trades away decode correctness, so it is not a free win).

## Tokenizer / chat template

The chat template is shipped as a separate `chat_template.jinja` (it is **not**
set on the tokenizer), and the adapter control token is injected only when an
`adapter_name` is passed. `GraniteSwitchTokenizer` handles both:

```js
import { GraniteSwitchTokenizer } from "./src/granite-switch-tokenizer.js";

const chatTemplateText = await (await fetch(`${repoBase}/chat_template.jinja`)).text();
const tok = await GraniteSwitchTokenizer.load({
  localModelPath: `${repoBase}/`, modelName: "<repo-dir>",  // or { modelId: "org/repo" }
  chatTemplateText, meta,
});

const ids = tok.encode("The adversary used spearphishing emails…");  // fires the control token
const out = await gs.generate(ids, 12);
const text = tok.decode(out.slice(ids.length));
```

It is engine-agnostic — pass the template **text** so no Node `fs` leaks into a
browser bundle. In the browser, resolve the bare `@huggingface/transformers`
import via an import map (see `example/index-ort.html`).

## Run

**Browser (full text I/O, real model):** serve `web/` with COOP/COEP headers and
open `example/index-ort.html` (point its *repo base URL* at a packaged repo —
e.g. symlink one into `example/repo`):

```bash
cd web && npm run serve:coi -- 8733 "$PWD"
# open http://127.0.0.1:8733/example/index-ort.html
```

`serve:coi` (`test/coi_server.py`) sets `Cross-Origin-Opener-Policy` /
`Cross-Origin-Embedder-Policy` so the page is cross-origin isolated, enabling the
multi-threaded onnxruntime-web WASM build. (For the 350 M fp32 model this is not
strictly required — 1.3 GB fits the single-threaded heap once weights come in via
`externalData` — but it is the correct way to serve larger browser models.)

`example/index.html` is the older raw-token-ids demo (no tokenizer); prefer
`index-ort.html` for real use.

**Headless (Node, same ORT engine):**

```bash
cd web && npm install
npm run validate               # GATE 4: embedded fixture, greedy decode == golden
npm run validate:browser-path  # externalData wiring + tokenizer end-to-end == goldens
```

`validate_node.mjs` decodes a prompt with an adapter control token and asserts
the tokens match the Python golden. `validate_browser_path_node.mjs` additionally
loads the externalized decode graph through the `externalData` option (the
browser's path), then runs CTI **text** through the tokenizer + chat template,
asserts the control token is injected, generates, and asserts the decoded text
matches `test/golden_text.json`.

## Status / limitations

- Numeric parity with the HuggingFace backend is proven in `tests/onnx/` (tiny
  CPU fixture) **and on the real 4 B `granite-switch-4.1-3b-preview`**
  (max|diff| 9.5e-5, argmax 100% — GATE 5).
- The **350 M `granite-switch-4.0-350m-cti`** model runs end-to-end in a real
  browser tab (fp32, externalData, JS tokenizer + chat template; output matches
  the Python golden — verified in Chrome).
- Dense Granite only — MoE export is not yet wired (the export decoder layer
  asserts `not has_experts`). The real 4 B model is dense, so this is not a
  blocker for it.
- Tokenizer + chat-template integration is **done** (`GraniteSwitchTokenizer`).
  Greedy decode only — sampling (temperature/top-k/top-p via
  `@huggingface/transformers` `LogitsProcessor`) is the remaining
  productionization step.
- In-tab load of a quantized **4 B** model (sub-4 GB, WebGPU) remains out of
  reach until a sub-4 GB correct quantization exists; the 4 B runs in Node today.
