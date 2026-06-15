# Granite Switch — native transformers.js demo (Vite, shippable)

A self-contained browser demo that loads a **Granite Switch** model with the
standard `AutoModelForCausalLM.from_pretrained` — `model_type: granite_switch`,
**no `gpt2` workaround** — and runs it fully client-side via ONNX Runtime Web.
Built as a static bundle, deployable to an HF **Static Space**.

## How it works

`src/granite-switch-register.js` (in the parent `web/src/`) is a **self-registering
shim**: importing it teaches transformers.js the `granite_switch` architecture by
mutating its internal type-resolution maps and supplying a custom forward that
threads the per-token adapter **switch state** through transformers.js's own
generation loop + KV cache. The shim deep-imports transformers.js internals
(`@huggingface/transformers/src/...`), which the package `exports` map hides — so
this demo must be **bundled** (Vite resolves those subpaths via an alias in
`vite.config.js`); a bare-CDN `<script>` cannot reach them.

`vite.config.js` also:
- aliases `@huggingface/transformers/src/*` to the package's `src/` files (the
  `dist/` build is a separate module instance and doesn't expose the maps);
- stubs Node-only deps pulled in by `src/` (`onnxruntime-node`, `sharp`,
  `utils/io.js`) so the browser bundle builds;
- forces a single chunk + the `AutoTokenizer` re-export from the package root, to
  avoid a Rollup temporal-dead-zone error in the cyclic `src/` tokenizer graph;
- (dev only) returns real 404s for missing repo files (Vite's SPA fallback would
  otherwise serve `index.html`, breaking transformers.js's optional-file probes).

## Run locally (dev)

You need a packaged Granite Switch repo served at `/repo`. Create one under
`public/repo/` (gitignored) — a `granite_switch`-typed `config.json`
(`python -m granite_switch.onnx.package <export> <repo> --native`), the
`onnx/model.onnx` + `model.onnx.data` sidecar, `gs_onnx.json`,
`generation_config.json`, `chat_template.jinja`, and the tokenizer files. For the
350 M model you can symlink the big `onnx/` from a packaged repo to avoid copying.

```bash
npm install
npm run dev     # http://localhost:5173
```

## Build + ship (HF Static Space)

```bash
npm run build   # -> dist/ (self-contained static bundle)
```

Serve `dist/` from any static host with **COOP/COEP** headers (cross-origin
isolation, for the multi-threaded ORT-web WASM). For a quick local check of the
built bundle:

```bash
# from web/: real static server with COOP/COEP + true 404s
python3 test/coi_server.py 8744 "$PWD/example-demo/dist"
# open http://127.0.0.1:8744/index.html
```

For a real deploy, point `VITE_MODEL_ID` at a **public HF Hub repo** so the weights
are fetched from the Hub at runtime instead of being bundled into `dist/`. This is
the shippable Space configuration — the flagship demo loads the multi-adapter model
in its `int8` variant for a smaller, faster first download:

```bash
VITE_MODEL_ID="barha/granite-switch-4.0-350m-demo-onnx" VITE_DTYPE=int8 npm run build
```

`main.js` detects `VITE_MODEL_ID` and uses transformers.js's native Hub resolution
(`huggingface.co/<id>/resolve/main`). `VITE_DTYPE` selects the ONNX variant
(`fp32` | `int8` | `q4`); `loadGraniteSwitch` maps it to the matching
`model[_int8].onnx` + `.onnx.data` sidecar. Omit `VITE_MODEL_ID` for the local
`/repo` dev mode above (which uses `VITE_REPO_BASE`, default `/repo`).

## The demo: one model, many skills

`src/main.js` drives a tabbed UI — one tab per embedded adapter
(`cti-technique-mapping`, `genai-attack-vector`, `text-to-json`). The model loads
**once**; each tab fires a different adapter via its control token (the
`adapterName` passed to `GraniteSwitchTokenizer.encode`). Each adapter has its own
trained prompt framing, expressed per-tab in the `ADAPTERS` config: an `instruction`
+ `wrapTag` (`<cti>` / `<incident>`), or a `buildContent` override for the
text-to-json schema-preamble format. Outputs are rendered per task: a MITRE ID→name
for CTI, the matched attack-vector slug, or pretty-printed JSON.

## Verified

The real `granite-switch-4.0-350m` multi-adapter model loads natively in a Chrome
tab (both `npm run dev` and the built `dist/`). The CTI tab greedy-decodes the
spearphishing prompt to `T1105`, matching the Python golden
(`web/test/golden_text.json`); the legacy single-adapter `encode({ instruction })`
framing is unchanged (regression-guarded by `web/test/validate_browser_path_node.mjs`).
