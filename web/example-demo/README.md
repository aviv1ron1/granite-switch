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

For a real deploy, point `VITE_REPO_BASE` at a **public HF Hub repo** instead of
bundling 1.4 GB of weights into `dist/`:

```bash
VITE_REPO_BASE="https://huggingface.co/<org>/<repo>/resolve/main" npm run build
```

(and set `env` to allow remote models — `main.js`'s `configureRemote` already
drives the fetch via `remoteHost` + `remotePathTemplate`).

## Verified

The real `granite-switch-4.0-350m-cti` model loads natively in a Chrome tab (both
`npm run dev` and the built `dist/`) and greedy-decodes the CTI prompt to `T1105`,
matching the Python golden (`web/test/golden_text.json`). Screenshots in
`docs/granite_switch_350m_native_tfjs_*.png`.
