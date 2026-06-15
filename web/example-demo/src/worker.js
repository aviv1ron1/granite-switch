// SPDX-License-Identifier: Apache-2.0
// Granite Switch demo — model ENGINE, running in a Web Worker.
//
// Why a worker: onnxruntime-web decodes on whatever thread drives generation, and
// transformers.js's generation loop (sampling, logits, KV-cache tensor ops) PLUS our
// shim's custom per-token forward (graniteSwitchForward, threading the switch state)
// run between every token. On the main thread that janks the UI — the browser pops its
// "page unresponsive" dialog. Following the official transformers.js-examples pattern
// (e.g. smollm-webgpu), the WHOLE model — load + generate — lives here, off the main
// thread, and tokens stream back to the UI via postMessage. The main thread only renders.
//
// This file imports the SAME shim specifiers main.js used to, so the granite_switch
// architecture is registered in THIS worker's module instance. The shim, tokenizer, and
// core are all DOM-free, so they run unchanged in a worker.

import {
  loadGraniteSwitch,
  AutoTokenizer, // dist/-instance tokenizer (paired with envForTokenizer)
  envForTokenizer,
  env,
  Tensor,
} from "../../src/granite-switch-register.js";
import { GraniteSwitchTokenizer } from "../../src/granite-switch-tokenizer.js";
// TextStreamer from the PACKAGE ROOT (dist instance) — the same instance tok.tok is
// built from. Importing it from src/ would risk the same cyclic-init TDZ as the tokenizer.
import { TextStreamer } from "@huggingface/transformers";

let model = null;
let tok = null;

// ORT thread/proxy config lives here because it configures the same shim `env`
// instance this worker's model uses. We're already in a worker, so proxy=false:
// proxy would spawn a SECOND worker for the WASM backend, adding needless hops.
// Multi-threaded intra-op still requires cross-origin isolation (SharedArrayBuffer),
// which coi-serviceworker provides on the page (and thus in this worker too).
function configureOrt() {
  const threaded = self.crossOriginIsolated === true;
  try {
    const wasm = env.backends.onnx.wasm;
    wasm.numThreads = threaded ? (navigator.hardwareConcurrency || 4) : 1;
    wasm.proxy = false;
  } catch (_) { /* env shape varies across transformers.js versions; non-fatal */ }
  return threaded;
}

// Mirror main.js's configureRemote for local/served-repo mode.
function configureRemote(e, remoteHost) {
  e.allowLocalModels = false;
  e.allowRemoteModels = true;
  e.remoteHost = remoteHost;
  e.remotePathTemplate = "{model}";
  e.useBrowserCache = false;
}

async function handleLoad({ mode, name, fileBase, dtype, remoteHost }) {
  if (mode === "hub") {
    for (const e of [env, envForTokenizer]) {
      e.allowLocalModels = false;
      e.allowRemoteModels = true;
      e.useBrowserCache = false; // the Hub is the source of truth
    }
  } else {
    configureRemote(env, remoteHost);
    configureRemote(envForTokenizer, remoteHost);
  }

  const threaded = configureOrt();

  // ONE shared load: all three adapters live in this single model; the control
  // token (chosen per generation) selects which fires. Forward progress events on.
  model = await loadGraniteSwitch(name, {
    dtype,
    progress_callback: (ev) => self.postMessage({ type: "progress", data: ev }),
  });

  // Workers have fetch(); same env/template as the model, so the same URLs resolve.
  const meta = await (await fetch(`${fileBase}/gs_onnx.json`)).json();
  const chatTemplateText = await (await fetch(`${fileBase}/chat_template.jinja`)).text();
  tok = await GraniteSwitchTokenizer.load({
    modelId: name,
    chatTemplateText,
    meta,
    AutoTokenizer,
    env: envForTokenizer,
  });

  self.postMessage({ type: "ready", threaded });
}

// One greedy generation, streaming decoded chunks back per token. `which` ("base" |
// "adapter") lets the UI route the stream to the correct column.
async function handleGenerate(payload) {
  const { which, text, adapterName, instruction, wrapTag, content, maxNewTokens } = payload;
  try {
    const encodeOpts = { adapterName, instruction, wrapTag };
    // text-to-json: the main thread pre-built the trained `${query}${PREAMBLE}${schema}`
    // string (its buildContent fn can't cross postMessage), so we override directly.
    if (content != null) encodeOpts.buildContent = () => content;

    const ids = tok.encode(text, encodeOpts);
    const inputIds = new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);

    // TextStreamer decodes generated tokens to incremental text chunks. skip_prompt
    // drops the prompt; callback_function(chunk) fires per finalized piece (APPEND on
    // the main side). token_callback_function counts tokens and starts the tps clock.
    let numTokens = 0;
    let startTs = null;
    const streamer = new TextStreamer(tok.tok, {
      skip_prompt: true,
      skip_special_tokens: true,
      token_callback_function: () => {
        if (startTs == null) startTs = performance.now();
        numTokens++;
      },
      callback_function: (chunk) => {
        const elapsed = startTs != null ? (performance.now() - startTs) / 1000 : 0;
        const tps = elapsed > 0 ? numTokens / elapsed : 0;
        self.postMessage({ type: "update", which, text: chunk, numTokens, tps });
      },
    });

    const seq = await model.generate({
      inputs: inputIds,
      max_new_tokens: maxNewTokens,
      do_sample: false,
      num_beams: 1,
      streamer,
    });
    const out = Array.from(seq.tolist()[0], (v) => Number(v)).slice(ids.length);
    self.postMessage({ type: "complete", which, raw: tok.decode(out).trim(), numTokens });
  } catch (err) {
    self.postMessage({ type: "error", which, message: err?.message ?? String(err) });
  }
}

self.addEventListener("message", (e) => {
  const msg = e.data;
  if (msg.type === "load") {
    handleLoad(msg.payload).catch((err) =>
      self.postMessage({ type: "error", message: err?.message ?? String(err) }),
    );
  } else if (msg.type === "generate") {
    handleGenerate(msg.payload);
  }
});
