// SPDX-License-Identifier: Apache-2.0
// Granite Switch browser/Node runtime over ONNX Runtime Web.
//
// transformers.js has no custom-architecture API for `model_type:
// granite_switch`, so we drive the exported ONNX graph directly with raw
// onnxruntime. ONE graph:
//   - decode.onnx : input_ids [1,N] + past_* -> logits + grown present_* state.
//     Its input_ids seq axis is dynamic, so the SAME graph runs the batched first
//     pass over the whole prompt (empty past) and each single-token decode step.
//
// The switch's cumulative adapter selection is preserved across decode steps by
// threading present_switch_key0/val0 (compact per-position switch signal) back
// in as past_switch_*. This mirrors the Python parity tests (GATE 2/3).
//
// The runtime is engine-agnostic: pass in an `ort` module (onnxruntime-web in a
// browser, onnxruntime-node in Node) so the same code validates headlessly.
//
// External-data (weights) sidecars: large graphs keep their weights in a
// `<name>.onnx.data` file alongside the `.onnx` graph. onnxruntime-node resolves
// such a sidecar automatically from disk, but onnxruntime-web (WASM) does NOT —
// it must be handed the bytes explicitly via the `externalData` session option.
// `load()` therefore accepts a pre-fetched buffer (`decodeData`) or a
// `fetchExternalData` flag that fetches the sidecar in the browser. The `path`
// in the externalData entry MUST equal the sidecar basename baked into the
// graph's `location` (i.e. `<basename(onnxPath)>.data`), or ORT silently fails to
// mount the weights and aborts in WASM.

function _sidecarName(onnxPath) {
  const base = onnxPath.split(/[\\/]/).pop();
  return base + ".data"; // e.g. decode.onnx -> decode.onnx.data
}

function _dirOf(p) {
  const i = p.lastIndexOf("/");
  return i >= 0 ? p.slice(0, i) : ".";
}

function _toU8(data) {
  if (data instanceof Uint8Array) return data;
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return new Uint8Array(data);
}

export class GraniteSwitch {
  constructor(ort, decodeSession, meta) {
    this.ort = ort;
    this.decode = decodeSession;
    this.meta = meta;
  }

  // Resolve the externalData session option for one graph, or null. `explicit`
  // is a pre-fetched buffer (preferred — works in Node and browser); otherwise,
  // when `fetchExternalData` is set and a global `fetch` exists, fetch the
  // sidecar from `${externalDataBaseUrl || dirOf(onnxPath)}/${sidecarName}`.
  static async _externalDataOpt(onnxPath, { explicit, fetchExternalData, externalDataBaseUrl }) {
    const sidecar = _sidecarName(onnxPath);
    if (explicit != null) return [{ path: sidecar, data: _toU8(explicit) }];
    if (fetchExternalData && typeof fetch === "function") {
      const url = `${externalDataBaseUrl || _dirOf(onnxPath)}/${sidecar}`;
      const buf = await (await fetch(url)).arrayBuffer();
      return [{ path: sidecar, data: new Uint8Array(buf) }];
    }
    return null;
  }

  // executionProviders defaults to onnxruntime-web's: ["webgpu","wasm"].
  // For onnxruntime-node, pass ["cpu"].
  //
  // External-data opts (all optional, backward compatible):
  //   decodeData        : pre-fetched sidecar bytes (ArrayBuffer/Uint8Array)
  //   fetchExternalData : fetch the graph's .onnx.data in the browser
  //   externalDataBaseUrl : base URL/dir for that fetch (default: graph's dir)
  static async load(ort, {
    decodePath, meta, executionProviders,
    decodeData, fetchExternalData, externalDataBaseUrl,
  }) {
    const eps = executionProviders || ["webgpu", "wasm"];

    const decodeExt = await this._externalDataOpt(decodePath, {
      explicit: decodeData, fetchExternalData, externalDataBaseUrl,
    });
    const decodeOpts = { executionProviders: eps };
    if (decodeExt) decodeOpts.externalData = decodeExt;
    const decode = await ort.InferenceSession.create(decodePath, decodeOpts);

    return new GraniteSwitch(ort, decode, meta);
  }

  _t(type, data, dims) {
    return new this.ort.Tensor(type, data, dims);
  }

  // Greedy generation. Returns the full token sequence (prompt + generated).
  async generate(promptIds, maxNewTokens = 8) {
    const meta = this.meta;
    const L = meta.n_layers;

    // ── Prompt pass ───────────────────────────────────────────────
    // The single decode graph seeds its own KV/switch state. This validator
    // warms it by replaying the prompt token-by-token (one step() per token);
    // the browser runtime instead feeds the whole prompt in one pass (input_ids
    // [1,N] + empty past) — same graph, fewer session.run calls.
    let sk0 = new Float32Array(0), skDims = [1, 0];
    let sv0 = new Float32Array(0);
    let pastK = Array.from({ length: L }, () => new Float32Array(0));
    let pastV = Array.from({ length: L }, () => new Float32Array(0));
    let pastLen = 0;

    const kvh = meta.kv_heads, hd = meta.head_dim;
    const tokens = [...promptIds];
    let lastLogits = null;

    const step = async (tok) => {
      const feed = {
        input_ids: this._t("int64", BigInt64Array.from([BigInt(tok)]), [1, 1]),
        past_switch_key0: this._t("float32", sk0, [1, pastLen]),
        past_switch_val0: this._t("float32", sv0, [1, pastLen]),
      };
      for (let li = 0; li < L; li++) {
        feed[`past_key.${li}`] = this._t("float32", pastK[li], [1, kvh, pastLen, hd]);
        feed[`past_value.${li}`] = this._t("float32", pastV[li], [1, kvh, pastLen, hd]);
      }
      const out = await this.decode.run(feed);
      // advance state
      sk0 = out.present_switch_key0.data;
      sv0 = out.present_switch_val0.data;
      for (let li = 0; li < L; li++) {
        pastK[li] = out[`present_key.${li}`].data;
        pastV[li] = out[`present_value.${li}`].data;
      }
      pastLen += 1;
      return out.logits.data; // [vocab] for the single position
    };

    // Replay the prompt to build state; keep the last token's logits.
    for (const tok of promptIds) {
      lastLogits = await step(tok);
    }

    // ── Greedy decode loop ────────────────────────────────────────
    for (let i = 0; i < maxNewTokens; i++) {
      const next = argmax(lastLogits);
      tokens.push(next);
      lastLogits = await step(next);
    }
    return tokens;
  }
}

function argmax(arr) {
  let best = 0, bestV = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > bestV) { bestV = arr[i]; best = i; }
  }
  return best;
}
