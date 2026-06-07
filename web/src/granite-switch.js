// SPDX-License-Identifier: Apache-2.0
// Granite Switch browser/Node runtime over ONNX Runtime Web.
//
// transformers.js has no custom-architecture API for `model_type:
// granite_switch`, so we drive the exported ONNX graphs directly with raw
// onnxruntime. Two graphs:
//   - prefill.onnx : full prompt   -> logits + present_switch_* + present_* KV
//   - decode.onnx  : one token+past -> logits + grown present_* state
//
// The switch's cumulative adapter selection is preserved across decode steps by
// threading present_switch_key0/val0 (compact per-position switch signal) back
// in as past_switch_*. This mirrors the Python parity tests (GATE 2/3).
//
// The runtime is engine-agnostic: pass in an `ort` module (onnxruntime-web in a
// browser, onnxruntime-node in Node) so the same code validates headlessly.

export class GraniteSwitch {
  constructor(ort, prefillSession, decodeSession, meta) {
    this.ort = ort;
    this.prefill = prefillSession;
    this.decode = decodeSession;
    this.meta = meta;
  }

  // executionProviders defaults to onnxruntime-web's: ["webgpu","wasm"].
  // For onnxruntime-node, pass ["cpu"].
  static async load(ort, { prefillPath, decodePath, meta, executionProviders }) {
    const opts = { executionProviders: executionProviders || ["webgpu", "wasm"] };
    const prefill = await ort.InferenceSession.create(prefillPath, opts);
    const decode = await ort.InferenceSession.create(decodePath, opts);
    return new GraniteSwitch(ort, prefill, decode, meta);
  }

  _t(type, data, dims) {
    return new this.ort.Tensor(type, data, dims);
  }

  // Run the prefill graph over the full prompt. Returns { logits, state }.
  async _runPrefill(promptIds) {
    const n = promptIds.length;
    const input = this._t("int64", BigInt64Array.from(promptIds.map(BigInt)), [1, n]);
    const out = await this.prefill.run({ input_ids: input });
    return out;
  }

  // Greedy generation. Returns the full token sequence (prompt + generated).
  async generate(promptIds, maxNewTokens = 8) {
    const meta = this.meta;
    const L = meta.n_layers;

    // ── Prefill ───────────────────────────────────────────────────
    // The prefill graph only outputs logits in this build; to seed the decode
    // cache we instead warm the decode graph by replaying the prompt through it
    // (simplest correct path; a fused prefill->state export is a later optimization).
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
