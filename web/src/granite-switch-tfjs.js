// SPDX-License-Identifier: Apache-2.0
// Granite Switch loaded and run ON transformers.js (@huggingface/transformers).
//
// transformers.js has no public custom-architecture registration, and
// `AutoModelForCausalLM.from_pretrained` rejects `model_type: granite_switch`
// outright ("Unsupported model type"). To load ON transformers.js we therefore
// present the exported decode graph with a config that declares a supported
// DecoderOnly `model_type` (gpt2). transformers.js then:
//
//   * loads config.json via its AutoConfig,
//   * creates the ONNX InferenceSession through ITS OWN onnx backend
//     (createInferenceSession / the transformers.js ORT build + env), and
//   * hands back a transformers.js PreTrainedModel whose `.sessions.model`
//     is run with transformers.js's Tensor and session runner.
//
// We then drive a greedy loop over that transformers.js-owned session, threading
// the switch's cumulative adapter state (which transformers.js's generic
// generation loop has no concept of). The session execution, tensors, and model
// object are all transformers.js's — only the state-threading control flow is ours.

import { env, AutoModelForCausalLM, Tensor } from "@huggingface/transformers";

export class GraniteSwitchTfjs {
  constructor(model, meta) {
    this.model = model;          // transformers.js PreTrainedModel
    this.session = model.sessions.model;
    this.meta = meta;
  }

  /**
   * Load a packaged Granite Switch model ON transformers.js, locally or from
   * the HF Hub. The model repo carries config.json + onnx/model[_int8|_q4].onnx
   * + gs_onnx.json (see granite_switch.onnx.package).
   *
   * Local:
   *   GraniteSwitchTfjs.load({ localModelPath: "/abs/dir/", modelName: "tfjs", meta })
   * Remote (HF Hub):
   *   const meta = await (await fetch(
   *     "https://huggingface.co/ORG/REPO/resolve/main/gs_onnx.json")).json();
   *   GraniteSwitchTfjs.load({ modelId: "ORG/REPO", meta, dtype: "int8" })
   *
   * @param {object} opts
   * @param {string} [opts.modelId]        HF Hub repo id (remote load)
   * @param {string} [opts.localModelPath] base dir for local models
   * @param {string} [opts.modelName]      subdir under localModelPath (local load)
   * @param {object} opts.meta             gs_onnx.json metadata
   * @param {string} [opts.dtype="int8"]   "fp32" | "int8" | "q4" (int8 is the
   *                                       verified-correct browser default; q4
   *                                       can diverge for this architecture)
   */
  static async load({ modelId, localModelPath, modelName, meta, dtype = "int8" }) {
    if (modelId) {
      env.allowRemoteModels = true;
      env.allowLocalModels = false;
      const model = await AutoModelForCausalLM.from_pretrained(modelId, { dtype });
      return new GraniteSwitchTfjs(model, meta);
    }
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.localModelPath = localModelPath;
    const model = await AutoModelForCausalLM.from_pretrained(modelName, { dtype });
    return new GraniteSwitchTfjs(model, meta);
  }

  // Run the transformers.js-owned session for one decode step.
  async _step(tok, state) {
    const { sk0, sv0, pastK, pastV, pastLen } = state;
    const L = this.meta.n_layers, kvh = this.meta.kv_heads, hd = this.meta.head_dim;
    const feed = {
      input_ids: new Tensor("int64", BigInt64Array.from([BigInt(tok)]), [1, 1]),
      past_switch_key0: new Tensor("float32", sk0, [1, pastLen]),
      past_switch_val0: new Tensor("float32", sv0, [1, pastLen]),
    };
    for (let li = 0; li < L; li++) {
      feed[`past_key.${li}`] = new Tensor("float32", pastK[li], [1, kvh, pastLen, hd]);
      feed[`past_value.${li}`] = new Tensor("float32", pastV[li], [1, kvh, pastLen, hd]);
    }
    // transformers.js runs the session (its ORT backend) when we call the model.
    const out = await this.session.run(
      Object.fromEntries(Object.entries(feed).map(([k, v]) => [k, v.ort_tensor]))
    );
    return out;
  }

  async generate(promptIds, maxNewTokens = 8) {
    const L = this.meta.n_layers;
    let state = {
      sk0: new Float32Array(0), sv0: new Float32Array(0),
      pastK: Array.from({ length: L }, () => new Float32Array(0)),
      pastV: Array.from({ length: L }, () => new Float32Array(0)),
      pastLen: 0,
    };
    const tokens = [...promptIds];
    let lastLogits = null;

    const advance = async (tok) => {
      const out = await this._step(tok, state);
      state = {
        sk0: out.present_switch_key0.data,
        sv0: out.present_switch_val0.data,
        pastK: Array.from({ length: L }, (_, li) => out[`present_key.${li}`].data),
        pastV: Array.from({ length: L }, (_, li) => out[`present_value.${li}`].data),
        pastLen: state.pastLen + 1,
      };
      return out.logits.data;
    };

    for (const tok of promptIds) lastLogits = await advance(tok);
    for (let i = 0; i < maxNewTokens; i++) {
      const next = argmax(lastLogits);
      tokens.push(next);
      lastLogits = await advance(next);
    }
    return tokens;
  }
}

function argmax(arr) {
  let best = 0, bestV = -Infinity;
  for (let i = 0; i < arr.length; i++) if (arr[i] > bestV) { bestV = arr[i]; best = i; }
  return best;
}
