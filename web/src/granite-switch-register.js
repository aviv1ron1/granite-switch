// SPDX-License-Identifier: Apache-2.0
// Self-registering shim: teach @huggingface/transformers (transformers.js) about
// `model_type: granite_switch` so it loads NATIVELY via from_pretrained — no more
// gpt2-typed workaround (see granite-switch-tfjs.js for the old approach).
//
// transformers.js 4.2.0 has NO runtime API to register a new model type; its
// type->class maps are built from hard-coded lists at module load. We instead
// mutate those maps directly. The maps we need are exported from the package's
// internal modules, which the package `exports` map normally hides — so this file
// deep-imports by file path. That ONLY resolves under a bundler (Vite/esbuild) or
// in Node; it does NOT work via the bare-CDN <script> build. The browser demo is
// therefore bundled (see web/example-demo).
//
// Importing this module once (for its side effects) is enough:
//
//   import { GraniteSwitchForCausalLM } from "./granite-switch-register.js";
//   const model = await GraniteSwitchForCausalLM.from_pretrained(repo, { dtype });
//   // or, via the patched Auto entry point:
//   const model = await AutoModelForCausalLM.from_pretrained(repo, { dtype });
//
// HOW THE SWITCH STATE RIDES NATIVELY
// -----------------------------------
// The decode graph has extra I/O beyond a normal KV-cached decoder:
//   inputs : past_switch_key0, past_switch_val0   (float32 [1, pastLen])
//   outputs: present_switch_key0, present_switch_val0  ([1, pastLen+1])
// This compact signal is the switch's cumulative adapter selection; it MUST be
// threaded across decode steps or adapter selection breaks after the prompt.
//
// transformers.js's generation loop carries arbitrary `present*` outputs forward
// for free: getPastKeyValues() stores every output whose name starts with
// "present" into the DynamicCache (renaming "present" -> "past_key_values"), and
// addPastKeyValues() feeds every cache key straight back into the next step. So we
// let the stock flow carry our switch state; our custom _forward only has to
//   (1) on the first step (empty cache) zero-fill the graph's past_* inputs, and
//   (2) on later steps rename the cache's stored keys back to the graph's exact
//       input names before running the session.
//
// The graph names its KV cache past_key.{i}/past_value.{i} and its switch state
// past_switch_key0/val0 (see the exported decode.onnx / granite-switch-tfjs.js).
// transformers.js's getPastKeyValues stores a `present_switch_key0` output as
// `past_key_values_switch_key0` (it replaces the "present" prefix). Note the
// result has NO dot after `past_key_values`, so DynamicCache.get_seq_length()
// (which keys off the `past_key_values.` prefix, dot included) safely ignores it
// and reports the true KV length. That property is load-bearing.

// IMPORTANT: every transformers.js symbol used here — and by the app — must come
// from the SAME module instance. The package's `dist/` bundle and its `src/`
// tree are SEPARATE instances (their `PreTrainedModel` are not `===`), and only
// `src/` exposes the internal maps we must mutate. So we bind EVERYTHING to
// `src/` deep specifiers, and re-export the app-facing API (AutoModelForCausalLM,
// AutoConfig, AutoTokenizer, Tensor, env) from here so callers never touch the
// package root. Resolution: Vite resolves these subpaths natively; the Node
// validators run with test/hf-src-loader.mjs, which maps `@huggingface/
// transformers/src/*` to absolute file URLs (the package `exports` map otherwise
// blocks bare deep specifiers in Node).
import {
  PreTrainedModel,
  MODEL_TYPE_MAPPING,
  MODEL_NAME_TO_CLASS_MAPPING,
  MODEL_CLASS_TO_NAME_MAPPING,
  MODEL_TYPES,
  boolTensor,
} from "@huggingface/transformers/src/models/modeling_utils.js";
import { Tensor, ones } from "@huggingface/transformers/src/utils/tensor.js";
import { sessionRun, constructSessions } from "@huggingface/transformers/src/models/session.js";
import { pick } from "@huggingface/transformers/src/utils/core.js";
import { AutoModelForCausalLM } from "@huggingface/transformers/src/models/auto/modeling_auto.js";
import { AutoConfig } from "@huggingface/transformers/src/configs.js";
import { env } from "@huggingface/transformers/src/env.js";

// App-facing re-exports — import these FROM THE SHIM, not from the package root,
// so the whole app shares one module instance (see note above).
//
// AutoTokenizer is the one exception: it is re-exported from the PACKAGE ROOT
// (the dist/ build), NOT the src/ tree. Bundling the src/ tokenizer auto-map with
// Rollup trips a temporal-dead-zone error (its mapping object references class
// bindings hoisted out of order); the dist/ build is already correctly ordered.
// The tokenizer shares no mutable registration state with the model, so a
// separate instance is fine — but it has its OWN `env`, so configure tokenizer
// loads with the matching `envForTokenizer` export below.
export { AutoModelForCausalLM, AutoConfig, env, Tensor };
export { AutoTokenizer, env as envForTokenizer } from "@huggingface/transformers";

const MODEL_NAME = "GraniteSwitchForCausalLM";
const MODEL_TYPE = "granite_switch";

// getPastKeyValues stores every `present*` output after replacing the literal
// token "present" with "past_key_values". So our graph outputs are stored as:
//   present_key.{i}      -> past_key_values_key.{i}     (graph input: past_key.{i})
//   present_value.{i}    -> past_key_values_value.{i}   (graph input: past_value.{i})
//   present_switch_key0  -> past_key_values_switch_key0 (graph input: past_switch_key0)
//   present_switch_val0  -> past_key_values_switch_val0 (graph input: past_switch_val0)
// The inverse rename is a single rule: leading "past_key_values" -> "past"
// (i.e. drop the "_values"), which maps every stored key back to the exact graph
// input name. We never re-export the ONNX graph; the adaptation lives here.
function storedNameToGraphInput(storedName) {
  return storedName.replace(/^past_key_values/, "past");
}

// Zero-length typed-array factory by ORT element type string.
const ZERO_DATA = {
  float32: (n) => new Float32Array(n),
  float16: (n) => new Uint16Array(n),
  int64: (n) => new BigInt64Array(n),
};

// Build the empty (pastLen == 0) cache feeds for the FIRST step, reading each
// cache input's element type + symbolic shape from the session metadata. The
// graph's cache inputs are the names NOT also produced as a non-cache output;
// concretely: anything starting with `past_` (past_key.*, past_value.*,
// past_switch_*). Their leading "sequence" dim is 0 at the start.
function emptyCacheFeeds(session) {
  const feeds = {};
  for (const meta of session.inputMetadata ?? session.inputNames.map((name) => ({ name }))) {
    const name = meta.name ?? meta;
    if (!name.startsWith("past_")) continue;
    const type = meta.type ?? "float32";
    // Resolve symbolic dims: batch -> 1, the past-length axis -> 0, others as-is.
    const shape = (meta.shape ?? []).map((d) => {
      if (typeof d === "number") return d;
      // Symbolic. The only dim that is genuinely 0 at start is the past length.
      // Heuristic: a symbolic dim whose name mentions "past"/"seq"/"len" is the
      // sequence axis; the batch axis resolves to 1; anything else to its
      // numeric default is unknown -> treat as the sequence axis (0) only for
      // KV rank-4 / switch rank-2 tensors where exactly one axis is the length.
      return /past|seq|len/i.test(String(d)) ? 0 : 1;
    });
    const size = shape.reduce((a, b) => a * b, 1);
    const make = ZERO_DATA[type] ?? ((n) => new Float32Array(n));
    feeds[name] = new Tensor(type, make(size), shape);
  }
  return feeds;
}

// Map a session's `past_*` graph inputs to the `present_*` outputs they consume,
// so we can thread state across an INTERNAL multi-token loop. past_key.{i} pairs
// with present_key.{i}; past_switch_key0 with present_switch_key0; etc. The rule
// is the inverse of the present->past_key_values rename applied to the *graph's
// own* names: a graph input `past_X` is fed by the graph output `present_X`.
function graphPastToPresent(pastInputName) {
  return pastInputName.replace(/^past/, "present");
}

// Run the decode graph for a SINGLE token given the current `past_*` feeds
// (already named as graph inputs). Returns the raw session outputs.
async function runDecodeStep(session, singleTokenIds, pastFeeds) {
  const feeds = { input_ids: singleTokenIds, ...pastFeeds };
  if (session.inputNames.includes("use_cache_branch")) {
    // True once any past state exists (i.e. past length > 0).
    const hasPast = pastFeeds["past_key.0"]?.dims?.at(-2) > 0;
    feeds.use_cache_branch = boolTensor(hasPast);
  }
  if (session.inputNames.includes("num_logits_to_keep") && !feeds.num_logits_to_keep) {
    feeds.num_logits_to_keep = new Tensor("int64", [0n], []);
  }
  // Drop anything the session does not declare (e.g. attention_mask), as
  // decoder_forward does.
  return await sessionRun(session, pick(feeds, session.inputNames));
}

// Custom forward. Unlike a normal decoder, the Granite Switch decode graph has NO
// prefill: it only consumes ONE token per run while threading KV + switch state.
// transformers.js's generate() hands the WHOLE prompt to the first _forward, so
// here we replay a multi-token input_ids through the decode graph internally,
// threading state across those positions, and return the LAST position's outputs
// (logits + the accumulated present_*). For the subsequent single-token generate
// steps this loop runs exactly once. `self` is the model; `model_inputs` carries
// input_ids and `past_key_values` (a DynamicCache, or null/empty on the prefill).
async function graniteSwitchForward(self, model_inputs) {
  const session = self.sessions["model"];
  const { past_key_values, input_ids } = model_inputs;
  const hasPast = past_key_values && Object.keys(past_key_values).length > 0;

  // ── Batched prefill (first forward, empty cache) ──
  // transformers.js hands the WHOLE prompt to the first _forward. If a stateful
  // prefill session is loaded, run it ONCE over all prompt tokens — the Python HF
  // model's batched prefill — instead of replaying the prompt token-by-token
  // through the decode graph (which is N serial session.run calls => slow first
  // token). The prefill graph emits present_* with the SAME names as the decode
  // graph, so transformers.js's getPastKeyValues rolls them into the cache and the
  // first decode step consumes them. logits come back [1, seqLen, vocab]; generate
  // slices the last position itself — we return the raw outputs unchanged.
  if (!hasPast) {
    const prefill = self.sessions["prefill"];
    if (prefill) {
      const feeds = { input_ids };
      if (prefill.inputNames.includes("num_logits_to_keep")) {
        feeds.num_logits_to_keep = new Tensor("int64", [0n], []);
      }
      return await sessionRun(prefill, pick(feeds, prefill.inputNames));
    }
    // No prefill session (older repo / load failed) → fall back to the per-token
    // decode replay below. Correct, just slower for the first token.
  }

  // Seed the `past_*` feeds: from the cache (steady state) or zero-filled (start).
  let pastFeeds = {};
  if (hasPast) {
    for (const key in past_key_values) {
      pastFeeds[storedNameToGraphInput(key)] = past_key_values[key];
    }
  } else {
    pastFeeds = emptyCacheFeeds(session);
  }

  // input_ids is [batch=1, seqLen]; iterate its positions.
  const ids = input_ids.tolist()[0]; // bigint[] for the single batch row
  let out;
  for (let i = 0; i < ids.length; i++) {
    const tok = new Tensor("int64", BigInt64Array.from([BigInt(ids[i])]), [1, 1]);
    out = await runDecodeStep(session, tok, pastFeeds);
    if (i < ids.length - 1) {
      // Thread this step's present_* outputs into the next step's past_* feeds.
      const next = {};
      for (const name of session.inputNames) {
        if (!name.startsWith("past_")) continue;
        const presentName = graphPastToPresent(name);
        if (presentName in out) next[name] = out[presentName];
      }
      pastFeeds = next;
    }
  }
  // Return the last position's raw outputs. The stock getPastKeyValues() rolls
  // present_*/present_switch_* into the cache for the next generate step.
  return out;
}

// Derive the cached sequence length from OUR cache key names. We cannot use
// DynamicCache.get_seq_length(): it looks for a key prefixed `past_key_values.`
// (dot included), but our KV entries are stored as `past_key_values_key.{i}`
// (underscore, from the present->past_key_values rename), so it would throw.
function pastLengthFromCache(cache) {
  if (!cache) return 0;
  for (const name in cache) {
    if (name.startsWith("past_key_values_key.")) {
      return cache[name].dims.at(-2); // [batch, heads, past_len, head_dim]
    }
  }
  return 0;
}

// Custom prepare: a trimmed decoder_prepare_inputs_for_generation that computes
// past_length from our cache instead of the (throwing) get_seq_length. Its jobs:
//   (1) slice input_ids down to the single unprocessed token during decode;
//   (2) maintain `attention_mask` bookkeeping. Our graph has NO attention_mask
//       input (pick() drops it before the session), but transformers.js's
//       _update_model_kwargs_for_generation unconditionally grows attention_mask
//       each step, so it must exist or that step throws. We seed it here.
function graniteSwitchPrepareInputs(self, input_ids, model_inputs, _generation_config) {
  const past_length = pastLengthFromCache(model_inputs.past_key_values);
  if (model_inputs.past_key_values && past_length < model_inputs.input_ids.dims[1]) {
    model_inputs.input_ids = model_inputs.input_ids.slice(null, [past_length, null]);
  }
  if (!model_inputs.attention_mask) {
    const seqLen = model_inputs.input_ids.dims[1];
    model_inputs.attention_mask = ones([model_inputs.input_ids.dims[0], past_length + seqLen]);
  }
  return model_inputs;
}

export class GraniteSwitchForCausalLM extends PreTrainedModel {
  constructor(config, sessions, configs) {
    super(config, sessions, configs);
    // Belt-and-suspenders: the type registration below makes the base set these
    // from the DecoderOnly type config, but we pin them so our custom forward and
    // prepare are used even if the name lookup ever changes.
    this._forward = graniteSwitchForward;
    this._prepare_inputs_for_generation = graniteSwitchPrepareInputs;
    this.can_generate = true;
    if (!this.forward_params.includes("past_key_values")) {
      this.forward_params.push("past_key_values");
    }
  }
}

// ── Register the architecture in transformers.js's type-resolution maps. ──
// Without MODEL_TYPE_MAPPING the base PreTrainedModel constructor would silently
// fall back to an ENCODER config (no error), so this is mandatory.
MODEL_CLASS_TO_NAME_MAPPING.set(GraniteSwitchForCausalLM, MODEL_NAME);
MODEL_NAME_TO_CLASS_MAPPING.set(MODEL_NAME, GraniteSwitchForCausalLM);
MODEL_TYPE_MAPPING.set(MODEL_NAME, MODEL_TYPES.DecoderOnly);

// ── Patch the Auto entry point so the familiar call works for granite_switch. ──
// AutoModelForCausalLM.from_pretrained resolves its map value through the frozen
// models.js namespace, which a shim cannot extend; so we intercept the dispatch
// and delegate to our class directly. Other model types are unaffected.
const _autoFromPretrained = AutoModelForCausalLM.from_pretrained.bind(AutoModelForCausalLM);
AutoModelForCausalLM.from_pretrained = async function (pretrained_model_name_or_path, options = {}) {
  let config = options.config;
  if (!config) {
    // Read the config to detect our model type before the Auto dispatch would
    // throw "Unsupported model type". This is the same AutoConfig from_pretrained
    // uses internally (same src/ instance), so the read is not duplicated wastefully.
    config = await AutoConfig.from_pretrained(pretrained_model_name_or_path, options);
    options = { ...options, config };
  }
  if (config?.model_type === MODEL_TYPE) {
    return GraniteSwitchForCausalLM.from_pretrained(pretrained_model_name_or_path, options);
  }
  return _autoFromPretrained(pretrained_model_name_or_path, options);
};

// ── High-level loader: native from_pretrained + the external-data wiring. ──
// The real Granite Switch decode graph keeps its weights in a `model.onnx.data`
// sidecar. transformers.js's automatic `use_external_data_format` path derives an
// UNDERSCORE name (`model.onnx_data`) and so misses the dotted `model.onnx.data`
// the graph actually references. We therefore pass `session_options.externalData`
// explicitly, named exactly as the graph's baked `location` (`model.onnx.data`).
// In the browser, transformers.js fetches that sidecar itself (string -> bytes);
// in Node, onnxruntime-node resolves the adjacent file from disk.
//
// `opts.externalDataName` overrides the sidecar basename (default: the model
// file's `<name>.onnx.data`, e.g. `model.onnx.data` / `model_int8.onnx.data`).
// `opts.modelFileName` overrides the onnx file stem (default `model`); the suffix
// for a given dtype (e.g. `_int8`) is appended by transformers.js, so the sidecar
// name follows the resolved file. Pass `external: false` for embedded graphs
// (small fixtures) where no sidecar exists.
export async function loadGraniteSwitch(repo, opts = {}) {
  const {
    dtype = "fp32",
    external = true,
    externalDataName,
    subfolder = "onnx",
    ...rest
  } = opts;

  const suffix = { fp32: "", int8: "_int8", q4: "_q4" }[dtype] ?? "";

  const session_options = { ...(rest.session_options ?? {}) };
  if (external && !session_options.externalData) {
    // Default model file stem + dtype suffix -> sidecar basename.
    const stem = (opts.modelFileName ?? "model") + suffix;
    const sidecar = externalDataName ?? `${stem}.onnx.data`;
    session_options.externalData = [{ path: sidecar, data: `${subfolder}/${sidecar}` }];
  }
  const model = await GraniteSwitchForCausalLM.from_pretrained(
    repo, { ...rest, dtype, subfolder, session_options },
  );

  // ── Optionally load the batched-prefill session alongside the decode session. ──
  // The model's only built-in session is the decode graph (`model`). We load the
  // prefill graph (onnx/prefill[_suffix].onnx) as a SECOND session and stash it at
  // self.sessions.prefill so graniteSwitchForward can run the whole prompt in one
  // pass. Use the framework's constructSessions (NOT a raw ort.InferenceSession):
  // it joins onnxruntime-web's init/inference serialization chain and reuses the
  // same dtype/device/external-data plumbing — a hand-rolled session would risk
  // "Session already started" and bypass that plumbing.
  //
  // Optional by design: a repo without a stateful prefill graph (or a load failure)
  // leaves self.sessions.prefill unset, and graniteSwitchForward falls back to the
  // per-token decode replay. So this never blocks loading.
  try {
    const prefillStem = "prefill" + suffix;
    const prefillOpts = {
      ...rest,                       // carries `device`, `config`, `progress_callback`, …
      dtype,
      subfolder,
      session_options: external
        ? { externalData: [{ path: `${prefillStem}.onnx.data`,
                             data: `${subfolder}/${prefillStem}.onnx.data` }] }
        : {},
    };
    // names map { prefill: "prefill" }: session key `prefill`, file stem `prefill`
    // (+ dtype suffix appended by getSession). Resolves onnx/prefill[_suffix].onnx.
    const prefillSessions = await constructSessions(repo, { prefill: "prefill" }, prefillOpts);
    model.sessions.prefill = prefillSessions.prefill;
  } catch (_) {
    // Older repo / missing artifact / unsupported — fall back to decode replay.
  }
  return model;
}
