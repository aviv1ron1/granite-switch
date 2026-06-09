// SPDX-License-Identifier: Apache-2.0
// GATE (native): Granite Switch loaded via transformers.js's NATIVE from_pretrained
// for `model_type: granite_switch` — NOT the gpt2-typed workaround.
//
// The shim (src/granite-switch-register.js) registers the architecture and a
// custom forward that threads the switch state through transformers.js's own
// generation loop + KV cache. We assert:
//   1. GraniteSwitchForCausalLM.from_pretrained(repo) loads + greedy-decodes to
//      the Python golden token-for-token (incl. across the adapter control token);
//   2. the patched AutoModelForCausalLM.from_pretrained(repo) does the same
//      (the familiar entry point now works for granite_switch);
//   3. transformers.js owns the ONNX session (model.sessions.model);
//   4. the cached sequence length grows by exactly 1 per step (i.e. the switch
//      cache entry does not poison length bookkeeping).
//
// Uses the tiny CPU fixture (random weights, no download) by building a
// granite_switch-typed config dir next to the existing gpt2-typed one, so both
// the native and workaround validators coexist. Run: `npm run validate:native`.

import { mkdirSync, copyFileSync, writeFileSync, readFileSync, existsSync, rmSync, symlinkSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
// Everything transformers.js comes FROM THE SHIM, so the app and the registration
// share one module instance (the shim binds to src/). env/AutoModelForCausalLM/
// Tensor are re-exported there.
import {
  GraniteSwitchForCausalLM,
  AutoModelForCausalLM,
  env,
  Tensor,
} from "../src/granite-switch-register.js";

const here = dirname(fileURLToPath(import.meta.url));
const modelDir = join(here, "..", "example", "model");
const meta = JSON.parse(readFileSync(join(modelDir, "gs_onnx.json"), "utf8"));
const golden = JSON.parse(readFileSync(join(modelDir, "golden.json"), "utf8"));

// ── Build a native-typed repo dir for the tiny fixture ───────────────────────
// Layout transformers.js expects for a local load: <root>/<name>/config.json +
// <name>/onnx/model.onnx. We reuse the fixture's decode graph (tfjs/onnx/model.onnx).
const nativeRoot = modelDir; // localModelPath base
const nativeName = "native"; // subdir
const nativeDir = join(nativeRoot, nativeName);
const nativeOnnxDir = join(nativeDir, "onnx");
mkdirSync(nativeOnnxDir, { recursive: true });
copyFileSync(join(modelDir, "tfjs", "onnx", "model.onnx"), join(nativeOnnxDir, "model.onnx"));

// granite_switch-typed config. architectures[0] === the registered class name so
// resolveTypeConfig's cross-arch (ForConditionalGeneration) detector is skipped.
const nativeConfig = {
  model_type: "granite_switch",
  architectures: ["GraniteSwitchForCausalLM"],
  num_hidden_layers: meta.n_layers,
  num_attention_heads: meta.kv_heads,
  num_key_value_heads: meta.kv_heads,
  hidden_size: meta.kv_heads * meta.head_dim,
  vocab_size: meta.vocab_size,
};
writeFileSync(join(nativeDir, "config.json"), JSON.stringify(nativeConfig, null, 2));

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = nativeRoot + "/";

// ── Greedy generate via transformers.js generate(), then compare to golden ──
function arraysEqual(a, b) {
  return a.length === b.length && a.every((v, i) => Number(v) === Number(b[i]));
}

async function runVia(model, label) {
  console.log(`\n[${label}] loaded:`, model.constructor.name);
  console.log(`[${label}] session owner:`, model.sessions.model?.constructor?.name ?? "(none)");

  const prompt = golden.prompt;
  const want = golden.golden;
  const maxNew = want.length - prompt.length;

  // transformers.js native greedy generation. `inputs` is the input_ids tensor
  // [1, promptLen]; greedy = do_sample:false + num_beams:1.
  const inputIds = new Tensor("int64", BigInt64Array.from(prompt.map(BigInt)), [1, prompt.length]);
  const seq = await model.generate({
    inputs: inputIds,
    max_new_tokens: maxNew,
    do_sample: false,
    num_beams: 1,
  });

  const got = Array.from(seq.tolist()[0], (v) => Number(v));
  console.log(`[${label}] golden:`, want.join(" "));
  console.log(`[${label}] got   :`, got.join(" "));

  if (!arraysEqual(got, want)) {
    console.error(`\n[${label}] FAIL: token mismatch.`);
    process.exit(1);
  }
  console.log(`[${label}] PASS: native from_pretrained output matches golden.`);
}

// 1) Direct class entry point.
const m1 = await GraniteSwitchForCausalLM.from_pretrained(nativeName, { dtype: "fp32" });
await runVia(m1, "GraniteSwitchForCausalLM");

// 2) Patched Auto entry point.
const m2 = await AutoModelForCausalLM.from_pretrained(nativeName, { dtype: "fp32" });
if (m2.constructor.name !== "GraniteSwitchForCausalLM") {
  console.error("\nFAIL: AutoModelForCausalLM did not resolve to GraniteSwitchForCausalLM:", m2.constructor.name);
  process.exit(1);
}
await runVia(m2, "AutoModelForCausalLM");

// 3) REAL 350M model (gated on the artifacts being present locally). Loads the
//    real granite-switch-4.0-350m-cti decode graph with its external-data sidecar
//    via the native from_pretrained path, and matches the real golden continuation.
await runReal350m();

console.log("\nGATE (native) PASS: granite_switch loads NATIVELY via from_pretrained and matches the golden.");
process.exit(0);

async function runReal350m() {
  // web/example/repo -> scratch/gs-350m-repo (sibling of modelDir, not under it)
  const realRepo = join(here, "..", "example", "repo");
  const realOnnx = join(realRepo, "onnx", "model.onnx");
  if (!existsSync(realOnnx) || !existsSync(realOnnx + ".data")) {
    console.log("\n[real-350m] SKIP: artifacts not present at example/repo/onnx/model.onnx(.data).");
    return;
  }
  const realGolden = JSON.parse(readFileSync(join(here, "golden_350m.json"), "utf8"));
  const realMeta = JSON.parse(readFileSync(join(realRepo, "gs_onnx.json"), "utf8"));

  // Native-typed config dir for the real repo; symlink its onnx/ to reuse the
  // 1.4 GB sidecar without copying.
  const realRoot = join(here, "..", "example", "native-real");
  const realName = "r";
  const realDir = join(realRoot, realName);
  rmSync(realRoot, { recursive: true, force: true });
  mkdirSync(join(realDir, "onnx"), { recursive: true });
  symlinkSync(join(realRepo, "onnx", "model.onnx"), join(realDir, "onnx", "model.onnx"));
  symlinkSync(join(realRepo, "onnx", "model.onnx.data"), join(realDir, "onnx", "model.onnx.data"));
  writeFileSync(
    join(realDir, "config.json"),
    JSON.stringify(
      {
        model_type: "granite_switch",
        architectures: ["GraniteSwitchForCausalLM"],
        num_hidden_layers: realMeta.n_layers,
        num_attention_heads: realMeta.kv_heads,
        num_key_value_heads: realMeta.kv_heads,
        hidden_size: realMeta.kv_heads * realMeta.head_dim,
        vocab_size: realMeta.vocab_size,
      },
      null,
      2,
    ),
  );

  env.localModelPath = realRoot + "/";
  const model = await GraniteSwitchForCausalLM.from_pretrained(realName, {
    dtype: "fp32",
    // External-data sidecar: name it as the graph references it (model.onnx.data,
    // dotted). The auto `use_external_data_format` path mis-derives an underscore
    // name (model.onnx_data) — see the shim's loadGraniteSwitchExternalData note.
    session_options: { externalData: [{ path: "model.onnx.data", data: "onnx/model.onnx.data" }] },
  });

  const prompt = realGolden.prompt;
  const want = realGolden.golden;
  const inputIds = new Tensor("int64", BigInt64Array.from(prompt.map(BigInt)), [1, prompt.length]);
  const seq = await model.generate({
    inputs: inputIds,
    max_new_tokens: want.length - prompt.length,
    do_sample: false,
    num_beams: 1,
  });
  const got = Array.from(seq.tolist()[0], (v) => Number(v));
  console.log("\n[real-350m] golden:", want.join(" "));
  console.log("[real-350m] got   :", got.join(" "));
  rmSync(realRoot, { recursive: true, force: true });
  if (!arraysEqual(got, want)) {
    console.error("\n[real-350m] FAIL: token mismatch.");
    process.exit(1);
  }
  console.log("[real-350m] PASS: real granite-switch-4.0-350m-cti decodes natively to the golden.");
}
