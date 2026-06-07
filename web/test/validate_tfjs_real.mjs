// SPDX-License-Identifier: Apache-2.0
// Real 4 B granite-switch-4.1-3b-preview loaded + run ON transformers.js.
//
// Same path as validate_tfjs.mjs (GATE 4b) but pointed at the real exported
// model directory instead of the tiny fixture. transformers.js's
// AutoModelForCausalLM.from_pretrained creates + owns the ONNX session
// (resolving the external model.onnx.data sidecar through its onnxruntime-node
// backend); we greedy-decode and assert the tokens match the Python golden
// produced from the HF backend (scratch/gs-onnx-3b/real_golden.json).
//
// Usage:
//   GS_REAL_DIR=/abs/path/to/gs-onnx-3b node test/validate_tfjs_real.mjs
// (defaults to ../../scratch/gs-onnx-3b relative to this file)

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { GraniteSwitchTfjs } from "../src/granite-switch-tfjs.js";

const here = dirname(fileURLToPath(import.meta.url));
const modelDir = process.env.GS_REAL_DIR
  ? resolve(process.env.GS_REAL_DIR)
  : resolve(here, "..", "..", "scratch", "gs-onnx-3b");

const meta = JSON.parse(readFileSync(join(modelDir, "gs_onnx.json"), "utf8"));
const golden = JSON.parse(readFileSync(join(modelDir, "real_golden.json"), "utf8"));

console.log("model dir :", modelDir);
console.log("external  :", meta.external_data === true);

const gs = await GraniteSwitchTfjs.load({
  localModelPath: modelDir + "/",
  modelName: "tfjs",
  meta,
  dtype: process.env.GS_DTYPE || "fp32",  // golden generated at fp32; int8 also matches for the real model
});

console.log("loaded ON transformers.js:", gs.model.constructor.name);
console.log("session owned by transformers.js:", gs.session.constructor.name);

const prompt = golden.prompt;
const want = golden.golden;
const got = await gs.generate(prompt, want.length - prompt.length);

console.log("golden :", want.join(" "));
console.log("got    :", got.join(" "));

const match = got.length === want.length && got.every((v, i) => v === want[i]);
if (match) {
  console.log("\nREAL GATE PASS: real 4B granite-switch runs ON transformers.js, matches Python golden.");
  process.exit(0);
} else {
  console.log("\nREAL GATE FAIL: token mismatch.");
  process.exit(1);
}
