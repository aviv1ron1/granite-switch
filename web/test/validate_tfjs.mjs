// SPDX-License-Identifier: Apache-2.0
// GATE 4b: Granite Switch loaded and run ON transformers.js.
//
// Loads the exported decode graph through @huggingface/transformers'
// AutoModelForCausalLM.from_pretrained (which creates + owns the ONNX session
// via transformers.js's own backend), then greedy-decodes and asserts the
// tokens match the Python golden continuation.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { GraniteSwitchTfjs } from "../src/granite-switch-tfjs.js";

const here = dirname(fileURLToPath(import.meta.url));
const exampleDir = join(here, "..", "example");
const meta = JSON.parse(readFileSync(join(exampleDir, "model", "gs_onnx.json"), "utf8"));
const golden = JSON.parse(readFileSync(join(exampleDir, "model", "golden.json"), "utf8"));

const gs = await GraniteSwitchTfjs.load({
  localModelPath: join(exampleDir, "model") + "/",
  modelName: "tfjs",
  meta,
  dtype: "fp32",  // tiny fixture has only the fp32 graph; its int8 diverges (random weights)
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
  console.log("\nGATE 4b PASS: granite-switch runs ON transformers.js, output matches Python golden.");
  process.exit(0);
} else {
  console.log("\nGATE 4b FAIL: token mismatch.");
  process.exit(1);
}
