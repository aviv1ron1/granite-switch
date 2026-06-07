// SPDX-License-Identifier: Apache-2.0
// GATE 4 (headless): run the exported Granite Switch graphs through ONNX Runtime
// in Node and confirm greedy decode matches the Python golden continuation.
//
// Uses onnxruntime-node (same ORT engine as onnxruntime-web). Passing here means
// the exported graphs + the JS decode/state-threading logic are correct outside
// Python — the browser uses identical code with onnxruntime-web.

import * as ort from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { GraniteSwitch } from "../src/granite-switch.js";

const here = dirname(fileURLToPath(import.meta.url));
const modelDir = join(here, "..", "example", "model");

const meta = JSON.parse(readFileSync(join(modelDir, "gs_onnx.json"), "utf8"));
const golden = JSON.parse(readFileSync(join(modelDir, "golden.json"), "utf8"));

const gs = await GraniteSwitch.load(ort, {
  prefillPath: join(modelDir, "prefill.onnx"),
  decodePath: join(modelDir, "decode.onnx"),
  meta,
  executionProviders: ["cpu"], // onnxruntime-node
});

const prompt = golden.prompt;
const want = golden.golden;
const maxNew = want.length - prompt.length;

const got = await gs.generate(prompt, maxNew);

console.log("prompt :", prompt.join(" "));
console.log("golden :", want.join(" "));
console.log("got    :", got.join(" "));

const match = got.length === want.length && got.every((v, i) => v === want[i]);
if (match) {
  console.log("\nGATE 4 PASS: browser-runtime greedy decode matches Python golden.");
  process.exit(0);
} else {
  console.log("\nGATE 4 FAIL: token mismatch.");
  process.exit(1);
}
