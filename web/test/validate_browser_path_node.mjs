// SPDX-License-Identifier: Apache-2.0
// Validate the BROWSER loading path headlessly in Node:
//
//   1. externalData wiring — load the externalized decode graph
//      (onnx/model.onnx + model.onnx.data) through GraniteSwitch.load by passing
//      the sidecar bytes explicitly as `decodeData`. This forces the same code
//      path the browser uses (onnxruntime-web does not auto-resolve sidecars);
//      onnxruntime-node here exercises the identical `externalData` option.
//      Assert greedy decode over a token-id prompt matches the Python golden.
//
//   2. tokenizer leg — load GraniteSwitchTokenizer (AutoTokenizer +
//      chat_template.jinja), encode a real CTI sentence, assert the adapter
//      control token is present, generate, decode, and assert the decoded text
//      matches the committed text golden.
//
// Env (all optional; sensible defaults point at scratch/gs-350m-repo):
//   REPO    = packaged repo dir (config.json + gs_onnx.json + tokenizer + onnx/)
//   GOLDEN  = token-id golden (prompt + golden ids)   [default web/test/golden_350m.json]
//   TGOLDEN = text golden (CTI text + decoded output) [default web/test/golden_text.json]

import * as ort from "onnxruntime-node";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { GraniteSwitch } from "../src/granite-switch.js";
import { GraniteSwitchTokenizer } from "../src/granite-switch-tokenizer.js";

const here = dirname(fileURLToPath(import.meta.url));
const repo = process.env.REPO || join(here, "..", "..", "scratch", "gs-350m-repo");
const golden = JSON.parse(readFileSync(process.env.GOLDEN || join(here, "golden_350m.json"), "utf8"));
const textGolden = JSON.parse(readFileSync(process.env.TGOLDEN || join(here, "golden_text.json"), "utf8"));

const meta = JSON.parse(readFileSync(join(repo, "gs_onnx.json"), "utf8"));
const decodePath = join(repo, "onnx", "model.onnx");
const decodeData = readFileSync(join(repo, "onnx", "model.onnx.data")); // Buffer -> Uint8Array view
const chatTemplate = readFileSync(join(repo, "chat_template.jinja"), "utf8");

let failures = 0;
const ok = (cond, msg) => { console.log((cond ? "PASS" : "FAIL") + ": " + msg); if (!cond) failures++; };

// ── 1. externalData wiring + token-id golden ───────────────────────────────
const gs = await GraniteSwitch.load(ort, {
  decodePath,
  decodeData,                   // forces the externalData branch
  meta,
  executionProviders: ["cpu"],
});
console.log("loaded decode graph via explicit externalData");

const maxNew = golden.golden.length - golden.prompt.length;
const gotIds = await gs.generate(golden.prompt, maxNew);
console.log("golden ids:", golden.golden.join(" "));
console.log("got ids   :", gotIds.join(" "));
ok(gotIds.length === golden.golden.length && gotIds.every((v, i) => v === golden.golden[i]),
   "externalData decode matches token-id golden");

// ── 2. tokenizer leg: CTI text -> ids -> generate -> text ──────────────────
const tok = await GraniteSwitchTokenizer.load({
  localModelPath: dirname(repo) + "/",
  modelName: repo.split(/[\\/]/).pop(),
  chatTemplateText: chatTemplate,
  meta,
});

const ids = tok.encode(textGolden.text, { instruction: textGolden.instruction });
const ctrl = meta.adapter_token_ids[0];
console.log("encoded prompt len:", ids.length, "control token", ctrl, "present:", ids.includes(ctrl));
ok(ids.includes(ctrl), "chat template injected the adapter control token");

const outIds = await gs.generate(ids, textGolden.max_new_tokens);
const text = tok.decode(outIds.slice(ids.length));
console.log("golden text:", JSON.stringify(textGolden.output));
console.log("got text   :", JSON.stringify(text));
ok(text === textGolden.output, "tokenizer end-to-end decode matches text golden");

if (failures === 0) {
  console.log("\nBROWSER_PATH_PASS: externalData wiring + tokenizer end-to-end correct.");
  process.exit(0);
} else {
  console.log(`\nBROWSER_PATH_FAIL: ${failures} check(s) failed.`);
  process.exit(1);
}
