// SPDX-License-Identifier: Apache-2.0
// Validate an HF-upload-ready repo (granite_switch.onnx.package output) ON
// transformers.js: load config.json + onnx/model_<dtype>.onnx from the repo,
// greedy-decode, and assert tokens match a Python golden. This is the check
// that the *uploadable artifact* — not just the raw export — runs correctly.
//
//   REPO=/path/to/hf-repo GOLDEN=/path/to/golden.json DT=int8 \
//     node test/validate_hub_repo.mjs
//
// DT selects the dtype (fp32 | int8 | q4). For granite-switch-4.1-3b, int8
// reproduces the fp32 golden exactly; q4 diverges.

import { readFileSync } from "node:fs";
import { dirname, basename } from "node:path";
import { GraniteSwitchTfjs } from "../src/granite-switch-tfjs.js";

const repo = process.env.REPO;
const golden = JSON.parse(readFileSync(process.env.GOLDEN, "utf8"));
const meta = JSON.parse(readFileSync(repo + "/gs_onnx.json", "utf8"));
const dtype = process.env.DT || "int8";

// transformers.js resolves modelName as a subdir UNDER localModelPath.
const parent = dirname(repo) + "/";
const name = basename(repo);
const gs = await GraniteSwitchTfjs.load({ localModelPath: parent, modelName: name, meta, dtype });
console.log("loaded:", gs.model.constructor.name, "dtype:", dtype);
const got = await gs.generate(golden.prompt, golden.golden.length - golden.prompt.length);
console.log("golden:", golden.golden.join(" "));
console.log("got   :", got.join(" "));
const agree = got.filter((v,i)=>v===golden.golden[i]).length / golden.golden.length;
console.log("agreement:", (agree*100).toFixed(0)+"%");
console.log(got.every((v,i)=>v===golden.golden[i]) ? "REPO_MATCH" : "REPO_DIVERGES");
