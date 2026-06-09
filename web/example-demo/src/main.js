// SPDX-License-Identifier: Apache-2.0
// Granite Switch browser demo — NATIVE transformers.js load.
//
// Importing the shim (../../src/granite-switch-register.js) registers the
// `granite_switch` architecture and re-exports the transformers.js API bound to
// the single `src/` module instance the shim mutates. We then:
//   1. AutoModelForCausalLM.from_pretrained(repo)  -> a GraniteSwitchForCausalLM
//      (transformers.js owns the ONNX session + fetches the external-data sidecar)
//   2. tokenize CTI text (firing the adapter control token) via the chat template
//   3. greedy-generate and decode back to text.
//
// loadGraniteSwitch wires the `model.onnx.data` sidecar (see the shim).

import {
  loadGraniteSwitch,
  AutoModelForCausalLM, // re-exported (patched) — proves the Auto path too
  AutoTokenizer,        // dist/-instance tokenizer (see shim note re: bundling)
  envForTokenizer,      // the env paired with that AutoTokenizer
  env,
  Tensor,
} from "../../src/granite-switch-register.js";
import { GraniteSwitchTokenizer } from "../../src/granite-switch-tokenizer.js";
import MITRE_ID_TO_NAME from "./mitre_id_to_name.json";

// The adapter was trained on CTI sentences wrapped in this exact instruction
// format; sending bare text makes the model fall back to base behavior.
const USER_PROMPT = "What ATT&CK technique does the following CTI procedure sentence describe?";
// Match a MITRE technique id anywhere in the output, e.g. T1059 or T1059.001.
const TID_RE = /\bT\d{4}(?:\.\d{3})?\b/;
const EXAMPLES = [
  "The actor used PowerShell to download and execute a payload from a remote server.",
  "The malware created a scheduled task to maintain persistence across reboots.",
  "The threat actor dumped credentials from LSASS memory using a custom tool.",
  "The adversary encrypted files on the victim host and dropped a ransom note.",
  "The implant communicated with its command-and-control server over HTTPS on port 443.",
];

// Apply the same remote-fetch config to an env instance.
function configureRemote(e, remoteHost) {
  e.allowLocalModels = false;
  e.allowRemoteModels = true;
  e.remoteHost = remoteHost;
  e.remotePathTemplate = "{model}";
  e.useBrowserCache = false;
}

// Where the model lives. Two modes:
//   - VITE_MODEL_ID: an HF Hub repo id (e.g. "barha/granite-switch-4.0-350m-cti-onnx").
//     transformers.js resolves it natively at huggingface.co/<id>/resolve/main/.
//     This is the shippable Space configuration.
//   - VITE_REPO_BASE: a URL/path base to a repo served alongside the app
//     (default "/repo"), used for local dev. The app maps it to a flat
//     remoteHost + "{model}" template.
const MODEL_ID = import.meta.env.VITE_MODEL_ID || "";
const REPO_BASE = import.meta.env.VITE_REPO_BASE || "/repo";
const DTYPE = import.meta.env.VITE_DTYPE || "fp32";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const promptEl = $("prompt");
const baseOutEl = $("base-out");
const adapterOutEl = $("adapter-out");
const examplesEl = $("examples");
const runBtn = $("run");
const setStatus = (s) => (statusEl.textContent = s);

const progressWrap = $("progress-wrap");
const progressEl = $("progress");
const progressLabel = $("progress-label");

function fmtMB(bytes) {
  return (bytes / 1024 / 1024).toFixed(0) + " MB";
}

// transformers.js fires { status, file, progress (0-100), loaded, total } per
// downloaded file. We track each file's bytes and show the combined percentage —
// dominated by the ~1.3 GB weight sidecar. Cached files report instantly.
const fileBytes = new Map(); // file -> { loaded, total }
function onProgress(e) {
  if (e.status === "progress" && e.file && e.total) {
    fileBytes.set(e.file, { loaded: e.loaded ?? 0, total: e.total });
    let loaded = 0, total = 0;
    for (const v of fileBytes.values()) { loaded += v.loaded; total += v.total; }
    if (total > 0) {
      const pct = Math.min(100, Math.round((loaded / total) * 100));
      progressEl.removeAttribute("indeterminate");
      progressEl.value = pct;
      progressLabel.textContent = `downloading model · ${fmtMB(loaded)} / ${fmtMB(total)} (${pct}%)`;
    }
  } else if (e.status === "done" && e.file) {
    progressLabel.textContent = `loaded ${e.file}`;
  }
}
function hideProgress() { progressWrap.hidden = true; }

let model, tok;

async function init() {
  let name, fileBase;

  if (MODEL_ID) {
    // HF Hub mode (the Space config). transformers.js's DEFAULT remote resolution
    // (huggingface.co + the standard {model}/resolve/{revision} template) already
    // maps a repo id to its files, so we just enable remote models and leave the
    // defaults in place. fileBase is the resolve URL for our direct fetches of
    // gs_onnx.json / chat_template.jinja.
    name = MODEL_ID;
    fileBase = `https://huggingface.co/${MODEL_ID}/resolve/main`;
    for (const e of [env, envForTokenizer]) {
      e.allowLocalModels = false;
      e.allowRemoteModels = true;
      e.useBrowserCache = false; // dev convenience; the Hub is the source of truth
    }
  } else {
    // Local/served-repo mode. REPO_BASE is a URL served alongside the app; treat
    // it as a REMOTE fetch (transformers.js's file-existence Range probe only runs
    // on the remote branch — the local branch is for fs paths, not URLs). Point
    // remoteHost at REPO_BASE's parent + a flat "{model}" template so `<name>`
    // maps to `${parent}/<name>/<file>`.
    const baseUrl = new URL(REPO_BASE, window.location.href).href.replace(/\/$/, "");
    const parent = baseUrl.slice(0, baseUrl.lastIndexOf("/") + 1);
    name = baseUrl.slice(baseUrl.lastIndexOf("/") + 1);
    fileBase = baseUrl;
    const remoteHost = parent.replace(/\/$/, "");
    configureRemote(env, remoteHost);
    configureRemote(envForTokenizer, remoteHost);
  }

  setStatus(`loading model (${DTYPE})…`);
  progressLabel.textContent = "downloading model…";
  model = await loadGraniteSwitch(name, { dtype: DTYPE, progress_callback: onProgress });

  setStatus("loading tokenizer…");
  const meta = await (await fetch(`${fileBase}/gs_onnx.json`)).json();
  const chatTemplateText = await (await fetch(`${fileBase}/chat_template.jinja`)).text();
  tok = await GraniteSwitchTokenizer.load({
    modelId: name, // resolves under the configured remote host/template
    chatTemplateText,
    meta,
    // The tokenizer uses the dist/ AutoTokenizer + its paired env (see shim note).
    AutoTokenizer,
    env: envForTokenizer,
  });

  hideProgress();
  setStatus("ready · loaded natively on transformers.js");
  runBtn.disabled = false;
}

// One greedy generation. `useAdapter` toggles the control token (adapter ON/OFF).
async function generate(text, { useAdapter, maxNewTokens }) {
  const ids = tok.encode(text, {
    instruction: USER_PROMPT,
    // null => render WITHOUT the control token (the plain base model);
    // undefined => default to the first adapter (adapter ON).
    adapterName: useAdapter ? undefined : null,
  });
  const inputIds = new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
  const seq = await model.generate({
    inputs: inputIds,
    max_new_tokens: maxNewTokens,
    do_sample: false,
    num_beams: 1,
  });
  const out = Array.from(seq.tolist()[0], (v) => Number(v)).slice(ids.length);
  return tok.decode(out).trim();
}

// Run base (adapter OFF) and Granite Switch (adapter ON) on the same input,
// side by side — mirroring the reference Gradio Space.
async function compare() {
  const text = (promptEl.value || "").trim();
  if (!text) return;
  runBtn.disabled = true;
  baseOutEl.textContent = "…";
  adapterOutEl.textContent = "…";
  try {
    // Base: no adapter. Tends to ramble in prose — give it room.
    setStatus("running base model (adapter OFF)…");
    const baseRaw = await generate(text, { useAdapter: false, maxNewTokens: 48 });
    baseOutEl.textContent = baseRaw || "(no output)";

    // Adapter: fires the control token; trained to emit one technique id.
    setStatus("running Granite Switch (adapter ON)…");
    const adapterRaw = await generate(text, { useAdapter: true, maxNewTokens: 16 });
    const m = adapterRaw.match(TID_RE);
    if (m) {
      const tid = m[0];
      const name = MITRE_ID_TO_NAME[tid];
      adapterOutEl.innerHTML = name
        ? `<span class="tid">${tid}</span><br /><span class="tname">${escapeHtml(name)}</span>`
        : `<span class="tid">${tid}</span><br /><span class="tname">(name not in MITRE map)</span>`;
    } else {
      adapterOutEl.textContent = adapterRaw || "(no technique id returned)";
    }
    setStatus("done");
  } catch (e) {
    adapterOutEl.textContent = "Error: " + (e?.message ?? String(e));
    setStatus("error");
    console.error(e);
  } finally {
    runBtn.disabled = false;
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// Render the example chips (fill the textbox; the user clicks Compare).
for (const ex of EXAMPLES) {
  const b = document.createElement("button");
  b.className = "example";
  b.textContent = ex.length > 60 ? ex.slice(0, 57) + "…" : ex;
  b.title = ex;
  b.addEventListener("click", () => { promptEl.value = ex; });
  examplesEl.appendChild(b);
}

runBtn.addEventListener("click", compare);
init().catch((e) => {
  hideProgress();
  setStatus("load failed: " + (e?.message ?? String(e)));
  console.error(e);
});
