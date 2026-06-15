// SPDX-License-Identifier: Apache-2.0
// Granite Switch browser demo — NATIVE transformers.js load, ALL adapters.
//
// "One model, many skills": a SINGLE 350m checkpoint embeds three LoRA adapters,
// each fired by its own control token — no weight reloading between tasks. The UI
// is one tab per adapter; each tab runs the same input twice (adapter OFF = base
// prose, adapter ON = the adapter's structured output) so the lift is visible.
//
// Importing the shim (../../src/granite-switch-register.js) registers the
// `granite_switch` architecture and re-exports the transformers.js API bound to
// the single `src/` module instance the shim mutates. We then:
//   1. AutoModelForCausalLM.from_pretrained(repo)  -> a GraniteSwitchForCausalLM
//      (transformers.js owns the ONNX session + fetches the external-data sidecar)
//   2. tokenize the framed prompt (firing the chosen adapter's control token)
//   3. greedy-generate and decode back to text.
//
// loadGraniteSwitch wires the `model[_int8].onnx.data` sidecar (see the shim);
// the dtype (fp32/int8) is chosen at build time via VITE_DTYPE.

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

// ── Adapter output recognizers ────────────────────────────────────────────────
// Match a MITRE technique id anywhere in the output, e.g. T1059 or T1059.001.
const TID_RE = /\bT\d{4}(?:\.\d{3})?\b/;
// The genai-attack-vector adapter emits one of these 14 closed-set slugs.
const ATTACK_SLUGS = [
  "rce", "xss", "deepfake", "misinformation", "data-exfiltration", "dos",
  "privacy-violation", "auth-bypass", "path-traversal", "ssrf", "prompt-injection",
  "deserialization", "command-injection", "info-disclosure",
];
// Longest-first alternation so e.g. "data-exfiltration" wins over a bare substring.
const SLUG_RE = new RegExp(
  "\\b(" + [...ATTACK_SLUGS].sort((a, b) => b.length - a.length).join("|") + ")\\b",
  "i",
);
// text-to-json: the schema travels IN the prompt after this exact preamble (the
// adapter was trained on `${query}${SCHEMA_PREAMBLE}${schema}`).
const SCHEMA_PREAMBLE = "\n\nRespond with a JSON object conforming to this schema:\n";

// ── Per-adapter configuration ─────────────────────────────────────────────────
// `name` MUST equal the adapter name in gs_onnx.json so the right control token
// fires. `render(raw)` returns { html } (trusted, pre-escaped) or { text }.
const ADAPTERS = [
  {
    name: "cti-technique-mapping",
    label: "CTI → ATT&CK",
    blurb: "Maps a cyber-threat-intelligence procedure sentence to the one matching MITRE ATT&CK technique ID.",
    instruction: "What ATT&CK technique does the following CTI procedure sentence describe?",
    wrapTag: "cti",
    inputLabel: "CTI text",
    placeholder: "Describe the observed adversary behavior…",
    baseMaxNewTokens: 48,
    adapterMaxNewTokens: 16,
    examples: [
      "The actor used PowerShell to download and execute a payload from a remote server.",
      "The malware created a scheduled task to maintain persistence across reboots.",
      "The threat actor dumped credentials from LSASS memory using a custom tool.",
      "The adversary encrypted files on the victim host and dropped a ransom note.",
      "The implant communicated with its command-and-control server over HTTPS on port 443.",
    ],
    render: renderCti,
  },
  {
    name: "genai-attack-vector",
    label: "GenAI attack vector",
    blurb: "Classifies a GenAI security incident into one attack-vector label (14-way closed set).",
    instruction: "What attack vector does the following GenAI security incident describe?",
    wrapTag: "incident",
    inputLabel: "Incident description",
    placeholder: "Describe the GenAI security incident…",
    baseMaxNewTokens: 48,
    adapterMaxNewTokens: 16,
    examples: [
      "A user pasted text containing hidden instructions that overrode the model's system prompt and made it ignore its safety rules.",
      "An attacker uploaded a crafted document that, when summarized, caused the assistant to call an internal tool and leak API keys in its response.",
      "The chatbot rendered model output containing a script tag that executed in the support agent's browser.",
      "A synthetic video of the CEO authorizing a wire transfer was used to trick the finance team into approving it.",
      "The model was coaxed into making a server-side request to the cloud metadata endpoint and returned the instance credentials.",
    ],
    render: renderAttackVector,
  },
  {
    name: "text-to-json",
    label: "Text → JSON",
    blurb: "Turns a natural-language request plus a JSON schema into a populated JSON object that conforms to the schema.",
    instruction: null, // no instruction line, no tag — schema preamble instead
    wrapTag: null,
    inputLabel: "Request",
    placeholder: "Describe what you want as JSON…",
    baseMaxNewTokens: 192, // JSON objects are long; give both columns room
    adapterMaxNewTokens: 192,
    // text-to-json has a second input field (the schema). buildContent joins them
    // in the trained format; the active schema is read at generate() time.
    schema: '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}, "email": {"type": "string"}}}',
    buildContent: ({ text }, schema) => `${text}${SCHEMA_PREAMBLE}${schema}`,
    examples: [
      {
        query: "Alice Johnson is 30 years old and her email is alice@example.com.",
        schema: '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}, "email": {"type": "string"}}}',
      },
      {
        query: "The order has 3 items totaling $59.97 and ships to Berlin.",
        schema: '{"type": "object", "properties": {"item_count": {"type": "number"}, "total_usd": {"type": "number"}, "city": {"type": "string"}}}',
      },
      {
        query: "Bug report: the login button is broken on mobile, priority is high, filed against the auth component.",
        schema: '{"type": "object", "properties": {"title": {"type": "string"}, "priority": {"type": "string"}, "component": {"type": "string"}}}',
      },
      {
        query: "The meeting is on 2026-07-01 at 14:00 in Room B with Sam and Priya.",
        schema: '{"type": "object", "properties": {"date": {"type": "string"}, "time": {"type": "string"}, "room": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"}}}}',
      },
    ],
    render: renderJson,
  },
];

// ── Output renderers ──────────────────────────────────────────────────────────
function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function renderCti(raw) {
  const m = raw.match(TID_RE);
  if (!m) return { text: raw || "(no technique id returned)" };
  const tid = m[0];
  const name = MITRE_ID_TO_NAME[tid];
  return {
    html: `<span class="tid">${tid}</span><br />` +
      `<span class="tname">${name ? escapeHtml(name) : "(name not in MITRE map)"}</span>`,
  };
}

function renderAttackVector(raw) {
  const m = raw.match(SLUG_RE);
  if (!m) return { text: raw || "(no attack vector returned)" };
  return { html: `<span class="slug">${escapeHtml(m[1].toLowerCase())}</span>` };
}

function renderJson(raw) {
  // The adapter may emit trailing tokens after the object; extract the first
  // balanced {…} span before parsing so a stray suffix doesn't break JSON.parse.
  const s = raw.indexOf("{");
  const e = raw.lastIndexOf("}");
  if (s !== -1 && e > s) {
    try {
      const obj = JSON.parse(raw.slice(s, e + 1));
      return { html: `<pre class="json">${escapeHtml(JSON.stringify(obj, null, 2))}</pre>` };
    } catch (_) { /* fall through to raw text */ }
  }
  return { text: raw || "(no JSON returned)" };
}

// ── Build config / model location ─────────────────────────────────────────────
// Where the model lives. Two modes:
//   - VITE_MODEL_ID: an HF Hub repo id (e.g. "barha/granite-switch-4.0-350m-demo-onnx").
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
const schemaEl = $("schema");
const schemaWrapEl = $("schema-wrap");
const inputLabelEl = $("input-label");
const tabBlurbEl = $("tab-blurb");
const tabsEl = $("tabs");
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
// dominated by the weight sidecar. Cached files report instantly.
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
let activeAdapter = ADAPTERS[0];

// Apply the same remote-fetch config to an env instance (local-dev mode).
function configureRemote(e, remoteHost) {
  e.allowLocalModels = false;
  e.allowRemoteModels = true;
  e.remoteHost = remoteHost;
  e.remotePathTemplate = "{model}";
  e.useBrowserCache = false;
}

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
  // ONE shared load: all three adapters live in this single model; the control
  // token (chosen per generation) selects which fires. No per-tab reload.
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

// ── Live inference indicator ───────────────────────────────────────────────────
// ORT-web runs the decode on the main thread; each token is one `await session.run`,
// which IS a real event-loop turn (transformers.js's generate loop awaits forward()
// between every token — see modeling_utils.js). So a `streamer` that touches the DOM
// per token paints, and a requestAnimationFrame ticker keeps a spinner moving even
// while a single token is in flight. Together they make it obvious the page is busy
// decoding, not hung.
const SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// Animates a "running" line into `el` (spinner · elapsed · token count) until
// stop() is called. Independent of the decode loop, so it moves even between tokens.
function startTicker(el, phaseLabel) {
  let raf = 0, frame = 0, tokens = 0;
  const t0 = performance.now();
  const tick = () => {
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    el.textContent = `${SPIN[frame++ % SPIN.length]} ${phaseLabel} · ${secs}s · ${tokens} tokens`;
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  return {
    bump: () => { tokens++; },
    stop: () => cancelAnimationFrame(raf),
  };
}

// One greedy generation for the active adapter. `useAdapter` toggles the control
// token (adapter ON/OFF). `outEl` (optional) receives a live "decoding…" indicator
// driven by the per-token streamer.
async function generate(adapter, { useAdapter, outEl, phaseLabel }) {
  const text = (promptEl.value || "").trim();
  const encodeOpts = {
    // Pass the EXPLICIT adapter name when ON (not `undefined`, which would always
    // default to the first adapter and make every tab fire CTI); null => base.
    adapterName: useAdapter ? adapter.name : null,
    instruction: adapter.instruction || undefined,
    wrapTag: adapter.wrapTag,
  };
  if (adapter.buildContent) {
    const schema = (schemaEl.value || "").trim();
    encodeOpts.buildContent = (a) => adapter.buildContent(a, schema);
  }
  const ids = tok.encode(text, encodeOpts);
  const inputIds = new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);

  // Drive a live "decoding…" indicator into the output card while tokens stream.
  const ticker = outEl ? startTicker(outEl, phaseLabel ?? "decoding") : null;
  // A minimal streamer: transformers.js calls put(all_ids) once for the prompt then
  // put(new_ids) per generated step, and end() at the finish. We only count steps.
  let prompted = false;
  const streamer = ticker
    ? {
        put() { if (!prompted) { prompted = true; return; } ticker.bump(); },
        end() {},
      }
    : null;

  try {
    const seq = await model.generate({
      inputs: inputIds,
      max_new_tokens: useAdapter ? adapter.adapterMaxNewTokens : adapter.baseMaxNewTokens,
      do_sample: false,
      num_beams: 1,
      ...(streamer ? { streamer } : {}),
    });
    const out = Array.from(seq.tolist()[0], (v) => Number(v)).slice(ids.length);
    return tok.decode(out).trim();
  } finally {
    ticker?.stop();
  }
}

function applyRender(el, r) {
  if (r.html != null) el.innerHTML = r.html;
  else el.textContent = r.text;
}

// Run base (adapter OFF) and Granite Switch (adapter ON) on the same input,
// side by side, for whichever adapter tab is active.
async function compare() {
  const adapter = activeAdapter;
  if ((promptEl.value || "").trim() === "") return;
  runBtn.disabled = true;
  const runLabel = runBtn.textContent;
  runBtn.textContent = "Running…";
  setTabsDisabled(true);
  baseOutEl.textContent = "…";
  adapterOutEl.textContent = "…";
  try {
    // Base: no adapter. Tends to ramble in prose — give it room. The output card
    // shows a live spinner · elapsed · token-count line while it decodes.
    setStatus("running base model (adapter OFF)…");
    const baseRaw = await generate(adapter, {
      useAdapter: false, outEl: baseOutEl, phaseLabel: "base model decoding",
    });
    baseOutEl.textContent = baseRaw || "(no output)";

    // Adapter: fires the control token; trained for this task's output shape.
    setStatus(`running Granite Switch (${adapter.label})…`);
    const adapterRaw = await generate(adapter, {
      useAdapter: true, outEl: adapterOutEl, phaseLabel: `${adapter.label} decoding`,
    });
    applyRender(adapterOutEl, adapter.render(adapterRaw));
    setStatus("done");
  } catch (e) {
    adapterOutEl.textContent = "Error: " + (e?.message ?? String(e));
    setStatus("error");
    console.error(e);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = runLabel;
    setTabsDisabled(false);
  }
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function setTabsDisabled(disabled) {
  for (const b of tabsEl.children) b.disabled = disabled;
}

function renderExampleChips(adapter) {
  examplesEl.replaceChildren();
  for (const ex of adapter.examples) {
    const isObj = typeof ex === "object";
    const labelText = isObj ? ex.query : ex;
    const b = document.createElement("button");
    b.className = "example";
    b.textContent = labelText.length > 60 ? labelText.slice(0, 57) + "…" : labelText;
    b.title = labelText;
    b.addEventListener("click", () => {
      promptEl.value = isObj ? ex.query : ex;
      if (isObj && ex.schema) schemaEl.value = ex.schema;
    });
    examplesEl.appendChild(b);
  }
}

function selectTab(adapter) {
  activeAdapter = adapter;
  for (const b of tabsEl.children) {
    b.setAttribute("aria-selected", String(b.dataset.name === adapter.name));
  }
  inputLabelEl.textContent = adapter.inputLabel;
  promptEl.placeholder = adapter.placeholder;
  promptEl.value = "";
  tabBlurbEl.textContent = adapter.blurb;

  const isJson = adapter.name === "text-to-json";
  schemaWrapEl.hidden = !isJson;
  if (isJson) schemaEl.value = adapter.schema;

  baseOutEl.textContent = "—";
  adapterOutEl.textContent = "—";
  renderExampleChips(adapter);
}

// Build the tab strip from ADAPTERS.
for (const a of ADAPTERS) {
  const b = document.createElement("button");
  b.className = "tab";
  b.dataset.name = a.name;
  b.textContent = a.label;
  b.setAttribute("role", "tab");
  b.addEventListener("click", () => selectTab(a));
  tabsEl.appendChild(b);
}
selectTab(ADAPTERS[0]);

runBtn.addEventListener("click", compare);
init().catch((e) => {
  hideProgress();
  setStatus("load failed: " + (e?.message ?? String(e)));
  console.error(e);
});
