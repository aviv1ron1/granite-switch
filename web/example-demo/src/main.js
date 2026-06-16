// SPDX-License-Identifier: Apache-2.0
// Granite Switch browser demo — UI thread. ALL model work runs in worker.js.
//
// "One model, many skills": a SINGLE 350m checkpoint embeds three aLoRA adapters,
// each fired by its own control token (placed before the assistant turn) — no weight
// reloading between tasks. The UI
// is one tab per adapter; each tab runs the same input twice (adapter OFF = base
// prose, adapter ON = the adapter's structured output) so the lift is visible.
//
// Architecture (transformers.js-examples pattern, e.g. smollm-webgpu): the model is
// loaded and run inside a dedicated Web Worker (./worker.js). The main thread does
// ONLY UI — it posts {load,generate} messages and renders {progress,ready,update,
// complete} messages back. Tokens stream into the output column live as they decode,
// so generation never freezes the page. dtype (fp32/int8) is chosen at build time via
// VITE_DTYPE and forwarded to the worker.

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
// Each adapter's control token is its name wrapped in `<|…|>` (aLoRA adapters; the
// chat template places it right before the assistant turn, not at the sequence start).
// Kept as a function so the cards and the raw-prompt highlight derive the exact
// spelling from `name`.
const controlTokenOf = (a) => `<|${a.name}|>`;

const ADAPTERS = [
  {
    name: "cti-technique-mapping",
    label: "CTI → ATT&CK",
    repo: "barha/granite-cti-technique-mapping-350m-alora",
    blurb: "Maps a cyber-threat-intelligence procedure sentence to the one matching MITRE ATT&CK technique ID.",
    outputDesc: "→ a MITRE ATT&CK technique ID (e.g. T1059.001)",
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
    repo: "barha/granite-genai-attack-vector-350m-alora",
    blurb: "Classifies a GenAI security incident into one attack-vector label (14-way closed set).",
    outputDesc: "→ one of 14 attack-vector labels (e.g. prompt-injection)",
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
    repo: "barha/granite-text-to-json-350m-alora",
    blurb: "Turns a natural-language request plus a JSON schema into a populated JSON object that conforms to the schema.",
    outputDesc: "→ a JSON object conforming to your schema",
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
const adapterCardsEl = $("adapter-cards");
const promptViewEl = $("prompt-view");
const promptBaseEl = $("prompt-base");
const promptAdapterEl = $("prompt-adapter");
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

let activeAdapter = ADAPTERS[0];

// Compute where the model lives (kept identical to the previous in-thread logic).
// Returns a plain object safe to postMessage to the worker.
//   - hub mode (VITE_MODEL_ID): transformers.js resolves the repo id natively at
//     huggingface.co/<id>/resolve/main/. fileBase is the resolve URL the worker also
//     uses for its direct fetches of gs_onnx.json / chat_template.jinja.
//   - local mode (VITE_REPO_BASE, default "/repo"): a URL served alongside the app,
//     treated as a remote fetch with remoteHost = REPO_BASE's parent + "{model}".
// Resolve the Hub model's current commit SHA so loads can be PINNED to it. This is
// what makes browser caching safe: transformers.js keys its Cache Storage by URL, so
// pinning every fetch to /resolve/<sha>/ means a future re-export (new SHA) lands on
// fresh URLs and naturally misses the stale cache — no manual cache wipe needed.
// Best-effort: on any failure we fall back to the "main" branch (uncached-but-correct).
async function resolveRevision(modelId) {
  try {
    const r = await fetch(`https://huggingface.co/api/models/${modelId}/revision/main`);
    if (r.ok) {
      const sha = (await r.json())?.sha;
      if (typeof sha === "string" && sha.length) return sha;
    }
  } catch (_) { /* offline / API change — fall back to main */ }
  return "main";
}

async function computeModelLocation() {
  if (MODEL_ID) {
    const revision = await resolveRevision(MODEL_ID);
    return {
      mode: "hub", name: MODEL_ID, revision,
      // Pin direct fetches (gs_onnx.json / chat_template.jinja) to the same revision
      // so they share the revision-keyed cache with the model files.
      fileBase: `https://huggingface.co/${MODEL_ID}/resolve/${revision}`,
    };
  }
  const baseUrl = new URL(REPO_BASE, window.location.href).href.replace(/\/$/, "");
  const parent = baseUrl.slice(0, baseUrl.lastIndexOf("/") + 1);
  const name = baseUrl.slice(baseUrl.lastIndexOf("/") + 1);
  return { mode: "local", name, fileBase: baseUrl, remoteHost: parent.replace(/\/$/, "") };
}

// The model engine. Created once; all model work happens off the main thread.
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

// Per-column streaming state. Only one generation leg is in flight at a time
// (compare() awaits base then adapter), so a small per-"which" map is unambiguous.
const COLUMN = { base: () => baseOutEl, adapter: () => adapterOutEl };
const accum = { base: "", adapter: "" };
const tickers = { base: null, adapter: null };
const resolvers = { base: null, adapter: null }; // resolve the in-flight leg's promise
// The templated prompt (after chat template) of the most recent run, per leg — fed
// into the "View raw prompt" panel once both legs of a Compare have completed.
const lastPrompts = { base: "", adapter: "" };
// The raw generated text of the most recent run, per leg — appended after the prompt
// in the "View raw prompt" panel so the full sequence (prompt + completion) is visible.
const lastOutputs = { base: "", adapter: "" };

function finishLeg(which) {
  tickers[which]?.stop();
  tickers[which] = null;
  resolvers[which]?.();
  resolvers[which] = null;
}

worker.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case "progress":
      onProgress(m.data); // transformers.js event, nested under .data
      break;
    case "ready": {
      hideProgress();
      // Show the live execution provider. WebGPU runs the matmuls on the GPU; WASM
      // is the CPU fallback (and notes single-threaded when not cross-origin isolated).
      const epNote =
        m.device === "webgpu" ? " · webgpu" : " · wasm" + (m.threaded ? "" : " · single-threaded");
      setStatus("ready · loaded on transformers.js" + epNote);
      if (m.device !== "webgpu" && !m.threaded) {
        console.warn(
          "Page is NOT cross-origin isolated → onnxruntime-web is single-threaded. " +
          "Generation will be slow. (coi-serviceworker should restore isolation after " +
          "a one-time reload.)",
        );
      }
      runBtn.disabled = false;
      break;
    }
    case "update": {
      // Append the incremental decoded chunk — tokens appear live in the column.
      accum[m.which] += m.text;
      COLUMN[m.which]().textContent = accum[m.which];
      tickers[m.which]?.bump(m.numTokens, m.tps);
      break;
    }
    case "complete":
      // Base column keeps the streamed prose; adapter column swaps to the rendered
      // view (MITRE id / slug / JSON) built from the authoritative full text.
      if (m.which === "adapter") applyRender(adapterOutEl, activeAdapter.render(m.raw));
      else baseOutEl.textContent = accum.base || m.raw || "(no output)";
      // Stash this leg's templated prompt (after chat template) for the raw-prompt viewer.
      if (typeof m.prompt === "string") lastPrompts[m.which] = m.prompt;
      // Stash the leg's raw completion too, so the viewer can show prompt + output.
      lastOutputs[m.which] = accum[m.which] || m.raw || "";
      finishLeg(m.which);
      break;
    case "error":
      if (m.which) COLUMN[m.which]().textContent = "Error: " + m.message;
      else setStatus("worker error: " + m.message);
      console.error("worker error:", m.message);
      finishLeg(m.which || "base");
      finishLeg("adapter");
      break;
  }
};
worker.onerror = (e) => {
  hideProgress();
  setStatus("worker failed to load: " + (e.message || "see console"));
  console.error(e);
};

// Kick off the one-time model load. Resolve the model revision first (hub mode) so the
// load can be pinned to a commit SHA — cached across reloads, auto-invalidated on
// re-export.
setStatus(`loading model (${DTYPE})…`);
progressLabel.textContent = "downloading model…";
computeModelLocation().then((loc) => {
  worker.postMessage({ type: "load", payload: { ...loc, dtype: DTYPE } });
});

// ── Live inference indicator ───────────────────────────────────────────────────
// The worker streams a decoded chunk per token; the column shows the text appearing
// live. Alongside it, this ticker animates a spinner + elapsed + token-count + tps on
// the STATUS line (not the column — the streamed text owns the column). The rAF loop
// keeps the spinner moving smoothly between token messages; bump() folds in the live
// token count / throughput reported by the worker.
const SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

function startTicker(phaseLabel) {
  let raf = 0, frame = 0, tokens = 0, tps = 0;
  const t0 = performance.now();
  const tick = () => {
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    const tpsStr = tps > 0 ? ` · ${tps.toFixed(1)} tok/s` : "";
    statusEl.textContent = `${SPIN[frame++ % SPIN.length]} ${phaseLabel} · ${secs}s · ${tokens} tok${tpsStr}`;
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  return {
    bump: (n, t) => { if (typeof n === "number") tokens = n; if (typeof t === "number") tps = t; },
    stop: () => cancelAnimationFrame(raf),
  };
}

function applyRender(el, r) {
  if (r.html != null) el.innerHTML = r.html;
  else el.textContent = r.text;
}

// Render the two templated prompts (after chat template) into the "View raw prompt"
// panel, highlighting the active adapter's control token so the injected switch is
// obvious. `controlToken` is highlighted only in the adapter leg (it never appears in
// the base leg). Escapes first, then wraps the (already-escaped) token spelling.
function renderPromptView(adapter) {
  const token = controlTokenOf(adapter); // e.g. <|cti-technique-mapping|>
  const escToken = escapeHtml(token);
  const highlight = (s) =>
    escapeHtml(s).split(escToken).join(`<span class="ctrl-tok">${escToken}</span>`);
  // Render one leg as: highlighted/escaped prompt, then the model's completion wrapped
  // in a <span class="completion"> so it's visually distinct from the fed-in prompt.
  const leg = (prompt, output, hl) => {
    if (!prompt) return "—";
    const head = hl ? highlight(prompt) : escapeHtml(prompt);
    if (!output) return head;
    return head + `<span class="completion">${escapeHtml(output)}</span>`;
  };

  promptBaseEl.innerHTML = leg(lastPrompts.base, lastOutputs.base, false);
  promptAdapterEl.innerHTML = leg(lastPrompts.adapter, lastOutputs.adapter, true);
  promptViewEl.hidden = false;
}

// Build the per-adapter info cards in the "How the switch works" explainer.
function renderAdapterCards() {
  adapterCardsEl.replaceChildren();
  for (const a of ADAPTERS) {
    const card = document.createElement("div");
    card.className = "adapter-card";
    const link = a.repo
      ? `<br /><a class="ac-link" href="https://huggingface.co/${escapeHtml(a.repo)}" target="_blank" rel="noopener">${escapeHtml(a.repo)}</a>`
      : "";
    card.innerHTML =
      `<span class="ac-label">${escapeHtml(a.label)}</span>` +
      `<span class="tok">${escapeHtml(controlTokenOf(a))}</span><br />` +
      `${escapeHtml(a.blurb)}<br />` +
      `<span class="ac-out">${escapeHtml(a.outputDesc)}</span>` +
      link;
    adapterCardsEl.appendChild(card);
  }
}

// Post one generation leg to the worker and resolve when its "complete"/"error"
// message arrives. `useAdapter` toggles the control token (adapter ON/OFF).
function runLeg(which, adapter, useAdapter, phaseLabel) {
  const text = (promptEl.value || "").trim();
  const payload = {
    which,
    text,
    // Pass the EXPLICIT adapter name when ON (not undefined, which would default to
    // the first adapter and make every tab fire CTI); null => base model.
    adapterName: useAdapter ? adapter.name : null,
    instruction: adapter.instruction || undefined,
    wrapTag: adapter.wrapTag,
    maxNewTokens: useAdapter ? adapter.adapterMaxNewTokens : adapter.baseMaxNewTokens,
  };
  // text-to-json: buildContent is a main-thread function that can't cross postMessage,
  // so build the trained `${query}${PREAMBLE}${schema}` string here and send it.
  if (adapter.buildContent) {
    payload.content = adapter.buildContent({ text }, (schemaEl.value || "").trim());
  }

  accum[which] = "";
  COLUMN[which]().textContent = "";
  tickers[which] = startTicker(phaseLabel);
  const done = new Promise((res) => { resolvers[which] = res; });
  worker.postMessage({ type: "generate", payload });
  return done;
}

// Run base (adapter OFF) then Granite Switch (adapter ON) on the same input, side by
// side, for whichever adapter tab is active. Sequential so the single model serves one
// leg at a time and the per-column streaming state stays unambiguous.
async function compare() {
  const adapter = activeAdapter;
  if ((promptEl.value || "").trim() === "") return;
  runBtn.disabled = true;
  const runLabel = runBtn.textContent;
  runBtn.textContent = "Running…";
  setTabsDisabled(true);
  baseOutEl.textContent = "…";
  adapterOutEl.textContent = "…";
  lastPrompts.base = lastPrompts.adapter = "";
  lastOutputs.base = lastOutputs.adapter = "";
  try {
    await runLeg("base", adapter, false, "base model decoding");
    await runLeg("adapter", adapter, true, `${adapter.label} decoding`);
    renderPromptView(adapter); // both legs' templated prompts are now captured
    setStatus("done");
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
  // The raw-prompt panel reflects the last run of the previous tab; hide it until
  // this tab is run so it can't show a mismatched (other-adapter) prompt.
  promptViewEl.hidden = true;
  lastPrompts.base = lastPrompts.adapter = "";
  lastOutputs.base = lastOutputs.adapter = "";
  renderExampleChips(adapter);
}

// Per-adapter info cards in the explainer (static; built once from ADAPTERS).
renderAdapterCards();

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
// The worker load was kicked off above (worker.postMessage({type:"load",...})); its
// "ready" message enables the Run button and its "progress" messages drive the bar.
