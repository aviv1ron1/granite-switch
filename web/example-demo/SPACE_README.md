---
title: Granite Switch 350M — one model, many skills
emoji: 🪶
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: apache-2.0
header: mini
app_file: index.html
# Cross-origin isolation enables the multi-threaded onnxruntime-web WASM build.
# NOTE: HF Static Spaces honor cross_origin_opener_policy but do NOT emit
# cross_origin_embedder_policy, so these alone don't make the page isolated. The
# bundled coi-serviceworker.js restores isolation client-side (COEP: credentialless)
# after a one-time reload. These keys are kept for any host that does honor them.
cross_origin_embedder_policy: require-corp
cross_origin_opener_policy: same-origin
cross_origin_resource_policy: cross-origin
models:
  - barha/granite-switch-4.0-350m-demo-onnx
---

# Granite Switch 4.0 350M — one model, many skills

A **fully client-side** (in-browser) demo of
[`barha/granite-switch-4.0-350m-demo-onnx`](https://huggingface.co/barha/granite-switch-4.0-350m-demo-onnx):
**one** 350M base model (`ibm-granite/granite-4.0-350m`) with **three embedded LoRA
adapters** in a single checkpoint. A single **control token** switches between skills
per request — no weight reloading, no separate models to host.

Each tab runs the **same base model twice on the same input** and shows the results
side by side:

- **Adapter OFF** — the plain base model. It answers in prose and rarely produces a
  clean, structured result.
- **Adapter ON** — the embedded adapter for that skill is fired by a single control
  token, and the model emits the task's structured output.

The three skills:

| Tab | Adapter | Input | Output |
| --- | --- | --- | --- |
| **CTI → ATT&CK** | `cti-technique-mapping` | a CTI procedure sentence | one [MITRE ATT&CK](https://attack.mitre.org/) technique ID (e.g. `T1059.001`), resolved to its official name |
| **GenAI attack vector** | `genai-attack-vector` | a GenAI security incident description | one attack-vector label (14-way closed set, e.g. `prompt-injection`) |
| **Text → JSON** | `text-to-json` | a request + a JSON schema | a populated JSON object conforming to the schema |

Unlike a server-side Gradio app, this runs **entirely in your browser** via ONNX
Runtime Web. The ONNX model is loaded with the standard
`AutoModelForCausalLM.from_pretrained` for `model_type: granite_switch`, via a
self-registering transformers.js architecture shim that threads the per-token
"switch" state through transformers.js's own generation loop. The CTI ID→name lookup
uses the official
[mitre-attack/attack-stix-data](https://github.com/mitre-attack/attack-stix-data)
Enterprise bundle (697 techniques), bundled into the page.

The demo loads the **int8** model variant, so the first load downloads ~440 MB of
weights (a progress bar shows status); subsequent runs and tab switches reuse the one
in-memory model. Built from the `granite-switch` repo's `web/example-demo` Vite app.
