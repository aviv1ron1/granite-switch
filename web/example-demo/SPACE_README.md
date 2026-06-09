---
title: Granite Switch 350M CTI (native transformers.js)
emoji: 🪶
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: apache-2.0
header: mini
app_file: index.html
# Cross-origin isolation enables the multi-threaded onnxruntime-web WASM build.
# HF Static Spaces apply these headers to every response when set here.
cross_origin_embedder_policy: require-corp
cross_origin_opener_policy: same-origin
cross_origin_resource_policy: cross-origin
models:
  - barha/granite-switch-4.0-350m-cti-onnx
---

# Granite Switch 4.0 350M — CTI Technique Mapping

A **fully client-side** (in-browser) demo of
[`barha/granite-switch-4.0-350m-cti-onnx`](https://huggingface.co/barha/granite-switch-4.0-350m-cti-onnx).

Enter a piece of cyber threat intelligence (CTI) text describing adversary
behavior. The demo runs the **same base model twice on the same prompt** and shows
the results side by side:

- **Adapter OFF** — the plain `ibm-granite/granite-4.0-350m` base. It answers in
  prose and rarely produces a clean technique ID.
- **Adapter ON** — the embedded `cti-technique-mapping` LoRA is fired by a single
  control token, and the model emits one
  [MITRE ATT&CK](https://attack.mitre.org/) technique ID (e.g. `T1059.001`), which
  the page resolves to its official technique **name**.

Unlike the server-side [Gradio version](https://huggingface.co/spaces/barha/granite-switch-tiny),
this runs **entirely in your browser** via ONNX Runtime Web. The ONNX model is
loaded with the standard `AutoModelForCausalLM.from_pretrained` for
`model_type: granite_switch`, via a self-registering transformers.js architecture
shim that threads the per-token "switch" state through transformers.js's own
generation loop. The ID→name lookup uses the official
[mitre-attack/attack-stix-data](https://github.com/mitre-attack/attack-stix-data)
Enterprise bundle (697 techniques), bundled into the page.

First load downloads ~1.3 GB of fp32 weights from the model repo, so give it a
moment (a progress bar shows download status). Built from the `granite-switch`
repo's `web/example-demo` Vite app.
