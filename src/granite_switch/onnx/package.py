# SPDX-License-Identifier: Apache-2.0
"""Assemble an HF-Hub-uploadable, transformers.js-loadable model repo.

``export_model_dir`` already emits a ``tfjs/`` layout (``config.json`` +
``onnx/model.onnx`` + ``gs_onnx.json``) that ``@huggingface/transformers`` can
load. This packages that into a clean repo directory ready for
``huggingface-cli upload``:

    <repo>/
      config.json                 # gpt2-typed DecoderOnly config (so tfjs loads it)
      gs_onnx.json                # Granite Switch runtime metadata (switch threading)
      tokenizer.json              # copied from the source checkpoint (if available)
      tokenizer_config.json
      special_tokens_map.json
      onnx/
        model.onnx[.data]         # fp32 (dtype: "fp32")
        model_int8.onnx[.data]    # dtype: "int8"   (optional)
        model_q4.onnx[.data]      # dtype: "q4"     (optional)
      README.md                   # model card + the switch-state caveat

transformers.js selects the variant by ``dtype`` (``""`` / ``_int8`` / ``_q4``
suffix). The generic generation loop cannot thread the switch's cumulative
adapter state, so the repo ships with a pointer to the ``web/`` decode loop that
does — the README spells this out.
"""

import json
import os
import shutil

_TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "chat_template.jinja",
)


def _copy_onnx_variant(src_onnx, dst_onnx):
    """Copy an ONNX graph + its single external sidecar (if any).

    When the source has an external sidecar, load it from the *source* dir
    (where the sidecar resolves) and re-save at the destination with a sidecar
    named after the destination file. A plain ``shutil.copy`` of the ``.onnx``
    alone would leave it referencing a sidecar that isn't next to it.
    """
    if os.path.exists(src_onnx + ".data"):
        import onnx

        m = onnx.load(src_onnx, load_external_data=True)
        for t in m.graph.initializer:
            if t.HasField("data_location") and t.data_location == onnx.TensorProto.EXTERNAL:
                t.ClearField("external_data")
                t.data_location = onnx.TensorProto.DEFAULT
        onnx.save(
            m, dst_onnx, save_as_external_data=True, all_tensors_to_one_file=True,
            location=os.path.basename(dst_onnx) + ".data",
            size_threshold=1024, convert_attribute=False,
        )
    else:
        shutil.copy(src_onnx, dst_onnx)


def _native_config(meta):
    """A ``granite_switch``-typed config.json for the NATIVE transformers.js path.

    The Granite Switch web shim (``web/src/granite-switch-register.js``) registers
    a real ``GraniteSwitchForCausalLM`` architecture, so the repo can declare its
    true ``model_type`` instead of masquerading as ``gpt2``. ``architectures[0]``
    matches the registered class name so transformers.js's cross-arch detector
    (which only fires for ``*ForConditionalGeneration``) is skipped.
    """
    kv_heads = meta.get("kv_heads")
    head_dim = meta.get("head_dim")
    return {
        "model_type": "granite_switch",
        "architectures": ["GraniteSwitchForCausalLM"],
        "num_hidden_layers": meta.get("n_layers"),
        "num_attention_heads": kv_heads,
        "num_key_value_heads": kv_heads,
        "hidden_size": (kv_heads * head_dim) if (kv_heads and head_dim) else None,
        "vocab_size": meta.get("vocab_size"),
    }


def package_for_hub(export_dir, repo_dir, *, tokenizer_src=None, model_name="granite-switch",
                    repo_id=None, native=False):
    """Build an upload-ready repo from an ``export_model_dir`` output.

    ``export_dir``      directory produced by ``export_model_dir`` (has ``tfjs/``).
    ``repo_dir``        destination repo directory (created/overwritten).
    ``tokenizer_src``   optional path/HF-id to copy tokenizer files from.
    ``repo_id``         optional ``user/name`` used in the README usage snippet.
    ``native``          when True, write a ``granite_switch``-typed ``config.json``
                        (loads via the web shim's native ``from_pretrained``)
                        instead of the legacy ``gpt2``-typed workaround config.
    """
    tfjs = os.path.join(export_dir, "tfjs")
    if not os.path.isdir(tfjs):
        raise FileNotFoundError(f"no tfjs/ layout under {export_dir}; run export_model_dir first")

    os.makedirs(os.path.join(repo_dir, "onnx"), exist_ok=True)

    # switch metadata (always copied)
    shutil.copy(os.path.join(tfjs, "gs_onnx.json"), os.path.join(repo_dir, "gs_onnx.json"))
    meta = json.load(open(os.path.join(tfjs, "gs_onnx.json")))

    # config.json: native granite_switch-typed, or the legacy gpt2-typed workaround.
    if native:
        cfg = {k: v for k, v in _native_config(meta).items() if v is not None}
        with open(os.path.join(repo_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    else:
        shutil.copy(os.path.join(tfjs, "config.json"), os.path.join(repo_dir, "config.json"))

    # onnx variants (fp32 + any quantized)
    variants = []
    src_onnx_dir = os.path.join(tfjs, "onnx")
    for fname in sorted(os.listdir(src_onnx_dir)):
        if fname.endswith(".onnx"):
            _copy_onnx_variant(os.path.join(src_onnx_dir, fname),
                               os.path.join(repo_dir, "onnx", fname))
            variants.append(fname)

    # tokenizer files
    copied_tok = []
    if tokenizer_src:
        tok_dir = _resolve_tokenizer_dir(tokenizer_src)
        if tok_dir:
            for tf in _TOKENIZER_FILES:
                p = os.path.join(tok_dir, tf)
                if os.path.exists(p):
                    shutil.copy(p, os.path.join(repo_dir, tf))
                    copied_tok.append(tf)

    _write_model_card(repo_dir, model_name=model_name, meta=meta,
                      variants=variants, tokenizer=copied_tok, repo_id=repo_id, native=native)
    return repo_dir


def _resolve_tokenizer_dir(src):
    """Return a local dir holding tokenizer files for ``src`` (path or HF id)."""
    if os.path.isdir(src):
        return src
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(src, allow_patterns=list(_TOKENIZER_FILES))
    except Exception:
        return None


def _write_model_card(repo_dir, *, model_name, meta, variants, tokenizer, repo_id, native=False):
    rid = repo_id or "your-org/granite-switch-onnx-web"
    has_int8 = any("_int8" in v for v in variants)
    has_q4 = any("_q4" in v for v in variants)
    dtype_lines = []
    if any(v == "model.onnx" for v in variants):
        dtype_lines.append('- `fp32` — full precision (`onnx/model.onnx`)')
    if has_int8:
        dtype_lines.append('- `int8` — `onnx/model_int8.onnx`')
    if has_q4:
        dtype_lines.append('- `q4` — `onnx/model_q4.onnx` (smallest)')
    dtypes = "\n".join(dtype_lines) or "- `fp32`"

    if native:
        config_section = f"""## Loading (native transformers.js architecture)

This repo declares its true `model_type: granite_switch`. The Granite Switch web
runtime ships a **self-registering shim** that teaches transformers.js the
`GraniteSwitchForCausalLM` architecture, so it loads via the standard
`AutoModelForCausalLM.from_pretrained`. Granite Switch selects adapters via a
**causal, cumulative** switch attention; the shim's custom forward threads that
state (`past_switch_key0` / `past_switch_val0`) across decode steps — the generic
loop alone is not sufficient.

```js
// import the shim once (registers granite_switch), then use the re-exported API
import {{ AutoModelForCausalLM, loadGraniteSwitch }} from "granite-switch-web/src/granite-switch-register.js";

// loadGraniteSwitch wires the external-data sidecar (model.onnx.data) for you:
const gs = await loadGraniteSwitch("{rid}", {{ dtype: "fp32" }});
const out = await gs.generate({{ inputs, max_new_tokens: 16, do_sample: false }});
```"""
    else:
        config_section = f"""## Why a custom loop is required

`transformers.js` has no built-in custom-architecture API, so this repo presents a
**supported DecoderOnly `model_type` (`gpt2`)** in `config.json` purely so
`AutoModelForCausalLM.from_pretrained` will load it and create the ONNX session.
Granite Switch selects adapters via a **causal, cumulative** switch attention,
so the generic generation loop is *not* sufficient — the switch's state
(`past_switch_key0` / `past_switch_val0`) must be threaded across decode steps.

Use the Granite Switch web runtime, which drives the transformers.js-owned
session with that state threading:

```js
import {{ env, AutoModelForCausalLM, Tensor }} from "@huggingface/transformers";
import {{ GraniteSwitchTfjs }} from "granite-switch-web";

const meta = await (await fetch("https://huggingface.co/{rid}/resolve/main/gs_onnx.json")).json();
env.allowRemoteModels = true;
const gs = await GraniteSwitchTfjs.load({{
  localModelPath: "https://huggingface.co/{rid}/resolve/main/",
  modelName: ".",            // config.json + onnx/ live at the repo root
  meta,
}});
const tokens = await gs.generate([10, 20, 30, /* control token */ , 40], 16);
```"""

    card = f"""---
library_name: transformers.js
tags:
- granite
- granite-switch
- onnx
- transformers.js
pipeline_tag: text-generation
---

# {model_name} (ONNX / transformers.js)

ONNX export of a **Granite Switch** model — base Granite with per-token LoRA
adapter switching driven by control tokens — packaged to load **on
transformers.js**.

- adapters: **{meta.get('num_adapters', '?')}** ({', '.join(meta.get('adapter_names', [])[:6])}{'…' if len(meta.get('adapter_names', []))>6 else ''})
- layers: {meta.get('n_layers')}, kv-heads: {meta.get('kv_heads')}, head-dim: {meta.get('head_dim')}
- vocab: {meta.get('vocab_size')}

## Available dtypes

{dtypes}

{config_section}

`gs_onnx.json` carries the runtime metadata (layer/head counts, adapter token
ids) the decode loop needs.

## Provenance

Exported with `granite_switch.onnx.export` + `granite_switch.onnx.package`.
See the project docs (`docs/ONNX_BROWSER_PORT.md`) for the export pipeline and
parity gates (HF-vs-ONNX `max|diff|` ~1e-4, argmax 100% on the source model).
"""
    with open(os.path.join(repo_dir, "README.md"), "w") as f:
        f.write(card)


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("export_dir", help="directory produced by export_model_dir (has tfjs/)")
    p.add_argument("repo_dir", help="destination upload-ready repo directory")
    p.add_argument("--tokenizer-src", default=None,
                   help="path or HF id to copy tokenizer files from")
    p.add_argument("--model-name", default="granite-switch")
    p.add_argument("--repo-id", default=None, help="user/name for README usage snippet")
    p.add_argument("--native", action="store_true",
                   help="write a granite_switch-typed config.json (native transformers.js "
                        "load via the web shim) instead of the gpt2-typed workaround")
    args = p.parse_args()

    out = package_for_hub(args.export_dir, args.repo_dir,
                          tokenizer_src=args.tokenizer_src,
                          model_name=args.model_name, repo_id=args.repo_id,
                          native=args.native)
    print(f"Packaged upload-ready repo → {out}")
    print("Upload with:  huggingface-cli upload <repo-id> "
          f"{out} . --repo-type model")


if __name__ == "__main__":
    main()
