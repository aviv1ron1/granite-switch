# SPDX-License-Identifier: Apache-2.0
"""Export a Granite Switch model to ONNX (prefill + decode graphs) + a CLI.

Produces a self-contained model directory runnable by ``onnxruntime`` /
``onnxruntime-web``:

    <out>/prefill.onnx      input_ids -> logits, present_switch_*, present_*  (full prompt)
    <out>/decode.onnx       one token + past_* -> logits, present_*          (incremental)
    <out>/gs_onnx.json      metadata the JS runtime needs (layers, kv heads, hd, etc.)

CLI::

    python -m granite_switch.onnx.export <model_path> --output ./gs-onnx
"""

import argparse
import glob
import json
import os

import torch

from .wrapper import OnnxPrefillWrapper, reskin_for_export
from .decode import OnnxDecodeWrapper

OPSET = 18  # dynamo exporter; lowest opset with the ops the switch/LoRA path needs.

# ONNX serializes to protobuf, which caps a single message at 2 GiB. A model
# whose initializers approach that limit cannot embed its weights inline and
# MUST keep them in an external ``.onnx.data`` sidecar. We embed only when the
# whole thing fits comfortably under the cap (browser-simple, single file).
_PROTOBUF_LIMIT = 2 * 1024**3
_EMBED_THRESHOLD = int(1.8 * 1024**3)  # headroom below the 2 GiB protobuf cap


def _sidecar_paths(onnx_path):
    """All possible sidecar files the dynamo exporter may have written."""
    base = os.path.splitext(onnx_path)[0]
    return [onnx_path + ".data", base + ".onnx.data", base + ".data"]


def _embed_external_data(onnx_path):
    """Fold weights back into a single ``.onnx`` file (no external sidecar).

    The dynamo exporter writes large initializers to a sidecar ``*.onnx.data``
    file. ``onnxruntime-web`` (WASM) cannot resolve that sidecar by default
    ("Module.MountedFiles is not available"), so for small models we fold the
    weights back into the single ``.onnx`` file for a one-file browser load.
    Large models (> ~2 GiB) cannot do this — see :func:`_keep_external_data`.
    """
    import onnx

    model = onnx.load(onnx_path, load_external_data=True)
    for tensor in model.graph.initializer:
        if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
            tensor.ClearField("external_data")
            tensor.data_location = onnx.TensorProto.DEFAULT
    onnx.save(model, onnx_path, save_as_external_data=False)
    for sidecar in _sidecar_paths(onnx_path):
        if os.path.exists(sidecar):
            try:
                os.remove(sidecar)
            except OSError:
                pass


def _keep_external_data(onnx_path):
    """Normalize a large model to ``<name>.onnx`` + ``<name>.onnx.data``.

    For models over the protobuf cap we keep weights external. The dynamo
    exporter's sidecar name varies (``prefill.onnx.data`` vs ``prefill.data``),
    so we re-save with a single, predictable sidecar named ``<basename>.data``
    and rewrite every external-data ``location`` to match. ``onnxruntime-web``
    loads such a sidecar when its filename is registered as an external-data
    file (see ``web/`` runtime), and ``onnxruntime`` (Python/Node) resolves it
    automatically from the adjacent path.

    Returns the sidecar filename (base name) that the model references.
    """
    import shutil
    import tempfile

    import onnx

    model = onnx.load(onnx_path, load_external_data=True)
    sidecar_name = os.path.basename(onnx_path) + ".data"  # e.g. model.onnx.data
    # Clear any stale external refs so save rewrites them uniformly.
    for tensor in model.graph.initializer:
        if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
            tensor.ClearField("external_data")
            tensor.data_location = onnx.TensorProto.DEFAULT

    # Save into a fresh temp dir, then move into place. Saving directly over the
    # exporter's sidecar (which already carries our canonical name) makes onnx
    # write external data alongside stale bytes, doubling the file on disk.
    # Writing to a clean dir guarantees the sidecar contains weights exactly once.
    tmp_dir = tempfile.mkdtemp(prefix="gs_onnx_ext_", dir=os.path.dirname(onnx_path) or ".")
    try:
        tmp_onnx = os.path.join(tmp_dir, os.path.basename(onnx_path))
        onnx.save(
            model, tmp_onnx,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=sidecar_name,
            size_threshold=1024,  # keep tiny scalars inline, weights external
            convert_attribute=False,
        )
        # Remove every prior sidecar variant, then move the fresh pair into place.
        for sidecar in _sidecar_paths(onnx_path):
            if os.path.exists(sidecar):
                try:
                    os.remove(sidecar)
                except OSError:
                    pass
        os.replace(tmp_onnx, onnx_path)
        os.replace(os.path.join(tmp_dir, sidecar_name),
                   os.path.join(os.path.dirname(onnx_path) or ".", sidecar_name))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return sidecar_name


def _finalize_onnx(onnx_path, *, embed):
    """Either embed weights inline (small) or keep a canonical sidecar (large)."""
    if embed:
        _embed_external_data(onnx_path)
        return None
    return _keep_external_data(onnx_path)


def export_prefill(model, output_path, *, example_input_ids, opset=OPSET, embed=True):
    """Export the no-cache prefill graph (``input_ids -> logits``).

    ``model`` is reskinned in place to the branchless Onnx* modules. ``seq_len``
    is a dynamic axis so any prompt length works at runtime. ``embed`` folds
    weights inline (small models); set ``embed=False`` to keep an external
    ``.onnx.data`` sidecar for large (> 2 GiB) models.
    """
    reskin_for_export(model)
    wrapper = OnnxPrefillWrapper(model).eval()
    seq = torch.export.Dim("seq_len")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (example_input_ids,), output_path,
            input_names=["input_ids"], output_names=["logits"],
            dynamic_shapes=({1: seq},),
            opset_version=opset, dynamo=True,
        )
    _finalize_onnx(output_path, embed=embed)
    return output_path


def export_decode(model, output_path, *, n_layers, kv_heads, head_dim, opset=OPSET, embed=True):
    """Export the single-token KV-cached decode graph.

    ``model`` must already be reskinned (call :func:`export_prefill` first, or
    :func:`reskin_for_export`). Past sequence-length is a dynamic axis.
    """
    wrapper = OnnxDecodeWrapper(model).eval()
    bsz = 1
    ex = [
        torch.tensor([[1]]),
        torch.zeros(bsz, 1),
        torch.zeros(bsz, 1),
    ]
    for _ in range(n_layers):
        ex += [torch.zeros(bsz, kv_heads, 1, head_dim),
               torch.zeros(bsz, kv_heads, 1, head_dim)]

    in_names = ["input_ids", "past_switch_key0", "past_switch_val0"]
    out_names = ["logits", "present_switch_key0", "present_switch_val0"]
    for li in range(n_layers):
        in_names += [f"past_key.{li}", f"past_value.{li}"]
    for li in range(n_layers):
        out_names.append(f"present_key.{li}")
    for li in range(n_layers):
        out_names.append(f"present_value.{li}")

    past_len = torch.export.Dim("past_len")
    past_kv_shapes = tuple({2: past_len} for _ in range(2 * n_layers))
    dynamic_shapes = ({}, {1: past_len}, {1: past_len}, past_kv_shapes)

    with torch.no_grad():
        torch.onnx.export(
            wrapper, tuple(ex), output_path,
            input_names=in_names, output_names=out_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=opset, dynamo=True,
        )
    _finalize_onnx(output_path, embed=embed)
    return output_path


def _param_bytes(model):
    """Total byte size of model parameters (the bulk of the ONNX initializers)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def export_model_dir(model, out_dir, *, opset=OPSET, embed=None):
    """Export prefill + decode graphs and metadata into ``out_dir``.

    ``embed`` controls whether weights are folded inline (one ``.onnx`` file) or
    kept in an external ``.onnx.data`` sidecar. ``None`` (default) auto-selects:
    embed when the model fits under the ~2 GiB protobuf cap, otherwise keep the
    sidecar (required for large models like the 4 B granite-switch-4.1-3b).
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg = model.config
    gsm = model.model
    n_layers = len(gsm.layers)
    attn0 = gsm.layers[0].self_attn
    kv_heads = attn0.num_key_value_heads
    head_dim = attn0.head_dim

    if embed is None:
        embed = _param_bytes(model) < _EMBED_THRESHOLD

    example = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    export_prefill(model, os.path.join(out_dir, "prefill.onnx"),
                   example_input_ids=example, opset=opset, embed=embed)
    # model is now reskinned; decode reuses it.
    export_decode(
        model, os.path.join(out_dir, "decode.onnx"),
        n_layers=n_layers, kv_heads=kv_heads, head_dim=head_dim,
        opset=opset, embed=embed)

    meta = {
        "n_layers": n_layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": cfg.vocab_size,
        "num_adapters": cfg.num_adapters,
        "adapter_token_ids": list(getattr(cfg, "adapter_token_ids", []) or []),
        "adapter_names": list(getattr(cfg, "adapter_names", []) or []),
        "logits_scaling": getattr(cfg, "logits_scaling", 1.0),
        "external_data": not embed,  # True => weights live in a .onnx.data sidecar
    }
    with open(os.path.join(out_dir, "gs_onnx.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Also emit a transformers.js-loadable layout under <out>/tfjs/ so the model
    # can be loaded ON transformers.js via AutoModelForCausalLM.from_pretrained.
    #
    # transformers.js has no custom-architecture API and rejects
    # `model_type: granite_switch` ("Unsupported model type"). So the tfjs config
    # declares a SUPPORTED DecoderOnly model_type (gpt2): transformers.js then
    # loads config.json + creates/owns the ONNX session through its own backend.
    # The decode graph is placed at onnx/model.onnx (transformers.js's default
    # decoder session file). Granite Switch's own runtime metadata stays in
    # gs_onnx.json (copied alongside) for the switch-state-threading decode loop.
    import shutil
    tfjs_dir = os.path.join(out_dir, "tfjs")
    os.makedirs(os.path.join(tfjs_dir, "onnx"), exist_ok=True)
    hidden = kv_heads * head_dim
    tfjs_config = {
        "model_type": "gpt2",  # a supported DecoderOnly type so tfjs will load it
        "architectures": ["GPT2LMHeadModel"],
        "vocab_size": cfg.vocab_size,
        "n_layer": n_layers,
        "n_head": kv_heads,
        "n_embd": hidden,
        "num_attention_heads": kv_heads,
        "num_hidden_layers": n_layers,
        "hidden_size": hidden,
    }
    with open(os.path.join(tfjs_dir, "config.json"), "w") as f:
        json.dump(tfjs_config, f, indent=2)

    tfjs_model = os.path.join(tfjs_dir, "onnx", "model.onnx")
    decode_path = os.path.join(out_dir, "decode.onnx")
    if embed:
        # Single self-contained file; a plain copy is correct.
        shutil.copy(decode_path, tfjs_model)
    else:
        # The decode graph references "decode.onnx.data"; the tfjs copy lives at
        # onnx/model.onnx and must reference "model.onnx.data". Re-save under the
        # new name so transformers.js / onnxruntime-web resolves the sidecar.
        import onnx
        m = onnx.load(decode_path, load_external_data=True)
        for tensor in m.graph.initializer:
            if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
                tensor.ClearField("external_data")
                tensor.data_location = onnx.TensorProto.DEFAULT
        onnx.save(
            m, tfjs_model,
            save_as_external_data=True, all_tensors_to_one_file=True,
            location="model.onnx.data", size_threshold=1024, convert_attribute=False,
        )
    shutil.copy(os.path.join(out_dir, "gs_onnx.json"),
                os.path.join(tfjs_dir, "gs_onnx.json"))
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Path to a composed GraniteSwitch checkpoint")
    parser.add_argument("--output", default="./gs-onnx")
    parser.add_argument("--opset", type=int, default=OPSET)
    embed_grp = parser.add_mutually_exclusive_group()
    embed_grp.add_argument(
        "--embed", dest="embed", action="store_true", default=None,
        help="Force weights inline in a single .onnx (fails > 2 GiB)")
    embed_grp.add_argument(
        "--no-embed", dest="embed", action="store_false",
        help="Force an external .onnx.data sidecar (required for large models)")
    parser.add_argument(
        "--dtype", default="float32",
        help="torch dtype to load the checkpoint in (e.g. float32, float16)")
    parser.add_argument(
        "--quantize", choices=["int8", "q4"], action="append", default=None,
        help="Also emit a quantized variant for browser delivery "
             "(repeatable: --quantize int8 --quantize q4)")
    args = parser.parse_args()

    from granite_switch.hf import GraniteSwitchForCausalLM

    dtype = getattr(torch, args.dtype)
    model = GraniteSwitchForCausalLM.from_pretrained(args.model, torch_dtype=dtype).eval()
    out = export_model_dir(model, args.output, opset=args.opset, embed=args.embed)
    print(f"Exported prefill + decode graphs → {out}")

    for scheme in args.quantize or []:
        from .quantize import quantize_model_dir
        print(f"Quantizing ({scheme})...")
        res = quantize_model_dir(out, scheme=scheme)
        for k, v in res.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
