# SPDX-License-Identifier: Apache-2.0
"""Quantize an exported Granite Switch ONNX graph for browser delivery.

The fp32 export of a 4 B model is ~16 GB — too large to ship to a browser. Two
weight-only schemes shrink it without touching the (branchless) switch/LoRA
dataflow, which stays fp32:

* ``int8`` — :func:`onnxruntime.quantization.quantize_dynamic`. ~4x smaller,
  widely supported by ``onnxruntime-web`` on WASM. Robust default.
* ``q4`` — :class:`onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBits
  Quantizer` (4-bit block quantization of ``MatMul`` weights). ~8x smaller; this
  is the scheme ``transformers.js`` ships as ``model_q4.onnx``.

Both are *weight-only* and *post-training* (no calibration data needed). They
quantize the large linear weights; activations stay float, so the switch's
adapter-selection arithmetic (CumSum/Softmax/round) is unaffected.

Output keeps the external-data layout (``<name>.onnx`` + ``<name>.onnx.data``)
so it composes with the large-model export path.
"""

import os

SCHEMES = ("int8", "q4")


def _ext_save(model, onnx_path):
    """Save with a canonical single ``<basename>.onnx.data`` sidecar."""
    import onnx

    sidecar = os.path.basename(onnx_path) + ".data"
    onnx.save(
        model, onnx_path,
        save_as_external_data=True, all_tensors_to_one_file=True,
        location=sidecar, size_threshold=1024, convert_attribute=False,
    )
    return sidecar


def _src_total_bytes(onnx_path):
    """On-disk size of an ONNX model including any external-data sidecar."""
    base = os.path.splitext(onnx_path)[0]
    total = os.path.getsize(onnx_path) if os.path.exists(onnx_path) else 0
    for sidecar in (onnx_path + ".data", base + ".onnx.data", base + ".data"):
        if os.path.exists(sidecar):
            total += os.path.getsize(sidecar)
            break
    return total


def _normalize_sidecar(onnx_path):
    """Re-save so external data lives in a single canonical ``<name>.onnx.data``.

    ``quantize_dynamic(use_external_data_format=True)`` writes its own sidecar
    name; load it and re-emit through :func:`_ext_save` (writing to a temp dir
    then moving in, to avoid doubling the file as the export path does).
    """
    import shutil
    import tempfile

    import onnx

    model = onnx.load(onnx_path, load_external_data=True)
    for tensor in model.graph.initializer:
        if tensor.HasField("data_location") and tensor.data_location == onnx.TensorProto.EXTERNAL:
            tensor.ClearField("external_data")
            tensor.data_location = onnx.TensorProto.DEFAULT
    sidecar = os.path.basename(onnx_path) + ".data"
    tmp_dir = tempfile.mkdtemp(prefix="gs_q_ext_", dir=os.path.dirname(onnx_path) or ".")
    try:
        tmp_onnx = os.path.join(tmp_dir, os.path.basename(onnx_path))
        _ext_save(model, tmp_onnx)
        base = os.path.splitext(onnx_path)[0]
        for old in (onnx_path + ".data", base + ".onnx.data", base + ".data"):
            if os.path.exists(old):
                try:
                    os.remove(old)
                except OSError:
                    pass
        os.replace(tmp_onnx, onnx_path)
        os.replace(os.path.join(tmp_dir, sidecar),
                   os.path.join(os.path.dirname(onnx_path) or ".", sidecar))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return sidecar


def quantize_int8(src_onnx, dst_onnx):
    """Dynamic int8 weight quantization (onnxruntime-web friendly).

    For large models the int8 result can still exceed ONNX's 2 GiB protobuf cap
    (a 4 B model is ~4-5 GB int8), so the writer must use external data. We
    detect that from the *source* fp32 size — anything that needed an external
    fp32 sidecar will need one int8 too — and normalize the sidecar afterward.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    src_bytes = _src_total_bytes(src_onnx)
    # int8 is ~1/4 of fp32; if fp32 was over ~1.6 GiB the int8 output may still
    # approach/exceed 2 GiB, and even mid-size models are safer external.
    use_external = src_bytes > 1.5 * 1024**3

    # per_channel=True keeps accuracy high for the big projection weights.
    quantize_dynamic(
        src_onnx, dst_onnx,
        weight_type=QuantType.QInt8,
        per_channel=True,
        use_external_data_format=use_external,
    )
    if use_external:
        _normalize_sidecar(dst_onnx)
    return dst_onnx


def quantize_q4(src_onnx, dst_onnx, *, block_size=32):
    """4-bit block quantization of MatMul weights (transformers.js q4 scheme)."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    model = onnx.load(src_onnx, load_external_data=True)
    quantizer = MatMulNBitsQuantizer(model, block_size=block_size, is_symmetric=True)
    quantizer.process()
    _ext_save(quantizer.model.model, dst_onnx)
    return dst_onnx


def quantize_onnx(src_onnx, dst_onnx, *, scheme="int8", block_size=32):
    """Quantize ``src_onnx`` into ``dst_onnx`` using ``scheme`` (int8 | q4)."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")
    if scheme == "int8":
        return quantize_int8(src_onnx, dst_onnx)
    return quantize_q4(src_onnx, dst_onnx, block_size=block_size)


def quantize_model_dir(out_dir, *, scheme="int8", block_size=32):
    """Quantize the prefill + decode graphs already exported into ``out_dir``.

    Writes ``prefill_<scheme>.onnx`` and ``decode_<scheme>.onnx`` (plus their
    sidecars) alongside the fp32 graphs, and mirrors the quantized decode graph
    into the ``tfjs/`` layout as ``onnx/model_<scheme>.onnx`` so transformers.js
    can pick it via ``dtype: "<scheme>"``.
    """
    import shutil

    results = {}
    for name in ("prefill", "decode"):
        src = os.path.join(out_dir, f"{name}.onnx")
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_dir, f"{name}_{scheme}.onnx")
        quantize_onnx(src, dst, scheme=scheme, block_size=block_size)
        results[name] = dst

    # Mirror the quantized decode graph into tfjs/onnx/model_<scheme>.onnx.
    tfjs_onnx = os.path.join(out_dir, "tfjs", "onnx")
    dec = results.get("decode")
    if dec and os.path.isdir(tfjs_onnx):
        import onnx
        tfjs_q = os.path.join(tfjs_onnx, f"model_{scheme}.onnx")
        m = onnx.load(dec, load_external_data=True)
        onnx.save(
            m, tfjs_q,
            save_as_external_data=True, all_tensors_to_one_file=True,
            location=f"model_{scheme}.onnx.data", size_threshold=1024,
            convert_attribute=False,
        )
        results["tfjs_decode"] = tfjs_q

    return results
