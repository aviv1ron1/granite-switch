# SPDX-License-Identifier: Apache-2.0
"""Reskin a Granite Switch model into its branchless, export-clean form.

``torch.onnx.export`` cannot trace the switched-LoRA / switch layers'
data-dependent control flow. :func:`reskin_for_export` swaps those layers in place
for the branchless :mod:`granite_switch.onnx.export_modules` variants (keeping the
same parameters) and pins the eager attention backend, leaving the model ready for
:class:`granite_switch.onnx.decode.OnnxDecodeWrapper` to export the single
KV-cached graph (which serves both the batched first pass and incremental decode).
"""

import torch.nn as nn

from .export_modules import reskin_lora_modules_for_export, OnnxSingleSwitch


def reskin_for_export(model: nn.Module) -> nn.Module:
    """In-place swap of LoRA + switch modules to their branchless Onnx* variants.

    Returns the same model (mutated) for convenience. The model keeps all its
    parameters/buffers; only the ``forward`` implementations change.
    """
    reskin_lora_modules_for_export(model)
    # The switch lives at model.model.switch (GraniteSwitchModel.switch).
    inner = getattr(model, "model", model)
    if getattr(inner, "switch", None) is not None:
        inner.switch.__class__ = OnnxSingleSwitch

    # Force the eager attention backend for export. The SDPA backend lowers to
    # aten.scaled_dot_product_attention with enable_gqa=True; the ONNX converter
    # mishandles that path when q_num_heads == kv_num_heads (no real GQA),
    # producing wrong numerics (and asserting outright with a None mask). Eager
    # attention decomposes to plain matmul + softmax, which exports correctly.
    if hasattr(model, "config"):
        model.config._attn_implementation = "eager"
    if hasattr(inner, "config"):
        inner.config._attn_implementation = "eager"
    for module in model.modules():
        if getattr(module, "config", None) is not None:
            try:
                module.config._attn_implementation = "eager"
            except Exception:
                pass

    model.eval()
    return model
