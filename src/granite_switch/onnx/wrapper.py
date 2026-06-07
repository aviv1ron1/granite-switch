# SPDX-License-Identifier: Apache-2.0
"""Trace-friendly wrapper assembling the full Granite Switch forward for export.

``torch.onnx.export`` traces a single ``nn.Module.forward``. This module wraps a
:class:`~granite_switch.hf.modeling_granite_switch.GraniteSwitchForCausalLM`
whose switched-LoRA / switch layers have been swapped for the branchless
:mod:`granite_switch.onnx.export_modules` variants, and exposes a flat
``forward(input_ids)`` signature returning a plain ``logits`` tensor (no
``Cache`` objects, no kwargs) so the exporter sees a clean dataflow graph.

Phase 1 targets the prefill, no-cache signature. The decode signature with the
threaded ``past.switch.*`` cumulative state is added in Phase 3.
"""

import torch
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


class OnnxPrefillWrapper(nn.Module):
    """Flat ``forward(input_ids) -> logits`` over a reskinned GraniteSwitch model.

    No KV cache: this is the prefill graph used to validate that the full
    switch + LoRA + base + lm_head forward exports to a clean ONNX graph
    (GATE 1) and matches the HF backend numerically (GATE 2).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, use_cache=False, return_dict=True)
        return out.logits
