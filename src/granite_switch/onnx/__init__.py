# SPDX-License-Identifier: Apache-2.0
"""ONNX export backend for Granite Switch (requires the ``[onnx]`` extra).

This backend exports the *entire* Granite Switch forward — adapter selection,
control-token rewrite, stacked switched-LoRA, the base Granite decoder, and the
LM head — into a single self-contained ONNX graph that can run via raw
``onnxruntime`` (Python) or ``onnxruntime-web`` (browser).

transformers.js has no custom-architecture registration API, so the
``granite_switch`` ``model_type`` cannot go through its ``pipeline()`` /
``AutoModel`` path. The browser runtime therefore drives the exported graph
directly with ``onnxruntime-web``, reusing ``@huggingface/transformers`` only
for tokenization and sampling.

The export modules (:mod:`export_modules`) reuse the *same* parameter tensors as
the HF backend (:mod:`granite_switch.hf`) and only swap in branchless,
trace-clean forward implementations. They never hand-assemble a config or
weights — models are built via the HF backend / composer, then re-skinned for
export.
"""
