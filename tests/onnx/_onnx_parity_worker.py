# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker: ONNX-exported prefill logits vs HF backend logits.

Mirrors ``tests/composer/_skinning_equivalence_worker.py`` in spirit: build a
tiny GraniteSwitch model on CPU with random weights and active adapters, compute
reference logits with the HF backend, export the (reskinned, branchless) forward
to ONNX, run it under ``onnxruntime``, and compare.

Parity is ``allclose`` (not bit-exact): the branchless gather+matmul reorders the
LoRA reduction vs the HF per-adapter loop (consistent with CLAUDE.md §9, which
only claims bit-exact for the fused/vLLM path).

Built directly from ``GraniteSwitchConfig`` (not via the composer): this is a
unit-level numeric check of a single forward over the model internals, the same
pattern ``tests/hf/test_model_forward.py`` uses. The composer path is exercised
by the Phase-5 production validation.

Runs in its own process so the ONNX/ORT state is torn down cleanly on exit.

Usage::

    python tests/onnx/_onnx_parity_worker.py [--seq-len N]
"""

import argparse
import sys
import tempfile
import os

import numpy as np
import torch

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM
from granite_switch.onnx.export import export_prefill


def build_tiny_model(seed: int = 0):
    """Tiny CPU GraniteSwitch with 2 adapters and visible (non-zero) LoRA deltas.

    Mirrors the ``tiny_single_config`` fixture + ``_set_nonzero_lora_B`` helper in
    ``tests/hf/test_model_forward.py``.
    """
    cfg = GraniteSwitchConfig(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        num_adapters=2, adapter_token_ids=[250, 251],
        adapter_substitute_token_ids=[1, 1],
        adapter_names=["adapter_1", "adapter_2"],
        max_lora_rank=4, adapter_ranks=[4, 4], switch_head_dim=16,
    )
    torch.manual_seed(seed)
    model = GraniteSwitchForCausalLM(cfg).eval()
    model.model.adapter_token_ids.data = torch.tensor([250, 251], dtype=torch.long)
    with torch.no_grad():
        for layer in model.model.layers:
            for b in layer.self_attn.qkv_proj.lora_B_slices:
                b.data = torch.randn_like(b) * 0.1
            layer.self_attn.o_proj.lora_B.data = (
                torch.randn_like(layer.self_attn.o_proj.lora_B) * 0.1
            )
            for b in layer.shared_mlp.input_linear.lora_B_slices:
                b.data = torch.randn_like(b) * 0.1
            layer.shared_mlp.output_linear.lora_B.data = (
                torch.randn_like(layer.shared_mlp.output_linear.lora_B) * 0.1
            )
    return model, cfg


def run_parity(seq_len: int = 6, verbose: bool = True):
    model, cfg = build_tiny_model()

    # Sequence with a control token at position 2 so adapter 1 fires cumulatively.
    torch.manual_seed(42)
    ids = torch.randint(2, 200, (1, seq_len))
    ids[0, 2] = 250  # adapter_1 control token
    input_ids = ids

    # ── Reference: HF backend ────────────────────────────────────────
    with torch.no_grad():
        ref_logits = model(input_ids=input_ids, use_cache=False).logits.cpu().numpy()

    # ── Export (reskins model in place) + run under onnxruntime ──────
    with tempfile.TemporaryDirectory() as td:
        onnx_path = os.path.join(td, "tiny_prefill.onnx")
        export_prefill(model, onnx_path, example_input_ids=input_ids)

        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(
            ["logits"], {"input_ids": input_ids.numpy().astype(np.int64)}
        )
        onnx_logits = ort_out[0]

    diff = np.abs(onnx_logits - ref_logits)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    if verbose:
        print(f"  seq_len={seq_len}")
        print(f"  ref logits shape: {ref_logits.shape}, onnx: {onnx_logits.shape}")
        print(f"  max |diff| = {max_diff:.3e}")
        print(f"  mean |diff| = {mean_diff:.3e}")

    ok = np.allclose(onnx_logits, ref_logits, rtol=1e-3, atol=1e-4)
    return ok, max_diff, mean_diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=6)
    args = parser.parse_args()

    print("ONNX prefill parity (HF backend vs onnxruntime):")
    ok, max_diff, _ = run_parity(seq_len=args.seq_len)
    if ok:
        print(f"\nPASS: ONNX logits match HF (max |diff| = {max_diff:.3e})")
        return 0
    print(f"\nFAIL: ONNX logits differ (max |diff| = {max_diff:.3e})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
