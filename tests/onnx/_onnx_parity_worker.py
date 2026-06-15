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
from granite_switch.onnx.export import export_prefill, export_decode


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


def run_prefill_decode_handoff(seq_len: int = 6, verbose: bool = True):
    """Verify the batched prefill seeds incremental decode correctly.

    This is the browser runtime's path: run the WHOLE prompt through the prefill
    graph once (emitting present_* state), then run ONE decode step on the next
    token seeded by that state. The next-token logits MUST match a pure per-token
    decode replay of the same prompt+token (the OLD browser behavior). This asserts
    that (a) the prefill graph's present_* names/shapes interlock with the decode
    graph's past_* inputs, and (b) the batched prefill is numerically faithful to
    the serial replay it replaces. Run on CPU under onnxruntime.
    """
    model, cfg = build_tiny_model()
    n_layers = len(model.model.layers)
    kvh = model.model.layers[0].self_attn.num_key_value_heads
    hd = model.model.layers[0].self_attn.head_dim

    torch.manual_seed(7)
    ids = torch.randint(2, 200, (1, seq_len))
    ids[0, 2] = 250  # adapter_1 control token (cumulative selection)
    next_tok = int(torch.randint(2, 200, (1,)).item())

    with tempfile.TemporaryDirectory() as td:
        prefill_path = os.path.join(td, "prefill.onnx")
        decode_path = os.path.join(td, "decode.onnx")
        # export_prefill reskins the model in place; decode reuses the reskinned model.
        export_prefill(model, prefill_path, example_input_ids=ids)
        export_decode(model, decode_path, n_layers=n_layers, kv_heads=kvh, head_dim=hd)

        import onnxruntime as ort

        pf = ort.InferenceSession(prefill_path, providers=["CPUExecutionProvider"])
        dc = ort.InferenceSession(decode_path, providers=["CPUExecutionProvider"])

        def decode_step(tok, state):
            """One decode step. `state` is the present_* dict from the prior step."""
            feed = {
                "input_ids": np.array([[tok]], dtype=np.int64),
                "past_switch_key0": state["switch_key0"],
                "past_switch_val0": state["switch_val0"],
            }
            for li in range(n_layers):
                feed[f"past_key.{li}"] = state["k"][li]
                feed[f"past_value.{li}"] = state["v"][li]
            out_names = ["logits", "present_switch_key0", "present_switch_val0"]
            out_names += [f"present_key.{li}" for li in range(n_layers)]
            out_names += [f"present_value.{li}" for li in range(n_layers)]
            outs = dict(zip(out_names, dc.run(out_names, feed)))
            new_state = {
                "switch_key0": outs["present_switch_key0"],
                "switch_val0": outs["present_switch_val0"],
                "k": [outs[f"present_key.{li}"] for li in range(n_layers)],
                "v": [outs[f"present_value.{li}"] for li in range(n_layers)],
            }
            return outs["logits"], new_state

        # ── Path A: prefill once, then ONE decode step on next_tok ──
        pf_out_names = ["logits", "present_switch_key0", "present_switch_val0"]
        pf_out_names += [f"present_key.{li}" for li in range(n_layers)]
        pf_out_names += [f"present_value.{li}" for li in range(n_layers)]
        pf_outs = dict(zip(pf_out_names, pf.run(pf_out_names, {"input_ids": ids.numpy().astype(np.int64)})))
        state_A = {
            "switch_key0": pf_outs["present_switch_key0"],
            "switch_val0": pf_outs["present_switch_val0"],
            "k": [pf_outs[f"present_key.{li}"] for li in range(n_layers)],
            "v": [pf_outs[f"present_value.{li}"] for li in range(n_layers)],
        }
        logits_A, _ = decode_step(next_tok, state_A)

        # ── Path B: pure per-token decode replay of the prompt, then next_tok ──
        zero_switch = np.zeros((1, 0), dtype=np.float32)
        state_B = {
            "switch_key0": zero_switch, "switch_val0": zero_switch,
            "k": [np.zeros((1, kvh, 0, hd), dtype=np.float32) for _ in range(n_layers)],
            "v": [np.zeros((1, kvh, 0, hd), dtype=np.float32) for _ in range(n_layers)],
        }
        for t in ids[0].tolist():
            _, state_B = decode_step(int(t), state_B)
        logits_B, _ = decode_step(next_tok, state_B)

    diff = np.abs(logits_A - logits_B)
    max_diff = float(diff.max())
    if verbose:
        print(f"  seq_len={seq_len}, next_tok={next_tok}")
        print(f"  prefill-seeded decode logits: {logits_A.shape}; replay-seeded: {logits_B.shape}")
        print(f"  max |diff| (prefill-handoff vs replay) = {max_diff:.3e}")
    ok = np.allclose(logits_A, logits_B, rtol=1e-3, atol=1e-4)
    return ok, max_diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=6)
    args = parser.parse_args()

    print("ONNX prefill parity (HF backend vs onnxruntime):")
    ok1, max_diff1, _ = run_parity(seq_len=args.seq_len)
    print(f"  prefill-vs-HF: {'PASS' if ok1 else 'FAIL'} (max |diff| = {max_diff1:.3e})")

    print("\nONNX prefill->decode handoff (batched prefill vs per-token replay):")
    ok2, max_diff2 = run_prefill_decode_handoff(seq_len=args.seq_len)
    print(f"  handoff-vs-replay: {'PASS' if ok2 else 'FAIL'} (max |diff| = {max_diff2:.3e})")

    if ok1 and ok2:
        print("\nPASS: batched prefill matches HF AND seeds decode identically to replay.")
        return 0
    print("\nFAIL: see per-check results above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
