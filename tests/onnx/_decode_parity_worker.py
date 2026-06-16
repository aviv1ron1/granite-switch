# SPDX-License-Identifier: Apache-2.0
"""GATE 3: KV-cached step-by-step decode matches the full HF forward.

Builds the tiny model, computes full-sequence HF logits as reference, then runs
the exported single decode graph step-by-step (seeding past state from an empty
past, one token at a time) and checks the per-step logits match HF at each
absolute position — including positions *after* a control token, which exercises
the cumulative switch state threading. Uses the production
:func:`granite_switch.onnx.export.export_decode` (the same graph the browser
ships) rather than a bespoke re-export, so it covers the real artifact.
"""

import sys
import tempfile
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _onnx_parity_worker import build_tiny_model  # noqa: E402

from granite_switch.onnx.export import export_decode  # noqa: E402


def run_decode_parity(verbose=True):
    model, cfg = build_tiny_model()
    n_layers = len(model.model.layers)
    kvh = cfg.num_key_value_heads
    hd = model.model.layers[0].self_attn.head_dim

    # Full sequence with a control token at pos 2 → adapter fires for pos>=2.
    seq = [50, 51, 250, 52, 53, 54, 55]
    full_ids = torch.tensor([seq])

    # Reference: HF full-sequence logits (un-reskinned model).
    with torch.no_grad():
        ref = model(input_ids=full_ids, use_cache=False).logits.numpy()

    # Export the production single decode graph (reskins the model in place).
    td = tempfile.mkdtemp()
    onnx_path = os.path.join(td, "decode.onnx")
    export_decode(model, onnx_path, n_layers=n_layers, kv_heads=kvh, head_dim=hd)

    out_names = ["logits", "present_switch_key0", "present_switch_val0"]
    out_names += [f"present_key.{li}" for li in range(n_layers)]
    out_names += [f"present_value.{li}" for li in range(n_layers)]

    # Seed past state from an empty past, feeding tokens one at a time.
    bsz = 1
    sk0 = torch.zeros(bsz, 0)
    sv0 = torch.zeros(bsz, 0)
    pk = [torch.zeros(bsz, kvh, 0, hd) for _ in range(n_layers)]
    pv = [torch.zeros(bsz, kvh, 0, hd) for _ in range(n_layers)]

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    max_diff = 0.0
    per_pos = []
    for t, tok in enumerate(seq):
        feed = {
            "input_ids": np.array([[tok]], dtype=np.int64),
            "past_switch_key0": sk0.numpy().astype(np.float32),
            "past_switch_val0": sv0.numpy().astype(np.float32),
        }
        for li in range(n_layers):
            feed[f"past_key.{li}"] = pk[li].numpy().astype(np.float32)
            feed[f"past_value.{li}"] = pv[li].numpy().astype(np.float32)
        outs = sess.run(out_names, feed)
        out_map = dict(zip(out_names, outs))

        logit = out_map["logits"]  # [1,1,vocab]
        d = float(np.abs(logit[0, 0] - ref[0, t]).max())
        per_pos.append(round(d, 5))
        max_diff = max(max_diff, d)

        # advance state
        sk0 = torch.from_numpy(out_map["present_switch_key0"])
        sv0 = torch.from_numpy(out_map["present_switch_val0"])
        pk = [torch.from_numpy(out_map[f"present_key.{li}"]) for li in range(n_layers)]
        pv = [torch.from_numpy(out_map[f"present_value.{li}"]) for li in range(n_layers)]

    if verbose:
        print(f"  per-position max|diff| vs HF full forward: {per_pos}")
        print(f"  overall max|diff| = {max_diff:.3e}")
    ok = max_diff < 1e-3
    return ok, max_diff, per_pos


def main():
    print("ONNX decode parity (KV-cached step-by-step vs HF full forward):")
    ok, max_diff, _ = run_decode_parity()
    if ok:
        print(f"\nPASS: decode matches HF full forward (max |diff| = {max_diff:.3e})")
        return 0
    print(f"\nFAIL: decode diverges (max |diff| = {max_diff:.3e})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
