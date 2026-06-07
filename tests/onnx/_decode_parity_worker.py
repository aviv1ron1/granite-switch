# SPDX-License-Identifier: Apache-2.0
"""GATE 3: KV-cached decode (with threaded switch state) matches full prefill.

Builds the tiny model, computes full-sequence HF logits as reference, then runs
the exported single-token decode graph step-by-step (seeding past state from a
short prefix) and checks the per-step logits match HF at each absolute position
— including positions *after* a control token, which exercises the cumulative
switch state threading.
"""

import sys
import tempfile
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _onnx_parity_worker import build_tiny_model  # noqa: E402

from granite_switch.onnx.wrapper import reskin_for_export  # noqa: E402
from granite_switch.onnx.decode import OnnxDecodeWrapper  # noqa: E402


def _export_decode(model, example_inputs):
    wrapper = OnnxDecodeWrapper(model).eval()
    n = wrapper.num_layers
    in_names = ["input_ids", "past_switch_key0", "past_switch_val0"]
    for li in range(n):
        in_names += [f"past_key.{li}", f"past_value.{li}"]
    out_names = ["logits", "present_switch_key0", "present_switch_val0"]
    for li in range(n):
        out_names.append(f"present_key.{li}")
    for li in range(n):
        out_names.append(f"present_value.{li}")

    # Past sequence-length axis is dynamic (the cache grows each step). The
    # dynamo exporter wants the modern `dynamic_shapes` API (positional, matching
    # forward's signature: input_ids, past_switch_key0, past_switch_val0, *past_kv).
    past_len = torch.export.Dim("past_len")
    # Signature is (input_ids, past_switch_key0, past_switch_val0, *past_kv), so
    # dynamic_shapes has 4 entries; the last is a tuple covering the var-args.
    past_kv_shapes = tuple({2: past_len} for _ in range(2 * n))
    dynamic_shapes = (
        {},                      # input_ids [b,1]
        {1: past_len},           # past_switch_key0 [b, past_len]
        {1: past_len},           # past_switch_val0 [b, past_len]
        past_kv_shapes,          # *past_kv: each [b, kvh, past_len, hd]
    )

    td = tempfile.mkdtemp()
    p = os.path.join(td, "decode.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, tuple(example_inputs), p,
            input_names=in_names, output_names=out_names,
            dynamic_shapes=tuple(dynamic_shapes),
            opset_version=18, dynamo=True,
        )
    return p, in_names, out_names


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

    # Reskin for export (same model object; eager attention forced).
    reskin_for_export(model)

    # Seed past state from the first token via the decode wrapper itself
    # (start from empty past, feed tokens one at a time).
    bsz = 1
    sk0 = torch.zeros(bsz, 0)
    sv0 = torch.zeros(bsz, 0)
    pk = [torch.zeros(bsz, kvh, 0, hd) for _ in range(n_layers)]
    pv = [torch.zeros(bsz, kvh, 0, hd) for _ in range(n_layers)]

    # Export with a NON-degenerate example (past_len=1) so torch.export infers a
    # genuine dynamic sequence axis; the actual loop still starts from past_len=0.
    ex_sk0 = torch.zeros(bsz, 1)
    ex_sv0 = torch.zeros(bsz, 1)
    ex_pk = [torch.zeros(bsz, kvh, 1, hd) for _ in range(n_layers)]
    ex_pv = [torch.zeros(bsz, kvh, 1, hd) for _ in range(n_layers)]
    example = [torch.tensor([[seq[0]]]), ex_sk0, ex_sv0]
    for li in range(n_layers):
        example += [ex_pk[li], ex_pv[li]]
    onnx_path, in_names, out_names = _export_decode(model, example)

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
        print(f"\nPASS: decode matches prefill/HF (max |diff| = {max_diff:.3e})")
        return 0
    print(f"\nFAIL: decode diverges (max |diff| = {max_diff:.3e})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
