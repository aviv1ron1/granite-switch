# SPDX-License-Identifier: Apache-2.0
"""ONNX export parity tests for the Granite Switch backend.

GATE 2: the exported (branchless, reskinned) prefill graph, run under
``onnxruntime``, must match the HF backend's logits to floating-point tolerance
on a tiny CPU model with active adapters.

Parity is ``allclose`` rather than bit-exact (the branchless gather+matmul
reorders the LoRA reduction vs the HF per-adapter loop — consistent with
CLAUDE.md §9, which only claims bit-exact for the fused/vLLM path).

CPU-only; no model download (uses the in-memory tiny config, the same pattern as
``tests/hf/test_model_forward.py``).
"""

import numpy as np
import pytest

from ._onnx_parity_worker import run_parity, run_prefill_decode_handoff
from ._decode_parity_worker import run_decode_parity


@pytest.mark.parametrize("seq_len", [6, 12])
def test_onnx_prefill_matches_hf(seq_len):
    """GATE 2: exported prefill logits match the HF backend (active control token)."""
    ok, max_diff, mean_diff = run_parity(seq_len=seq_len, verbose=False)
    assert ok, f"ONNX vs HF logits diverge: max|diff|={max_diff:.3e}"
    # Tight sanity bound: floating-point noise, not a structural difference.
    assert max_diff < 1e-3, f"max|diff| too large: {max_diff:.3e}"


def test_onnx_decode_matches_prefill():
    """GATE 3: KV-cached step-by-step decode matches the full HF forward.

    Exercises the threaded cumulative switch state across positions *after* a
    control token — the subtle correctness trap of incremental generation.
    """
    ok, max_diff, per_pos = run_decode_parity(verbose=False)
    assert ok, f"decode diverges from prefill/HF: max|diff|={max_diff:.3e} per_pos={per_pos}"
    assert max_diff < 1e-3, f"max|diff| too large: {max_diff:.3e}"


@pytest.mark.parametrize("seq_len", [6, 12])
def test_prefill_decode_handoff_matches_replay(seq_len):
    """The browser path: batched prefill seeds decode identically to a per-token replay.

    Runs the whole prompt through the prefill graph once, then one decode step on
    the next token seeded by prefill's present_* state, and asserts the next-token
    logits match a pure per-token decode replay of the same prompt. Guards the
    prefill->decode name/shape interlock and the batched-prefill numerics that the
    browser runtime (and the ~30s-to-1-step speedup) depend on.
    """
    ok, max_diff = run_prefill_decode_handoff(seq_len=seq_len, verbose=False)
    assert ok, f"prefill->decode handoff diverges from replay: max|diff|={max_diff:.3e}"
    assert max_diff < 1e-3, f"max|diff| too large: {max_diff:.3e}"
