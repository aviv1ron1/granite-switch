# vLLM Quantization — Status Summary

**Date**: 2026-05-13
**Branch**: `feature/quantized-inference-testing`
**Status**: Partially fixed, needs GPU validation

---

## Issues Found and Fixed

### 1. BnB dtype error in `SwitchedLoRALinear` (commit `93db1b0`)

**Problem**: When BitsAndBytes quantizes base layers, the weight dtype becomes `uint8`.
`SwitchedLoRALinear.__init__` used `base_layer.weight.dtype` to allocate LoRA params,
causing them to be `uint8` instead of float — breaking LoRA computation.

**Fix** (`src/granite_switch/vllm/core/lora.py:107-110`):
```python
if not dtype.is_floating_point:
    dtype = torch.bfloat16
```

### 2. OOM on A100 80GB with BnB loading (commit `3a91e71`)

**Problem**: BnB holds full-precision weights during quantization pass. Default
`gpu_memory_utilization=0.8` caused OOM.

**Fix**: Lowered to `0.5` in the diagnostic script.

### 3. xdist group marker (added `7a2b02b`, reverted `9cd26f6`)

Attempted to group quantization tests for xdist but reverted — not the right approach.

---

## What Exists

| File | Purpose | Status |
|------|---------|--------|
| `quantization/test_quantized_inference.py` | Standalone vLLM BnB diagnostic script | Ready to run on GPU, no pytest |
| `quantization/quantization_testing.ipynb` | Exploration notebook (FP8/GPTQ/AWQ via API server) | Unused, no outputs |
| `tests/hf/test_quantization.py` | Formal HF backend pytest (BnB NF4/FP4, Quanto INT4/FP8) | Passing |

---

## What's Missing

1. **No pytest coverage for vLLM + BnB** — only the standalone diagnostic script exists
2. **No GPU validation yet** — the dtype fix (`93db1b0`) hasn't been confirmed on a real GPU
   with the full vLLM loading path
3. **FP8/GPTQ/AWQ on vLLM** — the notebook was designed for this but never executed

---

## Key Parameters for vLLM BnB Loading

```python
from vllm import LLM

llm = LLM(
    model="ibm-granite/granite-switch-4.1-3b-preview",
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    gpu_memory_utilization=0.5,   # Lower than default — BnB needs headroom
    enforce_eager=True,            # CUDA graphs don't work with BnB
)
```

---

## Next Steps (when resuming)

1. Run `quantization/test_quantized_inference.py` on a GPU pod to confirm BnB fix works end-to-end
2. If it passes, convert into a proper pytest test in `tests/vllm/test_quantization.py`
3. Investigate FP8/GPTQ support via vLLM (different from BnB path)
