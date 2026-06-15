# SPDX-License-Identifier: Apache-2.0
"""Decode-path (KV-cached) export wrapper for Granite Switch.

The prefill graph (:mod:`granite_switch.onnx.wrapper`) processes the whole prompt
at once. Incremental generation in the browser feeds one token at a time and
must reuse cached K/V — and, crucially, must preserve the switch's *cumulative*
adapter selection across steps.

This wrapper exposes a flat single-token decode signature with explicit tensor
state (no HF ``Cache`` objects), suitable for ``onnxruntime-web``:

inputs
    input_ids            [b, 1]                      the new token
    past_switch_key0     [b, past_len]               switch signal: ±gain per past pos
    past_switch_val0     [b, past_len]               switch signal: adapter id per past pos
    past_key.{L}         [b, kv_heads, past_len, hd] per-layer cached keys
    past_value.{L}       [b, kv_heads, past_len, hd] per-layer cached values

outputs
    logits               [b, 1, vocab]
    present_switch_key0  [b, past_len+1]
    present_switch_val0  [b, past_len+1]
    present_key.{L}      [b, kv_heads, past_len+1, hd]
    present_value.{L}    [b, kv_heads, past_len+1, hd]

Threading ``switch_key0`` / ``switch_val0`` is what keeps cumulative adapter
selection correct: the new token's adapter index is computed by attending over
the concatenated past+current switch signal, identical to what prefill computes
for that absolute position.
"""

import torch
import torch.nn as nn
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    apply_rotary_pos_emb,
    repeat_kv,
)


def _forward_core(
    self,
    input_ids,            # [b, q_len]
    past_switch_key0,     # [b, past_len]
    past_switch_val0,     # [b, past_len]
    past_keys,            # list[L] of [b, kvh, past_len, hd]
    past_values,          # list[L] of [b, kvh, past_len, hd]
):
    """Shared forward for BOTH prefill (q_len=N, past_len=0) and decode (q_len=1).

    This is the single source of truth for the Granite Switch ONNX forward: the
    cumulative switch selection, embed, RoPE, per-layer attention over past+current
    K/V, and MLP — all generalized over ``q_len``. The decode wrapper calls it with
    a single new token and a non-empty past; the prefill wrapper calls it with the
    whole prompt and an empty past. Keeping them on ONE code path guarantees prefill
    and decode are bit-faithful to each other (the prefill->decode handoff depends on
    it). Mirrors the Python HF model's batched-prefill-then-cached-decode forward.

    Returns ``(logits [b,q_len,vocab], key0_full [b,total], val0_full [b,total],
    present_key.{i} [b,kvh,total,hd]…, present_value.{i}…)`` where total = past+q_len.
    """
    gsm = self.gsm
    switch = gsm.switch
    bsz, q_len = input_ids.shape
    device = input_ids.device
    past_len = past_switch_key0.shape[1]

    # ── Switch: cumulative adapter selection over past+current signal ──
    # compute_signals / select_from_signals already handle q_len >= 1, so this is
    # identical for prefill and decode.
    key0_cur, val0_cur = switch.compute_signals(input_ids, gsm.adapter_token_ids)
    key0_full = torch.cat([past_switch_key0, key0_cur], dim=1)  # [b, past+q_len]
    val0_full = torch.cat([past_switch_val0, val0_cur], dim=1)
    adapter_indices = switch.select_from_signals(key0_full, val0_full, q_len=q_len)
    modified_input_ids = switch.rewrite_tokens(input_ids)

    # ── Embed ──
    h = gsm.embed_tokens(modified_input_ids) * gsm.embedding_multiplier

    # ── RoPE for the current query positions (absolute past_len .. past_len+q_len-1) ──
    position_ids = torch.arange(
        past_len, past_len + q_len, dtype=torch.long, device=device
    ).unsqueeze(0)
    pe = gsm.rotary_emb(h, position_ids=position_ids) if gsm.rotary_emb is not None else None
    cos, sin = pe if pe is not None else (None, None)

    # ── Causal mask over the [q_len, total_len] block ──
    # query i (absolute pos past_len+i) may attend to key j iff j <= past_len+i.
    # For decode (q_len=1) this is all-allowed (a single query sees every past+current
    # key), matching the previous no-mask behavior exactly. For prefill (past_len=0)
    # it is the standard lower-triangular [q_len, q_len] mask.
    total_len = past_len + q_len
    qpos = torch.arange(q_len, device=device).view(q_len, 1) + past_len
    kpos = torch.arange(total_len, device=device).view(1, total_len)
    causal_bias = torch.where(
        kpos <= qpos,
        torch.zeros((), dtype=torch.float32, device=device),
        torch.full((), float("-inf"), dtype=torch.float32, device=device),
    ).view(1, 1, q_len, total_len)  # [1,1,q_len,total]

    present_keys = []
    present_values = []
    for li, layer in enumerate(gsm.layers):
        attn = layer.self_attn
        residual = h
        x = layer.input_layernorm(h)

        # fused QKV (+ LoRA)
        if getattr(attn, "has_qkv_lora", False):
            qkv = attn.qkv_proj(x, adapter_indices)
        else:
            qkv = attn.qkv_proj(x)
        q_size = attn.num_heads * attn.head_dim
        kv_size = attn.num_key_value_heads * attn.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(bsz, q_len, attn.num_heads, attn.head_dim)
        k = k.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim)
        v = v.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim)
        if getattr(attn, "qk_norm", False):
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        if pe is not None:
            qt = q.transpose(1, 2)
            kt = k.transpose(1, 2)
            qt, kt = apply_rotary_pos_emb(qt, kt, cos, sin)
            q = qt.transpose(1, 2)
            k = kt.transpose(1, 2)
        # to [b, heads, seq, hd]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # concat past K/V (past_len may be 0 for prefill)
        k = torch.cat([past_keys[li], k], dim=2)   # [b, kvh, past+q_len, hd]
        v = torch.cat([past_values[li], v], dim=2)
        present_keys.append(k)
        present_values.append(v)

        # attention (eager): expand KV groups, scaled dot product, causal mask.
        kx = repeat_kv(k, attn.num_key_value_groups)
        vx = repeat_kv(v, attn.num_key_value_groups)
        scores = torch.matmul(q, kx.transpose(-1, -2)) * attn.scaling  # [b,h,q_len,total]
        scores = scores + causal_bias
        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, vx)        # [b, h, q_len, hd]
        attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, attn.num_heads * attn.head_dim)

        if getattr(attn, "has_o_lora", False):
            attn_out = attn.o_proj(attn_out, adapter_indices)
        else:
            attn_out = attn.o_proj(attn_out)
        h = residual + attn_out * layer.residual_multiplier

        # MLP (+ LoRA), adapter_indices threaded explicitly
        residual = h
        x = layer.post_attention_layernorm(h)
        mlp = layer.shared_mlp
        if layer._has_shared_input_lora:
            up = mlp.input_linear(x, adapter_indices)
        else:
            up = mlp.input_linear(x)
        c = up.chunk(2, dim=-1)
        act = mlp.activation(c[0]) * c[1]
        if layer._has_shared_output_lora:
            mo = mlp.output_linear(act, adapter_indices)
        else:
            mo = mlp.output_linear(act)
        h = residual + mo * layer.residual_multiplier

    h = gsm.norm(h)
    logits = self.m.lm_head(h)
    if getattr(self.m.config, "logits_scaling", 1.0) != 1.0:
        logits = logits / self.m.config.logits_scaling

    return (logits, key0_full, val0_full, *present_keys, *present_values)


class OnnxDecodeWrapper(nn.Module):
    """Single-token KV-cached decode over a reskinned GraniteSwitch model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.m = model
        gsm = model.model
        self.gsm = gsm
        self.num_layers = len(gsm.layers)

    def forward(self, input_ids, past_switch_key0, past_switch_val0, *past_kv):
        # past_kv is flattened [past_key.0, past_value.0, past_key.1, ...]
        past_keys = past_kv[0::2]
        past_values = past_kv[1::2]
        return _forward_core(
            self, input_ids, past_switch_key0, past_switch_val0, past_keys, past_values
        )


class OnnxPrefillStateWrapper(nn.Module):
    """Batched prefill that emits the SAME state tuple shape as the decode graph.

    Runs the whole prompt through :func:`_forward_core` with an EMPTY past
    (``past_len=0``), so it produces ``logits [b,seq,vocab]`` plus the KV + switch
    state (``present_*``) needed to seed incremental decode. This is the Python HF
    model's batched prefill: ONE forward over the prompt instead of replaying it
    token-by-token through the decode graph (the old browser behavior).

    The output names/shapes MUST match the decode graph's ``present_*`` outputs so
    transformers.js's cache (present -> past_key_values) feeds them straight into the
    first decode step. See ``granite_switch.onnx.export.export_prefill``.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.m = model
        gsm = model.model
        self.gsm = gsm
        self.num_layers = len(gsm.layers)

    def forward(self, input_ids):
        bsz, _ = input_ids.shape
        device = input_ids.device
        kvh = self.gsm.layers[0].self_attn.num_key_value_heads
        hd = self.gsm.layers[0].self_attn.head_dim
        # Empty past: zero-length switch signal and zero-length per-layer K/V.
        zero_switch = torch.zeros(bsz, 0, dtype=torch.float32, device=device)
        empty_k = [torch.zeros(bsz, kvh, 0, hd, dtype=torch.float32, device=device)
                   for _ in range(self.num_layers)]
        empty_v = [torch.zeros(bsz, kvh, 0, hd, dtype=torch.float32, device=device)
                   for _ in range(self.num_layers)]
        return _forward_core(self, input_ids, zero_switch, zero_switch, empty_k, empty_v)
