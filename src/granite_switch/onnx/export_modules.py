# SPDX-License-Identifier: Apache-2.0
"""Branchless, ONNX-traceable forward implementations for Granite Switch.

The HF backend's switched-LoRA layers (:mod:`granite_switch.hf.core.lora`) use
data-dependent control flow that ``torch.onnx.export`` cannot trace:

* ``torch.any(adapter_indices > 0)`` early-exit,
* ``adapter_indices[mask].unique()`` (dynamic-shape output),
* a Python ``for adapter_idx in active_adapters`` loop,
* boolean ``torch.where(mask)[0]`` index extraction.

The classes here subclass the HF layers and reuse their **exact** parameter
tensors (``lora_A`` / ``lora_B`` / ``lora_A_slices`` / ``lora_B_slices``), but
replace ``forward`` with a branchless formulation:

    1. Prepend a zero adapter row at index 0 so ``adapter_index == 0`` (base)
       gathers a zero matrix and contributes an exact-zero delta.
    2. ``Gather`` the per-token A/B matrices by adapter index.
    3. Batched matmul ``(x · Aᵀ) · Bᵀ`` → per-token LoRA delta.
    4. ``output = base_layer(x) + delta``.

All resulting ops (Gather, MatMul, Add, Reshape) are supported by
onnxruntime-web on both the WASM and WebGPU execution providers.

.. note::
   The gather+batched-matmul reduction order differs from the HF loop, so
   parity against the HF backend is ``allclose`` rather than bit-exact. This is
   consistent with the repo only claiming bit-exact equivalence for the
   fused/vLLM path (see CLAUDE.md §9).
"""

from typing import Optional

import torch
import torch.nn.functional as F

from granite_switch.hf.core.lora import (
    SwitchedLoRALinear,
    MergedSwitchedLoRALinear,
)
from granite_switch.hf.switch.single import SingleSwitch
from granite_switch.hf.modeling_granite_switch import GraniteSwitchAttentionDecoderLayer


def _branchless_lora_delta(
    x_flat: torch.Tensor,        # [tokens, in_features]
    adapter_indices_flat: torch.Tensor,  # [tokens] long, 0=base, 1..N=adapters
    lora_A: torch.Tensor,        # [num_adapters, 1, rank, in_features]
    lora_B: torch.Tensor,        # [num_adapters, 1, out_features, rank]
) -> torch.Tensor:
    """Per-token LoRA delta via gather + batched matmul (no data-dependent flow).

    Returns ``[tokens, out_features]``. Tokens with ``adapter_index == 0`` get an
    exact-zero delta because we gather a prepended zero adapter row at index 0.
    """
    num_adapters = lora_A.shape[0]
    rank = lora_A.shape[2]
    in_features = lora_A.shape[3]
    out_features = lora_B.shape[2]

    # Squeeze the padding dim → A:[N, rank, in], B:[N, out, rank]
    A = lora_A.reshape(num_adapters, rank, in_features)
    B = lora_B.reshape(num_adapters, out_features, rank)

    # Prepend a zero row at index 0 so adapter index 0 (base) → zero delta.
    # Result: A_full:[N+1, rank, in], B_full:[N+1, out, rank].
    A_full = torch.cat([A.new_zeros(1, rank, in_features), A], dim=0)
    B_full = torch.cat([B.new_zeros(1, out_features, rank), B], dim=0)

    # Gather per-token matrices by adapter index (Gather, axis=0).
    A_tok = A_full.index_select(0, adapter_indices_flat)  # [tokens, rank, in]
    B_tok = B_full.index_select(0, adapter_indices_flat)  # [tokens, out, rank]

    # delta = (x · Aᵀ) · Bᵀ, batched over tokens.
    # x_flat: [tokens, in] → [tokens, 1, in]
    xt = x_flat.unsqueeze(1)                                  # [tokens, 1, in]
    down = torch.bmm(xt, A_tok.transpose(1, 2))              # [tokens, 1, rank]
    up = torch.bmm(down, B_tok.transpose(1, 2))              # [tokens, 1, out]
    return up.squeeze(1)                                      # [tokens, out]


class OnnxSwitchedLoRALinear(SwitchedLoRALinear):
    """Branchless, export-clean drop-in for :class:`SwitchedLoRALinear`.

    Reuses the parent's ``base_layer`` / ``lora_A`` / ``lora_B`` parameters;
    only ``forward`` changes.
    """

    def forward(
        self, x: torch.Tensor, adapter_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if adapter_indices is None:
            adapter_indices = self._adapter_indices

        output = self.base_layer(x)

        # No branch on adapter content: a zero index simply yields a zero delta.
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x_flat = x.reshape(-1, self.in_features)
            adapter_indices_flat = adapter_indices.reshape(-1)
            output_flat = output.reshape(-1, self.out_features)
        else:
            x_flat = x
            adapter_indices_flat = adapter_indices
            output_flat = output

        delta = _branchless_lora_delta(
            x_flat, adapter_indices_flat, self.lora_A, self.lora_B
        )
        output_flat = output_flat + delta

        if len(original_shape) == 3:
            return output_flat.reshape(batch_size, seq_len, self.out_features)
        return output_flat


class OnnxMergedSwitchedLoRALinear(MergedSwitchedLoRALinear):
    """Branchless, export-clean drop-in for :class:`MergedSwitchedLoRALinear`.

    Applies the branchless delta to each fused slice (Q/K/V or gate/up) and
    concatenates, reusing the parent's ``lora_A_slices`` / ``lora_B_slices``.
    """

    def forward(
        self, x: torch.Tensor, adapter_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if adapter_indices is None:
            adapter_indices = self._adapter_indices

        output = self.base_layer(x)

        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x_flat = x.reshape(-1, self.in_features)
            adapter_indices_flat = adapter_indices.reshape(-1)
            output_flat = output.reshape(-1, sum(self.output_slices))
        else:
            x_flat = x
            adapter_indices_flat = adapter_indices
            output_flat = output

        # Compute each slice's delta, concat in slice order, add once.
        deltas = []
        for slice_idx in range(self.num_slices):
            deltas.append(
                _branchless_lora_delta(
                    x_flat,
                    adapter_indices_flat,
                    self.lora_A_slices[slice_idx],
                    self.lora_B_slices[slice_idx],
                )
            )
        delta = torch.cat(deltas, dim=-1)  # [tokens, sum(output_slices)]
        output_flat = output_flat + delta

        if len(original_shape) == 3:
            return output_flat.reshape(batch_size, seq_len, sum(self.output_slices))
        return output_flat


class OnnxGraniteSwitchAttentionDecoderLayer(GraniteSwitchAttentionDecoderLayer):
    """Export-clean decoder layer: thread ``adapter_indices`` to the MLP explicitly.

    The HF layer passes ``adapter_indices`` to the shared MLP via a Python
    attribute side-effect (``_set_shared_mlp_context`` sets
    ``shared_mlp.input_linear._adapter_indices`` then resets it to ``None``).
    ``torch.export`` does not capture that attribute mutation as dataflow, so the
    traced graph applies the MLP LoRA against a stale/``None`` index — silently
    corrupting MLP-LoRA tokens (attention LoRA is fine, since it gets
    ``adapter_indices`` as a real positional argument).

    This override inlines the shared-MLP forward and passes ``adapter_indices``
    directly to ``input_linear`` / ``output_linear`` as arguments. MoE is not
    supported on the export path yet (dense Granite only); see the assertion.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        adapter_indices=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _attn_w, _present = self.self_attn(
            hidden_states=hidden_states,
            adapter_indices=adapter_indices,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        assert not self.has_experts, (
            "ONNX export path supports dense Granite only (MoE not yet wired)."
        )
        # Inline GraniteMoeHybridMLP.forward, threading adapter_indices as args.
        mlp = self.shared_mlp
        if self._has_shared_input_lora:
            up = mlp.input_linear(hidden_states, adapter_indices)
        else:
            up = mlp.input_linear(hidden_states)
        chunks = up.chunk(2, dim=-1)
        act = mlp.activation(chunks[0]) * chunks[1]
        if self._has_shared_output_lora:
            mlp_out = mlp.output_linear(act, adapter_indices)
        else:
            mlp_out = mlp.output_linear(act)

        hidden_states = residual + mlp_out * self.residual_multiplier

        outputs = (hidden_states,)
        if use_cache:
            outputs += (past_key_values,)
        return outputs


def reskin_lora_modules_for_export(model: torch.nn.Module) -> int:
    """In-place: swap every switched-LoRA layer's class to its Onnx* variant.

    Reassigns ``__class__`` so the branchless ``forward`` is used while keeping
    the exact same parameters/buffers. Also swaps decoder layers to the Onnx
    variant that threads ``adapter_indices`` to the MLP as explicit dataflow
    (the HF layer uses an attribute side-effect that ``torch.export`` drops).
    Returns the number of LoRA layers swapped.
    """
    swapped = 0
    for module in model.modules():
        # Order matters: MergedSwitchedLoRALinear is not a subclass of
        # SwitchedLoRALinear, so a plain isinstance chain is unambiguous.
        if type(module) is MergedSwitchedLoRALinear:
            module.__class__ = OnnxMergedSwitchedLoRALinear
            swapped += 1
        elif type(module) is SwitchedLoRALinear:
            module.__class__ = OnnxSwitchedLoRALinear
            swapped += 1
        elif type(module) is GraniteSwitchAttentionDecoderLayer:
            module.__class__ = OnnxGraniteSwitchAttentionDecoderLayer
    return swapped


class OnnxSingleSwitch(SingleSwitch):
    """Export-clean :class:`SingleSwitch`: functional (where-based) tensor writes.

    The HF switch uses in-place masked scatter (``key_states[...][mask] = gain``)
    to mark control tokens, which traces awkwardly. This recomputes the same
    Q/K/V purely functionally so ``torch.onnx.export`` produces clean Gather/
    Where/Softmax/MatMul ops. The cumulative-attention math and the
    token-exchange rewrite are otherwise identical to the parent.

    Cache handling (the cumulative switch K/V threaded across decode steps) is
    added in Phase 3; this Phase-1 variant targets the no-cache prefill export.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        adapter_token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position=None,
    ):
        bsz, q_len = input_ids.shape
        device = input_ids.device
        dtype = torch.float32

        # is_control[b, t] = 1.0 if input_ids[b,t] is any adapter control token.
        # adapter_id_per_pos[b, t] = (i+1) for control token of adapter i, else 0.
        # Both computed branchlessly from the adapter_token_ids table.
        # input_ids: [b, t, 1] vs adapter_token_ids: [N] → [b, t, N] match grid.
        matches = input_ids.unsqueeze(-1) == adapter_token_ids.view(1, 1, -1)  # [b,t,N]
        # adapter ids are 1..N for the N control tokens
        ids = torch.arange(
            1, self.num_adapters + 1, device=device, dtype=dtype
        ).view(1, 1, -1)
        adapter_id_per_pos = (matches.to(dtype) * ids).sum(dim=-1)  # [b, t]
        is_control = matches.any(dim=-1).to(dtype)                  # [b, t]

        # Build Q/K/V functionally. Only dim 0 of head_dim carries signal.
        # key dim0: -gain for non-control, +gain for control.
        key0 = torch.where(
            is_control > 0,
            torch.full_like(is_control, self.control_token_gain),
            torch.full_like(is_control, -self.control_token_gain),
        )  # [b, t]
        val0 = adapter_id_per_pos  # [b, t]

        def _pad_head(signal):  # [b, t] → [b, 1, t, head_dim] with signal in dim0
            # Functional construction (no in-place index assignment, which
            # traces unreliably under torch.export): put `signal` in channel 0
            # and zero-pad the remaining head_dim-1 channels.
            sig = signal.view(bsz, 1, q_len, 1)  # [b,1,t,1]
            if self.head_dim == 1:
                return sig
            pad = torch.zeros(
                (bsz, 1, q_len, self.head_dim - 1), device=device, dtype=dtype
            )
            return torch.cat([sig, pad], dim=-1)

        ones = torch.ones((bsz, q_len), device=device, dtype=dtype)
        query_states = _pad_head(ones)
        key_states = _pad_head(key0)
        value_states = _pad_head(val0)

        # Causal single-head attention → cumulative adapter selection.
        # scores[b,1,i,j] = q_i·k_j = key0_j for j<=i (q dim0 == 1).
        scores = torch.matmul(query_states, key_states.transpose(-1, -2))  # [b,1,t,t]
        scores = scores * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
        else:
            # explicit causal mask (no reliance on backend is_causal)
            causal = torch.full((q_len, q_len), float("-inf"), device=device, dtype=dtype)
            causal = torch.triu(causal, diagonal=1)
            scores = scores + causal.view(1, 1, q_len, q_len)
        weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(weights, value_states)  # [b,1,t,head_dim]

        sel = attn_output[:, 0, :, 0]  # [b, t]
        adapter_indices = torch.round(sel).to(torch.long)
        adapter_indices = torch.clamp(adapter_indices, 0, self.num_adapters)

        if self.control_to_substitute_lut is not None:
            sub_id_per_pos = self.control_to_substitute_lut[input_ids]
            modified_input_ids = torch.where(
                sub_id_per_pos >= 0, sub_id_per_pos, input_ids
            )
        else:
            modified_input_ids = input_ids

        return adapter_indices, modified_input_ids

    # ------------------------------------------------------------------
    # Decode-path helpers (Phase 3): the switch's adapter selection is
    # cumulative across the whole sequence, so single-token decode must see
    # the control-token signal from all prior positions. The switch K/V only
    # carry signal in head_dim channel 0, so we thread a compact per-position
    # ``key0`` / ``val0`` (shape ``[b, len]``) as explicit graph state instead
    # of full head_dim tensors.
    # ------------------------------------------------------------------
    def compute_signals(self, input_ids, adapter_token_ids):
        """Return ``(key0, val0)`` (each ``[b, q_len]``) for the given tokens.

        ``key0`` = +gain at control tokens, -gain elsewhere.
        ``val0`` = (i+1) at adapter-i control tokens, 0 elsewhere.
        """
        dtype = torch.float32
        device = input_ids.device
        matches = input_ids.unsqueeze(-1) == adapter_token_ids.view(1, 1, -1)
        ids = torch.arange(
            1, self.num_adapters + 1, device=device, dtype=dtype
        ).view(1, 1, -1)
        val0 = (matches.to(dtype) * ids).sum(dim=-1)            # [b, q_len]
        is_control = matches.any(dim=-1).to(dtype)              # [b, q_len]
        key0 = torch.where(
            is_control > 0,
            torch.full_like(is_control, self.control_token_gain),
            torch.full_like(is_control, -self.control_token_gain),
        )
        return key0, val0

    def select_from_signals(self, key0_full, val0_full, q_len):
        """Cumulative adapter selection for the last ``q_len`` query positions.

        ``key0_full`` / ``val0_full`` are ``[b, total_len]`` (past + current).
        Returns ``adapter_indices`` ``[b, q_len]`` for the current-token block.
        Each query attends causally over all keys up to its absolute position;
        with q·k = key0 this reduces to a softmax over key0 weighting val0.
        """
        bsz, total_len = key0_full.shape
        past_len = total_len - q_len
        # scores[b, i, j] = key0_full[b, j], masked so query i (absolute pos
        # past_len+i) only sees j <= past_len+i.
        scores = key0_full.unsqueeze(1).expand(bsz, q_len, total_len) * self.scaling
        # causal mask over the [q_len, total_len] block
        qpos = torch.arange(q_len, device=key0_full.device).view(q_len, 1) + past_len
        kpos = torch.arange(total_len, device=key0_full.device).view(1, total_len)
        allowed = kpos <= qpos                                   # [q_len, total_len]
        neg = torch.full_like(scores, float("-inf"))
        scores = torch.where(allowed.unsqueeze(0), scores, neg)
        weights = torch.softmax(scores, dim=-1)                  # [b, q_len, total_len]
        sel = torch.matmul(weights, val0_full.unsqueeze(-1)).squeeze(-1)  # [b, q_len]
        adapter_indices = torch.clamp(
            torch.round(sel).to(torch.long), 0, self.num_adapters
        )
        return adapter_indices

    def rewrite_tokens(self, input_ids):
        """Token-exchange rewrite (control id → substitute id)."""
        if self.control_to_substitute_lut is not None:
            sub = self.control_to_substitute_lut[input_ids]
            return torch.where(sub >= 0, sub, input_ids)
        return input_ids
