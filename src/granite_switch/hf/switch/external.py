# SPDX-License-Identifier: Apache-2.0
"""ExternalSelectionSwitch: adapter indices supplied by the caller.

Unlike :class:`SingleSwitch`, this switch performs **no** attention and writes
**no** KV-cache slot (``num_cache_layers == 0``). It returns ``adapter_indices``
that were armed externally via :meth:`set_external_indices`, enabling experiments
that drive adapter selection independently of ``input_ids`` -- e.g. forcing an
adapter switch mid-conversation while sharing one KV cache (the KV-cache
contamination study).

The call contract matches ``SingleSwitch`` exactly so the switch is a drop-in
behind ``GraniteSwitchModel`` (selected via ``config.switch_impl == "external"``):

    forward(input_ids, adapter_token_ids, attention_mask=, past_key_values=,
            cache_position=) -> (adapter_indices, modified_input_ids)

Differences from ``SingleSwitch``:

* ``adapter_indices`` come from ``self._external_indices`` (set by the caller
  before forward) rather than from control tokens in ``input_ids``.
* No LUT, no token-exchange rewrite -- ``modified_input_ids`` *is* ``input_ids``.
  Safe because the experiment uses the **base** Granite tokenizer, whose
  ``input_ids`` never contain control-token ids that would need swapping.
* ``num_cache_layers == 0`` (the switch reserves no cache slot), so
  ``layer_offset == 0`` and every ``config.num_hidden_layers`` becomes a decoder
  layer with cache indices identical to a model with no switch layer.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache


class ExternalSelectionSwitch(nn.Module):
    """Switch that emits caller-supplied per-token adapter indices.

    Args:
        num_adapters: Number of LoRA adapters (kept for parity with SingleSwitch;
            indices are validated by the LoRA modules, not here).
        config: Model configuration (unused; accepted for factory symmetry).
        layer_idx: Layer index (unused -- this switch reserves no cache slot;
            accepted for factory symmetry).
    """

    def __init__(self, num_adapters: int, config=None, layer_idx: int = 0):
        super().__init__()
        self.num_adapters = num_adapters
        self.config = config
        # Reserves no cache slot, but kept for parity with SingleSwitch's API.
        self.layer_idx = layer_idx
        # Armed externally via set_external_indices(). Shape [batch, total_seq],
        # covering the full conversation; sliced per forward by cache_position.
        self._external_indices: Optional[torch.Tensor] = None

    @property
    def num_cache_layers(self) -> int:
        """This switch reserves no KV-cache slot (does no attention)."""
        return 0

    def set_external_indices(self, adapter_indices: Optional[torch.Tensor]) -> None:
        """Arm the per-token adapter indices for subsequent forward passes.

        Args:
            adapter_indices: ``[batch, seq]`` (or ``[batch, total_seq]`` spanning
                a whole conversation) long tensor, ``0`` = base, ``1..num_adapters``
                = adapter. Pass ``None`` to clear (forward then emits all-base
                zeros). Unlike SingleSwitch's one-shot derivation, the armed
                tensor persists across forwards until re-armed or cleared, so the
                generation hook can keep it in sync with ``cache_position``.
        """
        self._external_indices = adapter_indices

    def forward(
        self,
        input_ids: torch.Tensor,
        adapter_token_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return externally-armed adapter indices and pass input_ids through.

        Args:
            input_ids: ``[batch, seq_len]`` token ids.
            adapter_token_ids: Accepted and ignored (the model passes it
                unconditionally; this switch does no control-token detection).
            attention_mask, past_key_values, cache_position: Accepted for call
                parity; only ``cache_position`` is used (to slice the armed
                indices to the positions in this forward during incremental
                decode).

        Returns:
            ``(adapter_indices, modified_input_ids)`` where ``adapter_indices`` is
            ``[batch, seq_len]`` long on ``input_ids.device`` and
            ``modified_input_ids is input_ids`` (no rewrite).
        """
        bsz, q_len = input_ids.shape
        device = input_ids.device

        if self._external_indices is None:
            # Unarmed -> behave as pure base model.
            adapter_indices = torch.zeros((bsz, q_len), dtype=torch.long, device=device)
            return adapter_indices, input_ids

        idx = self._external_indices.to(device=device, dtype=torch.long)

        # The armed tensor spans the conversation planned so far; an incremental
        # decode forward only sees q_len tokens at absolute positions
        # cache_position. Select exactly those columns so each token gets the
        # adapter the caller planned for that position.
        if idx.shape[1] != q_len:
            if cache_position is not None:
                # Decode can run past the armed plan (generated tokens whose
                # positions were not planned ahead of time). Repeat the last
                # planned column for any position beyond the plan, so generation
                # stays on the last planned adapter without a per-step re-arm.
                # A generation hook may instead extend the plan to override this.
                armed_len = idx.shape[1]
                pos = cache_position.to(idx.device)
                clamped = torch.clamp(pos, max=armed_len - 1)
                idx = idx.index_select(1, clamped)
            else:
                idx = idx[:, -q_len:]

        assert idx.shape == (bsz, q_len), (
            f"external adapter_indices shape {tuple(idx.shape)} must match "
            f"input shape {(bsz, q_len)} (after cache_position slicing)"
        )

        # No token-exchange rewrite: base tokenizer means input_ids carries no
        # control-token ids, so the decoder embeds input_ids directly.
        return idx, input_ids
