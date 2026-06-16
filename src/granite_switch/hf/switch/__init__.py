# SPDX-License-Identifier: Apache-2.0
"""Adapter switching implementations for Granite Switch (HuggingFace).

Provides the adapter-selection switches and a factory that picks the
implementation from ``config.switch_impl``:

* ``"single"`` (default) -- :class:`SingleSwitch`, control-token-driven selection
  with token-exchange (production path).
* ``"external"`` -- :class:`ExternalSelectionSwitch`, caller-supplied per-token
  indices with no LUT (KV-cache contamination experiments).
"""

from .single import SingleSwitch
from .external import ExternalSelectionSwitch

__all__ = [
    "SingleSwitch",
    "ExternalSelectionSwitch",
    "create_switch",
]


def create_switch(config, layer_idx=0):
    """Factory: create the switch selected by ``config.switch_impl``.

    Args:
        config: GraniteSwitchConfig
        layer_idx: Layer index for cache management (default: 0)

    Returns:
        A switch module (``SingleSwitch`` or ``ExternalSelectionSwitch``).
    """
    impl = getattr(config, "switch_impl", "single")
    if impl == "external":
        return ExternalSelectionSwitch(
            num_adapters=config.num_adapters,
            config=config,
            layer_idx=layer_idx,
        )
    if impl == "single":
        return SingleSwitch(
            num_adapters=config.num_adapters,
            config=config,
            control_token_gain=getattr(config, "control_token_gain", 15.0),
            switch_head_dim=config.switch_head_dim,
            layer_idx=layer_idx,
        )
    raise ValueError(
        f"Unknown switch_impl={impl!r}; expected 'single' or 'external'"
    )
