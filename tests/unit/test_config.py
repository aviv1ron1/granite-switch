# SPDX-License-Identifier: Apache-2.0
"""Config validation tests for GraniteSwitchConfig.

Covers the validators in __init__, default values, and the config
fields that survived the legacy-hiding removal.
"""

import pytest

from granite_switch.config import _LEAKY_INHERITED_KEYS, GraniteSwitchConfig

# Keys the GraniteSwitch model reads at load time. They resemble the leaky inherited
# defaults but must ALWAYS serialize, so they must never be in the leaky set.
_LOAD_BEARING_KEYS = (
    "num_local_experts",
    "position_embedding_type",
    "shared_intermediate_size",
)

# ── Helper ────────────────────────────────────────────────────────────


def _valid_kwargs(num_adapters=2, **overrides):
    """Return kwargs for a valid token-exchange config."""
    adapter_names = [f"adapter_{i}" for i in range(num_adapters)]
    base = dict(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=num_adapters,
        adapter_token_ids=list(range(500, 500 + num_adapters)),
        adapter_substitute_token_ids=[1] * num_adapters,
        adapter_names=adapter_names,
        max_lora_rank=8,
        adapter_ranks=[8] * num_adapters,
    )
    base.update(overrides)
    return base


# ════════════════════════════════════════════════════════════════════
# 1. Config validation — every ValueError path
# ════════════════════════════════════════════════════════════════════


class TestConfigValidation:
    def test_negative_num_adapters_raises(self):
        with pytest.raises(ValueError, match="num_adapters must be >= 0"):
            GraniteSwitchConfig(**_valid_kwargs(num_adapters=-1, adapter_ranks=None))

    def test_adapter_token_ids_wrong_length_raises(self):
        with pytest.raises(ValueError, match="adapter_token_ids length"):
            GraniteSwitchConfig(**_valid_kwargs(adapter_token_ids=[500]))

    def test_substitute_ids_required_when_adapters_present(self):
        with pytest.raises(
            ValueError, match="adapter_substitute_token_ids is required"
        ):
            GraniteSwitchConfig(**_valid_kwargs(adapter_substitute_token_ids=None))

    def test_substitute_ids_wrong_length_raises(self):
        with pytest.raises(ValueError, match="adapter_substitute_token_ids length"):
            GraniteSwitchConfig(**_valid_kwargs(adapter_substitute_token_ids=[1]))

    def test_substitute_ids_negative_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            GraniteSwitchConfig(**_valid_kwargs(adapter_substitute_token_ids=[-1, 1]))

    def test_duplicate_adapter_token_ids_raises(self):
        with pytest.raises(ValueError, match="adapter_token_ids must be unique"):
            GraniteSwitchConfig(**_valid_kwargs(adapter_token_ids=[500, 500]))

    def test_adapter_ranks_required(self):
        with pytest.raises(ValueError, match="adapter_ranks must be provided"):
            GraniteSwitchConfig(**_valid_kwargs(adapter_ranks=None))

    def test_adapter_ranks_wrong_length(self):
        with pytest.raises(ValueError, match="adapter_ranks length"):
            GraniteSwitchConfig(**_valid_kwargs(adapter_ranks=[8]))

    def test_max_lora_rank_must_match(self):
        with pytest.raises(ValueError, match="max_lora_rank"):
            GraniteSwitchConfig(**_valid_kwargs(max_lora_rank=4))


# ════════════════════════════════════════════════════════════════════
# 2. Defaults
# ════════════════════════════════════════════════════════════════════


class TestConfigDefaults:
    def test_zero_adapter_default(self):
        cfg = GraniteSwitchConfig(num_adapters=0)
        assert cfg.num_adapters == 0
        assert cfg.adapter_token_ids is None
        assert cfg.adapter_substitute_token_ids is None

    def test_projection_head_dim_inferred_from_hidden_size(self):
        cfg = GraniteSwitchConfig(**_valid_kwargs())
        assert cfg.projection_head_dim == 64 // 4


# ════════════════════════════════════════════════════════════════════
# 3. Serialization: inherited-default keys the base never declared are pruned
# ════════════════════════════════════════════════════════════════════


class TestLeakyKeyFiltering:
    def test_leaky_keys_stripped_when_base_declared_none(self):
        """A dense/attention-only base declares no mamba/MoE keys -> none serialize."""
        cfg = GraniteSwitchConfig(**_valid_kwargs())
        out = cfg.to_dict()
        present = _LEAKY_INHERITED_KEYS & out.keys()
        assert present == set(), (
            f"leaky keys should be stripped, found: {sorted(present)}"
        )

    def test_base_declared_leaky_key_is_kept(self):
        """A key the base actually declared (e.g. a genuine hybrid) is preserved."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(base_declared_keys=["mamba_d_state"]))
        out = cfg.to_dict()
        assert "mamba_d_state" in out
        others = (_LEAKY_INHERITED_KEYS - {"mamba_d_state"}) & out.keys()
        assert others == set(), (
            f"other leaky keys should be stripped, found: {sorted(others)}"
        )

    def test_load_bearing_keys_always_present(self):
        """Keys the model reads at load time must survive filtering in every case."""
        for kwargs in (
            _valid_kwargs(),
            _valid_kwargs(base_declared_keys=["mamba_d_state"]),
        ):
            out = GraniteSwitchConfig(**kwargs).to_dict()
            for key in _LOAD_BEARING_KEYS:
                assert key in out, f"{key} must always serialize"
                assert key not in _LEAKY_INHERITED_KEYS

    def test_round_trip_stable(self):
        """save -> load -> save reproduces identical serialization (allow-set persists)."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(base_declared_keys=["mamba_d_state"]))
        first = cfg.to_dict()
        reloaded = GraniteSwitchConfig.from_dict(first)
        assert reloaded.to_dict() == first
