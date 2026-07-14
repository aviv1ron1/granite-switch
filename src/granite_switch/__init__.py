# SPDX-License-Identifier: Apache-2.0
"""Granite Switch: Composable model building."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("granite-switch")
except PackageNotFoundError:
    # Running from a source tree that isn't installed (no distribution metadata).
    __version__ = "0.0.0+unknown"

from .config import GraniteSwitchConfig

__all__ = ["GraniteSwitchConfig", "__version__"]
