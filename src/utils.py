# SPDX-License-Identifier: GPL-3.0-or-later
"""General utilities."""

from __future__ import annotations

import os
from pathlib import Path


def is_running_in_flatpak() -> bool:
    """Return ``True`` if the process is running inside a Flatpak sandbox.

    Uses two independent indicators:

    1. The ``FLATPAK_ID`` environment variable — set automatically by the
       Flatpak runtime for every sandboxed process.
    2. The ``/.flatpak-info`` file — written by ``flatpak run`` into the
       mount namespace of the sandbox.

    Either indicator is sufficient; both are checked so that the function
    works correctly whether the app is started via ``flatpak run`` or from
    a development environment where only the env var is set for testing.
    """
    return bool(os.environ.get("FLATPAK_ID")) or Path("/.flatpak-info").exists()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
