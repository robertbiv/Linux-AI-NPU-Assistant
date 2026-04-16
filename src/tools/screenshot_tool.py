# SPDX-License-Identifier: GPL-3.0-or-later
"""Screenshot tool — capture the screen without including this application.

The tool is registered in the tool registry and may be invoked by the AI
assistant (with user approval) or called directly by the UI on every send.

The capture path avoids changing window opacity because some platform
plugins do not support it.  If the desktop environment or compositor cannot
exclude the app window automatically, the screenshot is still captured
without forcing transparency changes.

Usage (as a Tool called by the AI)::

    [TOOL: screenshot {"monitor": 0, "jpeg_quality": 75}]

Usage (direct call from UI)::

    tool = ScreenshotTool()
    result = tool.run({})
    if not result.error:
        b64 = result.results[0].snippet   # base64-encoded JPEG
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

from src.tools._base import SearchResult, Tool, ToolResult

logger = logging.getLogger(__name__)

# Directory where captured screenshots are saved (session-scoped temp dir).
_SCREENSHOT_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "npu-assistant-screenshots"


class ScreenshotTool(Tool):
    """Capture the primary screen and return a base64-encoded JPEG.

    The application window is not forced transparent during capture.  A saved
    copy is also written to ``$XDG_RUNTIME_DIR/npu-assistant-screenshots/``.

    Parameters accepted in *args*
    ------------------------------
    monitor : int
        Monitor index.  ``0`` = virtual desktop (all monitors), ``1`` =
        primary physical monitor.  Default ``0``.
    jpeg_quality : int
        JPEG compression quality 1–95.  Default ``75``.
    save : bool
        When ``true`` (default) persist the JPEG to disk in addition to
        returning the base64 string.
    """

    name = "screenshot"
    description = (
        "Capture the current screen (excluding this application) and return "
        "a base64-encoded JPEG image.  Optionally save to disk."
    )
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "monitor": {
                "type": "integer",
                "description": "Monitor index (0 = all monitors, 1 = primary).",
                "default": 0,
            },
            "jpeg_quality": {
                "type": "integer",
                "description": "JPEG quality 1–95.",
                "default": 75,
            },
            "save": {
                "type": "boolean",
                "description": "Whether to save the screenshot to disk.",
                "default": True,
            },
        },
        "required": [],
    }

    def __init__(self, hide_opacity_fn: "callable | None" = None) -> None:
        """Initialise the tool.

        The optional parameter is accepted for compatibility but ignored.
        """

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, args: dict[str, Any]) -> ToolResult:  # noqa: ANN201
        """Execute the screenshot capture.

        Parameters
        ----------
        args:
            Dict with optional keys: ``monitor`` (int), ``jpeg_quality`` (int),
            ``save`` (bool).
        """
        monitor = int(args.get("monitor", 0))
        quality = int(args.get("jpeg_quality", 75))
        do_save = bool(args.get("save", True))

        quality = max(1, min(95, quality))

        try:
            jpeg_bytes = self._capture(monitor, quality)
        except Exception as exc:  # noqa: BLE001
            logger.error("Screenshot capture failed: %s", exc)
            return ToolResult(tool_name=self.name, error=str(exc))

        b64 = base64.b64encode(jpeg_bytes).decode("ascii")

        saved_path = ""
        if do_save:
            saved_path = self._save(jpeg_bytes)

        snippet = b64
        result = SearchResult(
            path=saved_path or "(memory only)",
            snippet=snippet,
        )
        return ToolResult(tool_name=self.name, results=[result])

    # ── Class-level helpers (used by tests) ───────────────────────────────────

    @staticmethod
    def capture_for_send(
        window: "Any | None" = None,
        monitor: int = 0,
        jpeg_quality: int = 75,
    ) -> bytes | None:
        """Convenience method: capture the screen and return JPEG bytes."""
        try:
            from src.screen_capture import capture  # noqa: PLC0415
            return capture(monitor=monitor, jpeg_quality=jpeg_quality)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Screen capture failed: %s", exc)
            return None

    @staticmethod
    def _capture(monitor: int, quality: int) -> bytes:
        from src.screen_capture import capture  # noqa: PLC0415
        return capture(monitor=monitor, jpeg_quality=quality)

    @staticmethod
    def _save(jpeg_bytes: bytes) -> str:
        _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"screenshot_{int(time.time() * 1000)}.jpg"
        out = _SCREENSHOT_DIR / filename
        out.write_bytes(jpeg_bytes)
        logger.debug("Screenshot saved: %s", out)
        return str(out)
