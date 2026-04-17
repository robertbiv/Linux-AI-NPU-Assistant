# SPDX-License-Identifier: GPL-3.0-or-later
"""Base64 tool — encode or decode Base64 text."""

import base64
import logging
from typing import Any

from src.tools._base import SearchResult, Tool, ToolResult

logger = logging.getLogger(__name__)


class Base64Tool(Tool):
    """Encode or decode text to/from Base64."""

    name = "base64"
    description = "Encode or decode text to/from Base64 representation."
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["encode", "decode"],
                "description": "Whether to encode to Base64 or decode from Base64.",
            },
            "text": {
                "type": "string",
                "description": "The text to encode or decode.",
            },
        },
        "required": ["action", "text"],
    }

    def run(self, args: dict[str, Any]) -> ToolResult:
        action = args.get("action", "").lower().strip()
        text = args.get("text", "")

        if action not in ("encode", "decode"):
            return ToolResult(
                tool_name=self.name, error="Action must be 'encode' or 'decode'."
            )

        if not text:
            return ToolResult(tool_name=self.name, error="'text' is required.")

        try:
            if action == "encode":
                encoded_bytes = base64.b64encode(text.encode("utf-8"))
                result_text = encoded_bytes.decode("ascii")
            else:
                decoded_bytes = base64.b64decode(text.encode("ascii"))
                result_text = decoded_bytes.decode("utf-8")

            snippet = f"{action.capitalize()}d Base64:\n{result_text}"
            return ToolResult(
                tool_name=self.name,
                results=[SearchResult(path=f"base64:{action}", snippet=snippet)],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Base64 error: %s", exc)
            return ToolResult(
                tool_name=self.name,
                error=f"Base64 {action} failed: {exc}",
            )
