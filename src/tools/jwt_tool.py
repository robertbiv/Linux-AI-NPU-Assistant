# SPDX-License-Identifier: GPL-3.0-or-later
"""JWT tool — decode JSON Web Tokens locally."""

import base64
import json
import logging
from typing import Any

from src.tools._base import SearchResult, Tool, ToolResult

logger = logging.getLogger(__name__)


def _decode_base64url(data: str) -> bytes:
    # Pad to multiple of 4
    data += "=" * ((4 - len(data) % 4) % 4)
    # Convert urlsafe to standard
    return base64.urlsafe_b64decode(data)


class JWTDecoderTool(Tool):
    """Decode JWT header and payload locally."""

    name = "jwt_decode"
    description = "Decode the header and payload of a JSON Web Token (JWT) locally. Does not verify signatures."
    parameters_schema = {
        "type": "object",
        "properties": {
            "token": {
                "type": "string",
                "description": "The JWT string to decode.",
            },
        },
        "required": ["token"],
    }

    def run(self, args: dict[str, Any]) -> ToolResult:
        token = args.get("token", "").strip()

        if not token:
            return ToolResult(tool_name=self.name, error="'token' is required.")

        parts = token.split(".")
        if len(parts) not in (2, 3):
            return ToolResult(
                tool_name=self.name,
                error="Invalid JWT format. Expected 2 or 3 dot-separated parts.",
            )

        try:
            header_json = _decode_base64url(parts[0]).decode("utf-8")
            payload_json = _decode_base64url(parts[1]).decode("utf-8")

            # Pretty print if possible
            try:
                header = json.dumps(json.loads(header_json), indent=2)
                payload = json.dumps(json.loads(payload_json), indent=2)
            except json.JSONDecodeError:
                header = header_json
                payload = payload_json

            snippet = f"Header:\n{header}\n\nPayload:\n{payload}"

            if len(parts) == 3 and parts[2]:
                snippet += f"\n\nSignature:\n{parts[2]}"

            return ToolResult(
                tool_name=self.name,
                results=[SearchResult(path="jwt", snippet=snippet)],
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("JWT decode error: %s", exc)
            return ToolResult(
                tool_name=self.name,
                error=f"JWT decoding failed: {exc}",
            )
