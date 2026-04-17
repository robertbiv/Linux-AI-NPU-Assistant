# SPDX-License-Identifier: GPL-3.0-or-later
"""UUID tool — generate UUIDs."""

import logging
import uuid
from typing import Any

from src.tools._base import SearchResult, Tool, ToolResult

logger = logging.getLogger(__name__)


class UUIDTool(Tool):
    """Generate random UUIDs."""

    name = "generate_uuid"
    description = "Generate a random UUID (Universally Unique Identifier) version 4."
    parameters_schema = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "Number of UUIDs to generate (default 1, max 100).",
                "default": 1,
            },
        },
        "required": [],
    }

    def run(self, args: dict[str, Any]) -> ToolResult:
        try:
            count = int(args.get("count", 1))
        except (ValueError, TypeError):
            count = 1

        count = max(1, min(100, count))

        uuids = [str(uuid.uuid4()) for _ in range(count)]

        snippet = "\n".join(uuids)
        return ToolResult(
            tool_name=self.name,
            results=[SearchResult(path="uuid_generator", snippet=snippet)],
        )
