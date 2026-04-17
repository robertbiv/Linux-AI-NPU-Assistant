# SPDX-License-Identifier: GPL-3.0-or-later
"""Hash tool — calculate file and string hashes."""

import hashlib
import logging
from pathlib import Path
from typing import Any

from src.tools._base import SearchResult, Tool, ToolResult

logger = logging.getLogger(__name__)

_SUPPORTED_ALGORITHMS = ["md5", "sha1", "sha256", "sha512"]


class HashTool(Tool):
    """Calculate cryptographic hashes of text or files."""

    name = "hash"
    description = (
        "Calculate a cryptographic hash (SHA-256, MD5, etc.) of text or a file. "
        "Useful for verifying file integrity locally."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "algorithm": {
                "type": "string",
                "enum": _SUPPORTED_ALGORITHMS,
                "description": "The hash algorithm to use (e.g., 'sha256').",
            },
            "text": {
                "type": "string",
                "description": "The text to hash. Use this OR 'file_path'.",
            },
            "file_path": {
                "type": "string",
                "description": "The absolute path of the file to hash. Use this OR 'text'.",
            },
        },
        "required": ["algorithm"],
    }

    def run(self, args: dict[str, Any]) -> ToolResult:
        algo = args.get("algorithm", "").lower().strip()
        text = args.get("text")
        file_path = args.get("file_path")

        if algo not in _SUPPORTED_ALGORITHMS:
            return ToolResult(
                tool_name=self.name,
                error=f"Unsupported algorithm '{algo}'. Supported: {', '.join(_SUPPORTED_ALGORITHMS)}",
            )

        if text is not None and file_path is not None:
            return ToolResult(
                tool_name=self.name,
                error="Provide either 'text' or 'file_path', not both.",
            )

        if text is None and file_path is None:
            return ToolResult(
                tool_name=self.name,
                error="Provide either 'text' or 'file_path'.",
            )

        hasher = hashlib.new(algo)

        try:
            if text is not None:
                hasher.update(text.encode("utf-8"))
                result_hash = hasher.hexdigest()
                snippet = f"{algo}('{text}') = {result_hash}"
                return ToolResult(
                    tool_name=self.name,
                    results=[SearchResult(path="hash:text", snippet=snippet)],
                )

            if file_path is not None:
                path = Path(file_path).expanduser().resolve()
                if not path.is_file():
                    return ToolResult(
                        tool_name=self.name,
                        error=f"File not found or is a directory: {file_path}",
                    )

                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096 * 1024), b""):
                        hasher.update(chunk)

                result_hash = hasher.hexdigest()
                snippet = f"{algo}({path.name}) = {result_hash}"
                return ToolResult(
                    tool_name=self.name,
                    results=[SearchResult(path=str(path), snippet=snippet)],
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("HashTool error: %s", exc)
            return ToolResult(tool_name=self.name, error=f"Hashing failed: {exc}")

        return ToolResult(tool_name=self.name, error="Unknown error")
