"""AI assistant backend — vision-capable LLM interaction.

Supported backends
------------------
- **ollama**  – Local Ollama server (recommended; supports llava and other
  vision models out of the box).
- **openai**  – OpenAI-compatible REST API (gpt-4o or any vision endpoint).
- **npu**     – AMD Ryzen AI ONNX model running on the NPU / iGPU.

Resource efficiency
-------------------
- ``requests`` is imported lazily; no persistent ``Session`` is kept between
  calls unless ``close_http_after_request`` is ``False``.
- Responses are streamed token-by-token and yielded to the caller so the UI
  can update incrementally without buffering the full reply.
- Image bytes are passed through and can be deleted by the caller immediately
  after :func:`ask` returns (the bytes are not stored anywhere in this module).
- The module itself carries no global state beyond optional configuration.
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Iterator

logger = logging.getLogger(__name__)


# ── Public interface ──────────────────────────────────────────────────────────

class AIAssistant:
    """Facade for talking to a vision-capable LLM backend.

    Parameters
    ----------
    config:
        The application :class:`~src.config.Config` object.
    npu_manager:
        An optional :class:`~src.npu_manager.NPUManager`.  Only used when
        ``backend == "npu"``.
    """

    def __init__(self, config, npu_manager=None) -> None:  # noqa: ANN001
        self._config = config
        self._npu_manager = npu_manager

    # ── Main entry point ──────────────────────────────────────────────────────

    def ask(
        self,
        prompt: str,
        *,
        screenshot_jpeg: bytes | None = None,
        attachment_image_jpegs: list[bytes] | None = None,
        attachment_texts: list[str] | None = None,
    ) -> Generator[str, None, None]:
        """Send a prompt (with optional images/text) and stream the reply.

        This is a **generator**: iterate over it to receive response tokens as
        they arrive from the model.  The caller should delete ``screenshot_jpeg``
        and any attachment bytes once this function returns to free memory.

        Parameters
        ----------
        prompt:
            The user's natural-language question or instruction.
        screenshot_jpeg:
            JPEG bytes of the current screen (optional).
        attachment_image_jpegs:
            List of JPEG bytes for user-uploaded images (optional).
        attachment_texts:
            List of text file contents to include in the context (optional).

        Yields
        ------
        str
            Incremental response tokens as they arrive.
        """
        backend = self._config.backend
        if backend == "ollama":
            yield from self._ask_ollama(
                prompt, screenshot_jpeg, attachment_image_jpegs, attachment_texts
            )
        elif backend == "openai":
            yield from self._ask_openai(
                prompt, screenshot_jpeg, attachment_image_jpegs, attachment_texts
            )
        elif backend == "npu":
            yield from self._ask_npu(prompt, screenshot_jpeg)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    # ── Ollama backend ────────────────────────────────────────────────────────

    def _ask_ollama(
        self,
        prompt: str,
        screenshot_jpeg: bytes | None,
        attachment_images: list[bytes] | None,
        attachment_texts: list[str] | None,
    ) -> Iterator[str]:
        import base64

        cfg = self._config.ollama
        base_url = cfg["base_url"].rstrip("/")
        model = cfg["model"]
        timeout = cfg.get("timeout", 120)
        stream = self._config.resources.get("stream_response", True)

        # Build the full prompt text (include any text attachments)
        full_prompt = _build_text_prompt(prompt, attachment_texts)

        # Collect images (screenshot + uploaded)
        images_b64: list[str] = []
        if screenshot_jpeg:
            images_b64.append(base64.b64encode(screenshot_jpeg).decode())
        for img in (attachment_images or []):
            images_b64.append(base64.b64encode(img).decode())

        payload: dict = {
            "model": model,
            "prompt": full_prompt,
            "stream": stream,
        }
        if images_b64:
            payload["images"] = images_b64

        # Lazy import of requests – no persistent session held
        try:
            import requests  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "requests is not installed.  Install it with: pip install requests"
            ) from exc

        url = f"{base_url}/api/generate"
        logger.debug("Sending request to Ollama at %s (model=%s)", url, model)

        close_after = self._config.resources.get("close_http_after_request", True)
        headers = {"Connection": "close"} if close_after else {}

        try:
            with requests.post(
                url,
                json=payload,
                stream=stream,
                timeout=timeout,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                if stream:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                else:
                    data = resp.json()
                    yield data.get("response", "")
        except Exception as exc:
            logger.error("Ollama request failed: %s", exc)
            raise

    # ── OpenAI-compatible backend ─────────────────────────────────────────────

    def _ask_openai(
        self,
        prompt: str,
        screenshot_jpeg: bytes | None,
        attachment_images: list[bytes] | None,
        attachment_texts: list[str] | None,
    ) -> Iterator[str]:
        import base64

        cfg = self._config.openai
        base_url = cfg["base_url"].rstrip("/")
        api_key = cfg.get("api_key", "")
        model = cfg["model"]
        timeout = cfg.get("timeout", 60)
        stream = self._config.resources.get("stream_response", True)

        full_prompt = _build_text_prompt(prompt, attachment_texts)

        # Build content list for the vision message
        content: list[dict] = [{"type": "text", "text": full_prompt}]

        def _img_block(jpeg_bytes: bytes) -> dict:
            b64 = base64.b64encode(jpeg_bytes).decode()
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"},
            }

        if screenshot_jpeg:
            content.append(_img_block(screenshot_jpeg))
        for img in (attachment_images or []):
            content.append(_img_block(img))

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        }

        try:
            import requests  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "requests is not installed.  Install it with: pip install requests"
            ) from exc

        url = f"{base_url}/chat/completions"
        headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"}
        if self._config.resources.get("close_http_after_request", True):
            headers["Connection"] = "close"

        logger.debug("Sending request to OpenAI-compatible API at %s", url)

        try:
            with requests.post(
                url,
                json=payload,
                headers=headers,
                stream=stream,
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                if stream:
                    for line in resp.iter_lines():
                        if not line or line == b"data: [DONE]":
                            continue
                        raw = line.decode("utf-8", errors="replace")
                        if raw.startswith("data: "):
                            raw = raw[6:]
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            yield delta
                else:
                    data = resp.json()
                    yield (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
        except Exception as exc:
            logger.error("OpenAI API request failed: %s", exc)
            raise

    # ── NPU backend ───────────────────────────────────────────────────────────

    def _ask_npu(
        self,
        prompt: str,
        screenshot_jpeg: bytes | None,
    ) -> Iterator[str]:
        """Run inference on the AMD NPU via ONNX Runtime.

        The model is loaded, queried, and **immediately unloaded** (unless
        ``resources.unload_model_after_inference`` is False) so NPU memory is
        reclaimed right away.
        """
        if self._npu_manager is None:
            raise RuntimeError(
                "NPU backend selected but no NPUManager was provided."
            )

        try:
            import numpy as np  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "numpy is required for NPU inference: pip install numpy"
            ) from exc

        # Build a simple text-only feed; vision preprocessing is model-specific.
        # This provides a baseline that callers can extend for their ONNX model.
        logger.info("Running NPU inference (model=%s)", self._config.npu.get("model_path"))

        # Encode prompt as a basic int64 token sequence (placeholder –
        # real models need their tokenizer here).
        token_ids = np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8).astype(
            np.int64
        )[np.newaxis, :]  # shape [1, seq_len]

        feeds = {"input_ids": token_ids}
        if screenshot_jpeg:
            # Pass image as raw bytes tensor if the model accepts it
            img_array = np.frombuffer(screenshot_jpeg, dtype=np.uint8)[np.newaxis, :]
            feeds["image"] = img_array

        # run_inference handles load → infer → unload in one call
        outputs = self._npu_manager.run_inference(feeds)

        # Decode the first output as UTF-8 text (model-specific)
        if outputs:
            result = outputs[0]
            if hasattr(result, "tobytes"):
                yield result.tobytes().decode("utf-8", errors="replace")
            else:
                yield str(result)
        else:
            yield ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_text_prompt(
    user_prompt: str,
    attachment_texts: list[str] | None,
) -> str:
    """Combine the user prompt with any text-file attachments."""
    parts: list[str] = []
    if attachment_texts:
        for i, text in enumerate(attachment_texts, start=1):
            parts.append(f"[Attached file {i}]\n{text}")
    parts.append(user_prompt)
    return "\n\n".join(parts)
