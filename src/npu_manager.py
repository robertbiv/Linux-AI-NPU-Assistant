"""AMD Ryzen AI / ONNX Runtime NPU manager.

This module probes for AMD NPU availability via ONNX Runtime's VitisAI
Execution Provider and exposes a thin inference wrapper.  When the NPU is
unavailable it falls back gracefully to CPU inference.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NPUSession:
    """Wraps an onnxruntime.InferenceSession configured for AMD NPU.

    Supports the context-manager protocol so resources are released as soon
    as the ``with`` block exits::

        with NPUSession(model_path, providers) as session:
            outputs = session.run(feeds)
        # session memory freed here

    Parameters
    ----------
    model_path:
        Path to a pre-compiled ONNX model.
    providers:
        Ordered list of ONNX Runtime Execution Providers to try.
    vitisai_config:
        Optional path to the VitisAI EP JSON configuration file.
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
        vitisai_config: str | Path | None = None,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed.  Install it with:\n"
                "  pip install onnxruntime   # CPU-only\n"
                "  pip install onnxruntime-vitisai  # AMD NPU"
            ) from exc

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self._model_path}")

        available = ort.get_available_providers()
        logger.debug("Available ONNX RT providers: %s", available)

        if providers is None:
            providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]

        # Filter to only providers that are actually available
        selected: list[Any] = []
        for p in providers:
            if p in available:
                if p == "VitisAIExecutionProvider" and vitisai_config:
                    cfg_path = str(vitisai_config)
                    if os.path.exists(cfg_path):
                        selected.append(
                            (p, {"config_file": cfg_path})
                        )
                    else:
                        logger.warning(
                            "VitisAI config not found at %s; skipping VitisAI EP.",
                            cfg_path,
                        )
                else:
                    selected.append(p)
            else:
                logger.debug("Provider %s not available; skipping.", p)

        if not selected:
            logger.warning(
                "None of the requested providers %s are available. "
                "Falling back to CPUExecutionProvider.",
                providers,
            )
            selected = ["CPUExecutionProvider"]

        logger.info("Creating ONNX session with providers: %s", selected)
        opts = ort.SessionOptions()
        self._session = ort.InferenceSession(
            str(self._model_path), sess_options=opts, providers=selected
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        logger.info(
            "ONNX model loaded: inputs=%s outputs=%s",
            self._input_names,
            self._output_names,
        )

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "NPUSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Explicitly release the ONNX session to free GPU/NPU memory."""
        self._session = None
        logger.debug("NPUSession closed; ONNX runtime memory released.")

    def run(self, feeds: dict[str, Any]) -> list[Any]:
        """Run inference.

        Parameters
        ----------
        feeds:
            Dict mapping input names to numpy arrays.

        Returns
        -------
        list
            Raw ONNX Runtime output tensors.

        Note
        ----
        The session may be ``None`` after :py:meth:`close` is called.  Always
        use this object inside a ``with`` block or check :py:attr:`is_open`.
        """
        if self._session is None:
            raise RuntimeError("NPUSession has been closed.")
        return self._session.run(self._output_names, feeds)

    @property
    def is_open(self) -> bool:
        return self._session is not None

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        return self._output_names


class NPUManager:
    """High-level manager for AMD NPU availability and session lifecycle."""

    def __init__(self, npu_config: dict, resource_config: dict | None = None) -> None:
        self._config = npu_config
        self._resource_config = resource_config or {}
        self._session: NPUSession | None = None
        self._available: bool | None = None

    # ── Availability ──────────────────────────────────────────────────────────

    def is_npu_available(self) -> bool:
        """Return *True* if the VitisAI Execution Provider is usable."""
        if self._available is not None:
            return self._available

        try:
            import onnxruntime as ort  # type: ignore[import]
            self._available = "VitisAIExecutionProvider" in ort.get_available_providers()
        except ImportError:
            self._available = False

        if self._available:
            logger.info("AMD NPU (VitisAI EP) is available.")
        else:
            logger.info("AMD NPU not available; will use software backend.")

        return self._available

    def get_device_info(self) -> dict[str, Any]:
        """Return human-readable information about the detected AI accelerator."""
        info: dict[str, Any] = {"npu_available": self.is_npu_available()}

        try:
            import onnxruntime as ort  # type: ignore[import]
            info["onnxruntime_version"] = ort.__version__
            info["providers"] = ort.get_available_providers()
        except ImportError:
            info["onnxruntime_version"] = "not installed"
            info["providers"] = []

        # Detect AMD GPU/APU via /sys
        amd_gpu_path = Path("/sys/class/drm")
        if amd_gpu_path.exists():
            amd_devices = [
                d.name
                for d in amd_gpu_path.iterdir()
                if (d / "device" / "vendor").exists()
                and (d / "device" / "vendor")
                .read_text(errors="replace")
                .strip()
                .lower()
                == "0x1002"  # AMD vendor ID
            ]
            info["amd_gpu_devices"] = amd_devices
        else:
            info["amd_gpu_devices"] = []

        return info

    # ── Session management ────────────────────────────────────────────────────

    def load_model(self) -> NPUSession | None:
        """Load the configured ONNX model onto the NPU (or CPU fallback).

        Returns *None* if no model path is configured.
        """
        model_path = self._config.get("model_path", "")
        if not model_path:
            logger.debug("No NPU model_path configured; skipping model load.")
            return None

        if self._session is None:
            self._session = NPUSession(
                model_path=model_path,
                providers=self._config.get(
                    "providers",
                    ["VitisAIExecutionProvider", "CPUExecutionProvider"],
                ),
                vitisai_config=self._config.get("vitisai_config"),
            )
        return self._session

    def run_inference(self, feeds: dict[str, Any]) -> list[Any]:
        """Load the model, run inference, and immediately unload if configured.

        When ``resources.unload_model_after_inference`` is ``True`` (default)
        the ONNX session is destroyed after the call so NPU/GPU memory is
        released straight away.
        """
        session = self.load_model()
        if session is None:
            raise RuntimeError("No NPU model_path configured; cannot run inference.")
        try:
            return session.run(feeds)
        finally:
            if self._resource_config.get("unload_model_after_inference", True):
                self.unload()

    def get_session(self) -> NPUSession | None:
        """Return the cached session, loading it if necessary."""
        if self._session is None:
            return self.load_model()
        return self._session

    def unload(self) -> None:
        """Release the ONNX session."""
        self._session = None
        logger.debug("NPU session unloaded.")
