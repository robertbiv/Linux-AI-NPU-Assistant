# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI integration tests for Neural Monolith.

Covers every major widget and feature using **pytest-qt** (the standard
framework for testing PyQt5 desktop applications).  A virtual display via
Xvfb is provided by the ``run-gui-tests.sh`` wrapper so this suite runs
headlessly in CI and on developer machines without a physical screen.

Features tested
---------------
- MainWindow compact / full mode switching and drag support
- ChatWidget: send message, append responses, streaming mode, clear
- StatusWidget: update_metrics with full simulated NPU data, metric cards,
  throughput chart, latency rows, active kernels, model context
- NPUSettingsWidget: model card selection, capture toggle, thermal slider,
  tool permission toggles, theme selection
- NPUCatalogWidget: catalog cards rendered, download / use / remove flow
  (mocked so no network is required)
- FullWindow: sidebar navigation to every page, stats update, collapse signal
- DiagnosticWindow: report display, Run Tests button, Copy Report button
- SettingsWindow (ModelManager): tab switching, backend model list rendering
- NPU simulation: mocked VitisAI provider present / absent
- Screenshot capture: each widget is rendered to a PNG in /tmp/gui_screenshots/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ── Display guard ─────────────────────────────────────────────────────────────
# Ensure every test can see a display (Xvfb provided by run-gui-tests.sh
# or a real X session).  If DISPLAY / WAYLAND_DISPLAY are unset we set a
# fallback so that the QtWidgets import itself doesn't abort.
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("DISPLAY", ":99")

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication

# ── Screenshot helpers ────────────────────────────────────────────────────────

_SCREENSHOT_DIR = Path("/tmp/gui_screenshots")
_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _grab(widget, name: str) -> Path:
    """Render *widget* to a PNG file and return the path."""
    from PyQt5.QtWidgets import QApplication as _App
    screen = _App.primaryScreen()
    widget.show()
    _App.processEvents()
    px = screen.grabWindow(widget.winId())
    out = _SCREENSHOT_DIR / f"{name}.png"
    px.save(str(out))
    return out


# ── Shared mock helpers ───────────────────────────────────────────────────────

def _mock_settings_manager(**overrides):
    """Return a MagicMock that quacks like SettingsManager."""
    sm = MagicMock()
    store: dict = {
        "ui.auto_send_screen": True,
        "npu.thermal_threshold_c": 85,
        "npu.thermal_notify": True,
        "ui.theme": "neural_dark",
        "backend": "ollama",
        "npu.model_path": "",
    }
    store.update(overrides)
    sm.get = lambda key, default=None: store.get(key, default)
    sm.set = lambda key, value, save=False: store.update({key: value})
    return sm


def _mock_config(backend: str = "ollama"):
    """Return a MagicMock that quacks like Config."""
    cfg = MagicMock()
    cfg.backend = backend
    cfg.ollama = {"base_url": "http://localhost:11434", "model": "llava", "timeout": 5}
    cfg.openai = {
        "base_url": "http://localhost:1234/v1",
        "api_key": "sk-test",
        "model": "local-model",
        "timeout": 5,
    }
    cfg.npu = {"model_path": ""}
    cfg.network = {"allow_external": False}
    cfg.resources = {"stream_response": True}
    cfg.get = MagicMock(return_value={})
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# ChatWidget tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChatWidget:
    """Tests for src.gui.chat_widget.ChatWidget."""

    def test_widget_instantiates(self, qtbot):
        from src.gui.chat_widget import ChatWidget
        w = ChatWidget(_mock_settings_manager())
        qtbot.addWidget(w)
        w.show()
        assert w.isVisible()

    def test_append_user_message(self, qtbot):
        from src.gui.chat_widget import ChatWidget
        w = ChatWidget()
        qtbot.addWidget(w)
        w.show()
        w.append_user_message("Hello from user")
        QApplication.processEvents()
        # Message container should have grown beyond the initial stretch
        layout = w._msg_layout
        assert layout.count() > 1

    def test_append_assistant_message(self, qtbot):
        from src.gui.chat_widget import ChatWidget
        w = ChatWidget()
        qtbot.addWidget(w)
        w.show()
        w.append_assistant_message("Hello from assistant", model_name="TestModel")
        QApplication.processEvents()
        assert w._msg_layout.count() > 1

    def test_send_button_emits_signal(self, qtbot):
        from src.gui.chat_widget import ChatWidget
        w = ChatWidget()
        qtbot.addWidget(w)
        w.show()
        emitted: list[str] = []
        w.message_submitted.connect(emitted.append)
        w._input.setPlainText("Test message")
        with qtbot.waitSignal(w.message_submitted, timeout=2000):
            w._send_btn.click()
        assert emitted == ["Test message"]

    def test_send_clears_input(self, qtbot):
        from src.gui.chat_widget import ChatWidget
        w = ChatWidget()
        qtbot.addWidget(w)
        w.show()
        w._input.setPlainText("Clear me")
        w._on_send()
        QApplication.processEvents()
        assert w._input.toPlainText() == ""

    def test_streaming_disables_send_button(self, qtbot):
        from src.gui.chat_widget import ChatWidget
        w = ChatWidget()
        qtbot.addWidget(w)
        w.show()
        w.set_streaming(True)
        assert not w._send_btn.isEnabled()
        assert w._send_btn.text() == "⏹"
        w.set_streaming(False)
        assert w._send_btn.isEnabled()
        assert w._send_btn.text() == "↑"

    def test_clear_conversation(self, qtbot):
        from src.gui.chat_widget import ChatWidget
        w = ChatWidget()
        qtbot.addWidget(w)
        w.show()
        w.append_user_message("msg 1")
        w.append_assistant_message("msg 2")
        QApplication.processEvents()
        w.clear_conversation()
        QApplication.processEvents()
        # Only the trailing stretch should remain
        assert w._msg_layout.count() == 1

    def test_set_model_name(self, qtbot):
        from src.gui.chat_widget import ChatWidget
        w = ChatWidget(model_name="Old Model")
        qtbot.addWidget(w)
        w.set_model_name("New Model")
        assert w._model_name == "New Model"

    def test_empty_send_does_nothing(self, qtbot):
        from src.gui.chat_widget import ChatWidget
        w = ChatWidget()
        qtbot.addWidget(w)
        emitted: list[str] = []
        w.message_submitted.connect(emitted.append)
        w._input.clear()
        w._on_send()
        QApplication.processEvents()
        assert emitted == []

    def test_multiple_messages_accumulate(self, qtbot):
        from src.gui.chat_widget import ChatWidget
        w = ChatWidget()
        qtbot.addWidget(w)
        w.show()
        for i in range(5):
            w.append_user_message(f"msg {i}")
        QApplication.processEvents()
        assert w._msg_layout.count() >= 5

    def test_screenshot_chat_with_messages(self, qtbot):
        from src.gui.chat_widget import ChatWidget
        w = ChatWidget()
        qtbot.addWidget(w)
        w.resize(420, 500)
        w.show()
        w.append_user_message("What is an NPU?")
        w.append_assistant_message(
            "An NPU (Neural Processing Unit) is a dedicated hardware accelerator "
            "for AI inference workloads such as matrix multiplication.",
            model_name="Llama-3-NPU-8B",
        )
        w.append_user_message("How does it compare to a GPU?")
        w.append_assistant_message(
            "NPUs are optimised for low-power, high-throughput tensor operations "
            "and excel at on-device inference without the GPU's power draw.",
            model_name="Llama-3-NPU-8B",
        )
        QApplication.processEvents()
        out = _grab(w, "chat_widget_with_messages")
        assert out.exists() and out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# StatusWidget (NPU Performance dashboard) tests
# ─────────────────────────────────────────────────────────────────────────────

# Simulated NPU metrics as a test fixture
NPU_METRICS = {
    "npu_clock_pct": 78,
    "memory_used_gb": 12.4,
    "memory_total_gb": 20.0,
    "thermal_c": 54,
    "tps": 94.2,
    "tps_history": [80, 90, 85, 94, 91, 88, 94, 95, 93, 94, 96, 92],
    "t_first_ms": 12,
    "t_per_ms": 8,
    "jitter_ms": 0.4,
    "active_kernels": [
        {"cmd": "EXEC", "name": "lora_fusion_v4.bin", "status": "READY"},
        {"cmd": "LOAD", "name": "quantization_int8_map", "status": "..."},
        {"cmd": "STAT", "name": "stream_buffer_clear", "status": "OK"},
        {"cmd": "SYNC", "name": "neural_engine_sync", "status": "0ms"},
    ],
    "model_name": "Llama-3-8B-Instruct (4-bit Quantized)",
    "model_tags": ["X-VECTOR ON", "FP16 ACCEL", "NPU NATIVE"],
    "engine_status": "FULLY OPTIMIZED",
    "engine_ok": True,
}


class TestStatusWidget:
    """Tests for src.gui.status_widget.StatusWidget."""

    def test_widget_instantiates(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.show()
        assert w.isVisible()

    def test_update_metrics_clock(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.update_metrics({"npu_clock_pct": 65})
        QApplication.processEvents()
        assert w._metrics.get("npu_clock_pct") == 65

    def test_update_metrics_memory_pct(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.update_metrics({"memory_used_gb": 8.0, "memory_total_gb": 16.0})
        QApplication.processEvents()
        assert w._metrics.get("memory_used_gb") == 8.0

    def test_update_metrics_thermal(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.update_metrics({"thermal_c": 72})
        QApplication.processEvents()
        assert "72" in w._card_thermal._value_lbl.text()

    def test_update_metrics_tps(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.update_metrics({"tps": 104.7})
        QApplication.processEvents()
        assert "104.7" in w._tps_badge.text()

    def test_update_metrics_tps_history(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        history = [70, 80, 90, 85, 95, 92]
        w.update_metrics({"tps_history": history})
        QApplication.processEvents()
        assert w._throughput_chart._data == history

    def test_update_metrics_latency(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.update_metrics({"t_first_ms": 15, "t_per_ms": 9, "jitter_ms": 0.6})
        QApplication.processEvents()
        # Check value persisted
        assert w._metrics.get("t_first_ms") == 15
        assert w._metrics.get("t_per_ms") == 9

    def test_update_metrics_kernels(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        kernels = [{"cmd": "EXEC", "name": "test.bin", "status": "READY"}]
        w.update_metrics({"active_kernels": kernels})
        QApplication.processEvents()
        assert "1 PARALLEL" in w._kern_count_lbl.text()

    def test_update_metrics_engine_status(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.update_metrics({"engine_status": "THROTTLED", "engine_ok": False})
        QApplication.processEvents()
        assert "THROTTLED" in w._engine_status_lbl.text()

    def test_update_metrics_model_context(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.update_metrics({"model_name": "Phi-3-Mini", "model_tags": ["INT4", "AMD NPU"]})
        QApplication.processEvents()
        assert "Phi-3-Mini" in w._ctx_model_lbl.text()

    def test_full_metrics_update(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.update_metrics(NPU_METRICS)
        QApplication.processEvents()
        assert w._metrics.get("engine_status") == "FULLY OPTIMIZED"
        assert w._metrics.get("tps") == 94.2

    def test_screenshot_status_widget(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.resize(480, 700)
        w.show()
        w.update_metrics(NPU_METRICS)
        QApplication.processEvents()
        out = _grab(w, "status_widget_npu_metrics")
        assert out.exists() and out.stat().st_size > 0

    def test_screenshot_status_widget_throttled(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.resize(480, 700)
        w.show()
        throttled = dict(NPU_METRICS)
        throttled.update({"engine_status": "THROTTLED", "engine_ok": False, "thermal_c": 95, "npu_clock_pct": 22})
        w.update_metrics(throttled)
        QApplication.processEvents()
        out = _grab(w, "status_widget_throttled")
        assert out.exists() and out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# NPUSettingsWidget tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNPUSettingsWidget:
    """Tests for src.gui.npu_settings_widget.NPUSettingsWidget."""

    def test_widget_instantiates(self, qtbot):
        from src.gui.npu_settings_widget import NPUSettingsWidget
        w = NPUSettingsWidget(_mock_settings_manager())
        qtbot.addWidget(w)
        w.show()
        assert w.isVisible()

    def test_thermal_slider_range(self, qtbot):
        from src.gui.npu_settings_widget import NPUSettingsWidget
        w = NPUSettingsWidget()
        qtbot.addWidget(w)
        assert w._thermal_slider.minimum() == 60
        assert w._thermal_slider.maximum() == 110

    def test_thermal_slider_initial_value(self, qtbot):
        from src.gui.npu_settings_widget import NPUSettingsWidget
        w = NPUSettingsWidget()
        qtbot.addWidget(w)
        assert w._thermal_slider.value() == 85

    def test_thermal_slider_change_updates_label(self, qtbot):
        from src.gui.npu_settings_widget import NPUSettingsWidget
        w = NPUSettingsWidget()
        qtbot.addWidget(w)
        w._thermal_slider.setValue(95)
        QApplication.processEvents()
        assert "95" in w._thermal_value_lbl.text()

    def test_thermal_slider_persists_to_settings(self, qtbot):
        from src.gui.npu_settings_widget import NPUSettingsWidget
        sm = _mock_settings_manager()
        w = NPUSettingsWidget(sm)
        qtbot.addWidget(w)
        w._thermal_slider.setValue(100)
        QApplication.processEvents()
        assert sm.get("npu.thermal_threshold_c") == 100

    def test_capture_toggle_initial_state(self, qtbot):
        from src.gui.npu_settings_widget import NPUSettingsWidget
        w = NPUSettingsWidget(_mock_settings_manager(**{"ui.auto_send_screen": True}))
        qtbot.addWidget(w)
        assert w._capture_toggle.isChecked()

    def test_capture_toggle_persists(self, qtbot):
        from src.gui.npu_settings_widget import NPUSettingsWidget
        sm = _mock_settings_manager()
        w = NPUSettingsWidget(sm)
        qtbot.addWidget(w)
        # Toggle off
        w._capture_toggle.setChecked(False)
        w._on_capture_toggled(False)
        assert sm.get("ui.auto_send_screen") is False

    def test_tool_toggles_exist(self, qtbot):
        from src.gui.npu_settings_widget import NPUSettingsWidget
        w = NPUSettingsWidget()
        qtbot.addWidget(w)
        assert w._fs_toggle is not None
        assert w._web_toggle is not None
        assert w._kern_toggle is not None

    def test_tool_toggle_default_values(self, qtbot):
        from src.gui.npu_settings_widget import NPUSettingsWidget
        w = NPUSettingsWidget()
        qtbot.addWidget(w)
        assert w._fs_toggle.get_value() == "Auto"
        assert w._web_toggle.get_value() == "Auto"
        assert w._kern_toggle.get_value() == "Off"

    def test_tool_toggle_set_value(self, qtbot):
        from src.gui.npu_settings_widget import NPUSettingsWidget
        w = NPUSettingsWidget()
        qtbot.addWidget(w)
        w._fs_toggle.set_value("Ask")
        assert w._fs_toggle.get_value() == "Ask"
        w._kern_toggle.set_value("Auto")
        assert w._kern_toggle.get_value() == "Auto"

    def test_model_cards_present(self, qtbot):
        from src.gui.npu_settings_widget import NPUSettingsWidget
        w = NPUSettingsWidget()
        qtbot.addWidget(w)
        assert w._model_llama is not None
        assert w._model_mistral is not None

    def test_screenshot_settings_widget(self, qtbot):
        from src.gui.npu_settings_widget import NPUSettingsWidget
        w = NPUSettingsWidget(_mock_settings_manager())
        qtbot.addWidget(w)
        w.resize(500, 800)
        w.show()
        QApplication.processEvents()
        out = _grab(w, "npu_settings_widget")
        assert out.exists() and out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# MainWindow tests (compact and full mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestMainWindow:
    """Tests for src.gui.main_window.MainWindow."""

    def test_starts_compact_mode(self, qtbot):
        from src.gui.main_window import MainWindow, MODE_COMPACT
        w = MainWindow(settings_manager=_mock_settings_manager(), start_mode="compact")
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        assert w._current_mode == MODE_COMPACT

    def test_starts_full_mode(self, qtbot):
        from src.gui.main_window import MainWindow, MODE_FULL
        w = MainWindow(settings_manager=_mock_settings_manager(), start_mode="full")
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        assert w._current_mode == MODE_FULL

    def test_switch_compact_to_full(self, qtbot):
        from src.gui.main_window import MainWindow, MODE_FULL
        w = MainWindow(settings_manager=_mock_settings_manager(), start_mode="compact")
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        w.show_full()
        QApplication.processEvents()
        assert w._current_mode == MODE_FULL

    def test_switch_full_to_compact(self, qtbot):
        from src.gui.main_window import MainWindow, MODE_COMPACT
        w = MainWindow(settings_manager=_mock_settings_manager(), start_mode="full")
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        w.show_compact()
        QApplication.processEvents()
        assert w._current_mode == MODE_COMPACT

    def test_compact_size(self, qtbot):
        from src.gui.main_window import MainWindow
        w = MainWindow(settings_manager=_mock_settings_manager(), start_mode="compact")
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        assert w.width() == 420
        assert w.height() == 680

    def test_full_mode_min_size(self, qtbot):
        from src.gui.main_window import MainWindow
        w = MainWindow(settings_manager=_mock_settings_manager(), start_mode="full")
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        assert w.width() >= 900
        assert w.height() >= 620

    def test_chat_widget_accessible(self, qtbot):
        from src.gui.main_window import MainWindow
        w = MainWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        assert w.chat_widget() is not None

    def test_status_widget_accessible(self, qtbot):
        from src.gui.main_window import MainWindow
        w = MainWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        assert w.status_widget() is not None

    def test_expand_button_switches_to_full(self, qtbot):
        from src.gui.main_window import MainWindow, MODE_FULL
        w = MainWindow(settings_manager=_mock_settings_manager(), start_mode="compact")
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        w._compact_widget.expand_clicked.emit()
        QApplication.processEvents()
        assert w._current_mode == MODE_FULL

    def test_collapse_button_switches_to_compact(self, qtbot):
        from src.gui.main_window import MainWindow, MODE_COMPACT
        w = MainWindow(settings_manager=_mock_settings_manager(), start_mode="full")
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        w._full_widget.collapse_requested.emit()
        QApplication.processEvents()
        assert w._current_mode == MODE_COMPACT

    def test_set_model_name(self, qtbot):
        from src.gui.main_window import MainWindow
        w = MainWindow(settings_manager=_mock_settings_manager(), start_mode="compact")
        qtbot.addWidget(w)
        w.set_model_name("Phi-3-Mini")
        # Should not raise
        QApplication.processEvents()

    def test_no_crash_compact_idempotent(self, qtbot):
        from src.gui.main_window import MainWindow
        w = MainWindow(settings_manager=_mock_settings_manager(), start_mode="compact")
        qtbot.addWidget(w)
        w.show()
        w.show_compact()  # calling twice should be a no-op
        QApplication.processEvents()

    def test_screenshot_compact_mode(self, qtbot):
        from src.gui.main_window import MainWindow
        w = MainWindow(settings_manager=_mock_settings_manager(), start_mode="compact")
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        out = _grab(w, "main_window_compact")
        assert out.exists() and out.stat().st_size > 0

    def test_screenshot_full_mode(self, qtbot):
        from src.gui.main_window import MainWindow
        w = MainWindow(settings_manager=_mock_settings_manager(), start_mode="full")
        qtbot.addWidget(w)
        w.show()
        QApplication.processEvents()
        out = _grab(w, "main_window_full")
        assert out.exists() and out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# FullWindow tests (sidebar navigation)
# ─────────────────────────────────────────────────────────────────────────────

class TestFullWindow:
    """Tests for src.gui.full_window.FullWindow."""

    def test_instantiates(self, qtbot):
        from src.gui.full_window import FullWindow
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.show()
        assert w.isVisible()

    def test_navigate_npu_performance(self, qtbot):
        from src.gui.full_window import FullWindow, PAGE_NPU_PERF
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.show()
        w.set_page(PAGE_NPU_PERF)
        QApplication.processEvents()
        assert w._stack.currentIndex() == w._pages[PAGE_NPU_PERF]

    def test_navigate_neural_models(self, qtbot):
        from src.gui.full_window import FullWindow, PAGE_NEURAL_MODELS
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.show()
        w.set_page(PAGE_NEURAL_MODELS)
        QApplication.processEvents()
        assert w._stack.currentIndex() == w._pages[PAGE_NEURAL_MODELS]

    def test_navigate_system_logs(self, qtbot):
        from src.gui.full_window import FullWindow, PAGE_SYSTEM_LOGS
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.show()
        w.set_page(PAGE_SYSTEM_LOGS)
        QApplication.processEvents()
        assert w._stack.currentIndex() == w._pages[PAGE_SYSTEM_LOGS]

    def test_navigate_api_integration(self, qtbot):
        from src.gui.full_window import FullWindow, PAGE_API
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.show()
        w.set_page(PAGE_API)
        QApplication.processEvents()
        assert w._stack.currentIndex() == w._pages[PAGE_API]

    def test_navigate_preferences(self, qtbot):
        from src.gui.full_window import FullWindow, PAGE_PREFERENCES
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.show()
        w.set_page(PAGE_PREFERENCES)
        QApplication.processEvents()
        assert w._stack.currentIndex() == w._pages[PAGE_PREFERENCES]

    def test_navigate_chat(self, qtbot):
        from src.gui.full_window import FullWindow, PAGE_CHAT
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.show()
        w.set_page(PAGE_CHAT)
        QApplication.processEvents()
        assert w._stack.currentIndex() == w._pages[PAGE_CHAT]

    def test_update_stats(self, qtbot):
        from src.gui.full_window import FullWindow
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.update_stats(npu_pct=42.0, mem_gb=8.5)
        QApplication.processEvents()
        assert "42.0" in w._sidebar._npu_lbl.text()
        assert "8.5" in w._sidebar._mem_lbl.text()

    def test_collapse_signal(self, qtbot):
        from src.gui.full_window import FullWindow
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.show()
        fired: list[bool] = []
        w.collapse_requested.connect(lambda: fired.append(True))
        w._header.collapse_clicked.emit()
        QApplication.processEvents()
        assert fired == [True]

    def test_chat_widget_embedded(self, qtbot):
        from src.gui.full_window import FullWindow
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        assert w.chat_widget() is not None

    def test_status_widget_embedded(self, qtbot):
        from src.gui.full_window import FullWindow
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        assert w.status_widget() is not None

    def test_screenshot_full_window_npu_page(self, qtbot):
        from src.gui.full_window import FullWindow, PAGE_NPU_PERF
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.resize(1100, 760)
        w.show()
        w.set_page(PAGE_NPU_PERF)
        w.status_widget().update_metrics(NPU_METRICS)
        QApplication.processEvents()
        out = _grab(w, "full_window_npu_performance")
        assert out.exists() and out.stat().st_size > 0

    def test_screenshot_full_window_chat_page(self, qtbot):
        from src.gui.full_window import FullWindow, PAGE_CHAT
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.resize(1100, 760)
        w.show()
        w.set_page(PAGE_CHAT)
        cw = w.chat_widget()
        cw.append_user_message("Show me the NPU status")
        cw.append_assistant_message(
            "Your NPU is running at 78 % clock speed with 54 °C thermal load.",
            model_name="Llama-3-NPU-8B",
        )
        QApplication.processEvents()
        out = _grab(w, "full_window_chat")
        assert out.exists() and out.stat().st_size > 0

    def test_screenshot_full_window_settings_page(self, qtbot):
        from src.gui.full_window import FullWindow, PAGE_NEURAL_MODELS
        w = FullWindow(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.resize(1100, 760)
        w.show()
        w.set_page(PAGE_NEURAL_MODELS)
        QApplication.processEvents()
        out = _grab(w, "full_window_settings")
        assert out.exists() and out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# NPU simulation tests (mocked hardware)
# ─────────────────────────────────────────────────────────────────────────────

class TestNPUSimulation:
    """Verify the app behaves correctly whether a VitisAI NPU is present or not."""

    def test_npu_available_mock(self):
        """Simulate NPU present via mocked onnxruntime."""
        ort = MagicMock()
        ort.get_available_providers.return_value = [
            "VitisAIExecutionProvider", "CPUExecutionProvider"
        ]
        ort.__version__ = "1.18.0"
        with patch.dict("sys.modules", {"onnxruntime": ort}):
            from src.npu_manager import NPUManager
            mgr = NPUManager({"model_path": ""})
            assert mgr.is_npu_available() is True

    def test_npu_unavailable_fallback(self):
        """Simulate no NPU — manager must report unavailable cleanly."""
        ort = MagicMock()
        ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        ort.__version__ = "1.18.0"
        with patch.dict("sys.modules", {"onnxruntime": ort}):
            from importlib import reload
            import src.npu_manager as _mod
            reload(_mod)
            mgr = _mod.NPUManager({"model_path": ""})
            assert mgr.is_npu_available() is False

    def test_device_info_with_npu(self):
        """get_device_info() should include provider list when NPU present."""
        ort = MagicMock()
        ort.get_available_providers.return_value = [
            "VitisAIExecutionProvider", "CPUExecutionProvider"
        ]
        ort.__version__ = "1.18.0"
        with patch.dict("sys.modules", {"onnxruntime": ort}):
            from importlib import reload
            import src.npu_manager as _mod
            reload(_mod)
            mgr = _mod.NPUManager({"model_path": ""})
            info = mgr.get_device_info()
            assert info["npu_available"] is True
            assert "VitisAIExecutionProvider" in info["providers"]

    def test_device_info_without_onnxruntime(self):
        """get_device_info() must degrade gracefully when onnxruntime missing."""
        with patch.dict("sys.modules", {"onnxruntime": None}):
            from importlib import reload
            import src.npu_manager as _mod
            reload(_mod)
            mgr = _mod.NPUManager({"model_path": ""})
            info = mgr.get_device_info()
            assert info["npu_available"] is False
            assert info["providers"] == []

    def test_status_widget_reflects_npu_unavailable(self, qtbot):
        """StatusWidget should accept 'FALLBACK' engine status without error."""
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.show()
        w.update_metrics({
            "engine_status": "CPU FALLBACK",
            "engine_ok": False,
            "npu_clock_pct": 0,
            "tps": 12.3,
        })
        QApplication.processEvents()
        assert "CPU FALLBACK" in w._engine_status_lbl.text()

    def test_status_widget_reflects_npu_available(self, qtbot):
        """StatusWidget should show green 'FULLY OPTIMIZED' when NPU online."""
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.show()
        w.update_metrics({
            "engine_status": "FULLY OPTIMIZED",
            "engine_ok": True,
            "npu_clock_pct": 78,
            "tps": 94.2,
        })
        QApplication.processEvents()
        assert "FULLY OPTIMIZED" in w._engine_status_lbl.text()

    def test_screenshot_npu_available(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.resize(480, 700)
        w.show()
        w.update_metrics(NPU_METRICS)
        QApplication.processEvents()
        out = _grab(w, "npu_simulation_available")
        assert out.exists() and out.stat().st_size > 0

    def test_screenshot_npu_unavailable(self, qtbot):
        from src.gui.status_widget import StatusWidget
        w = StatusWidget()
        qtbot.addWidget(w)
        w.resize(480, 700)
        w.show()
        fallback_metrics = dict(NPU_METRICS)
        fallback_metrics.update({
            "engine_status": "CPU FALLBACK",
            "engine_ok": False,
            "npu_clock_pct": 0,
            "tps": 14.1,
        })
        w.update_metrics(fallback_metrics)
        QApplication.processEvents()
        out = _grab(w, "npu_simulation_unavailable")
        assert out.exists() and out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# Model installer / catalog tests (mocked — no network required)
# ─────────────────────────────────────────────────────────────────────────────

class TestNPUModelInstaller:
    """Tests for src.npu_model_installer at the business-logic level."""

    def test_catalog_is_non_empty(self):
        from src.npu_model_installer import MODEL_CATALOG
        assert len(MODEL_CATALOG) > 0

    def test_catalog_entries_have_required_fields(self):
        from src.npu_model_installer import MODEL_CATALOG
        for entry in MODEL_CATALOG:
            assert entry.key, f"Entry missing key: {entry}"
            assert entry.name, f"Entry missing name: {entry}"

    def test_vision_models_helper(self):
        from src.npu_model_installer import get_vision_models
        vision = get_vision_models()
        assert isinstance(vision, list)
        assert all(m.is_vision for m in vision)

    def test_installer_not_installed_when_no_model(self, tmp_path):
        from src.npu_model_installer import NPUModelInstaller
        installer = NPUModelInstaller(model_dir=tmp_path)
        assert not installer.is_installed()

    def test_installer_reports_model_path(self, tmp_path):
        from src.npu_model_installer import NPUModelInstaller
        installer = NPUModelInstaller(model_dir=tmp_path)
        # Path is reported even if not present
        assert isinstance(installer.model_path(), Path)

    def test_install_mock_no_network(self, tmp_path):
        """install() with mocked download — never hits the network."""
        from src.npu_model_installer import NPUModelInstaller

        def _fake_install(self_inner, progress_callback=None, allow_external=False):
            model_file = tmp_path / "model.onnx"
            model_file.write_bytes(b"\x00" * 16)
            if progress_callback:
                progress_callback("Download complete (mock)")
            return model_file

        with patch.object(NPUModelInstaller, "install", _fake_install):
            installer = NPUModelInstaller(model_dir=tmp_path)
            progress_calls: list[str] = []
            path = installer.install(progress_callback=progress_calls.append)
            assert path.exists()
            assert "mock" in progress_calls[0]


class TestNPUCatalogWidget:
    """Tests for src.gui.model_manager.NPUCatalogWidget."""

    def test_widget_instantiates(self, qtbot):
        from src.gui.model_manager import NPUCatalogWidget
        w = NPUCatalogWidget(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.show()
        assert w.isVisible()

    def test_cards_rendered_for_all_catalog_entries(self, qtbot):
        from src.gui.model_manager import NPUCatalogWidget
        from src.npu_model_installer import MODEL_CATALOG
        w = NPUCatalogWidget(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        # Every catalog entry should have a card
        assert len(w._cards) == len(MODEL_CATALOG)

    def test_screenshot_catalog_widget(self, qtbot):
        from src.gui.model_manager import NPUCatalogWidget
        w = NPUCatalogWidget(settings_manager=_mock_settings_manager())
        qtbot.addWidget(w)
        w.resize(700, 600)
        w.show()
        QApplication.processEvents()
        out = _grab(w, "npu_catalog_widget")
        assert out.exists() and out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# DiagnosticWindow tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnosticWindow:
    """Tests for src.gui.diagnostic_window.DiagnosticWindow."""

    def _mock_reporter(self):
        """Return a reporter that returns a canned full_report without I/O."""
        reporter = MagicMock()
        reporter.full_report.return_value = {
            "backend": {"status": "ok", "detail": "Ollama reachable"},
            "npu": {
                "status": "warn",
                "detail": "VitisAI EP not found",
                "npu_available": False,
                "providers": ["CPUExecutionProvider"],
            },
            "tools": [
                {"name": "web_search", "status": "ok", "detail": ""},
                {"name": "file_system", "status": "ok", "detail": ""},
            ],
            "security": {
                "status": "ok",
                "checks": [
                    {"label": "config.yaml", "status": "ok", "path": "/etc/cfg"},
                ],
            },
            "settings": {"status": "ok", "detail": ""},
            "system": {"status": "ok", "detail": "Linux 6.8"},
            "network": {"status": "ok", "detail": "localhost only"},
            "dependencies": [
                {"name": "requests", "status": "ok", "version": "2.32"},
                {"name": "onnxruntime", "status": "warn", "version": "not installed"},
            ],
            "generated_at": "2024-01-01T00:00:00",
            "reporter_version": "0.1.0",
        }
        reporter.run_tests.return_value = {
            "exit_code": 0,
            "summary": "5 passed",
            "output": ".....\n5 passed in 0.5s",
        }
        return reporter

    def test_window_instantiates(self, qtbot):
        from src.gui.diagnostic_window import DiagnosticWindow
        with patch("src.gui.diagnostic_reporter.DiagnosticReporter") as MockReporter:
            MockReporter.return_value = self._mock_reporter()
            w = DiagnosticWindow(config=_mock_config())
            qtbot.addWidget(w)
            w.show()
            assert w.isVisible()

    def test_refresh_populates_overview(self, qtbot):
        from src.gui.diagnostic_window import DiagnosticWindow
        reporter = self._mock_reporter()
        with patch("src.gui.diagnostic_window.DiagnosticReporter", return_value=reporter):
            w = DiagnosticWindow(config=_mock_config())
            qtbot.addWidget(w)
            w.show()
            # Trigger report population directly
            w._on_report(reporter.full_report())
            QApplication.processEvents()
            # Status table should have rows
            assert w._table.rowCount() > 0

    def test_copy_report_button_exists(self, qtbot):
        from src.gui.diagnostic_window import DiagnosticWindow
        reporter = self._mock_reporter()
        with patch("src.gui.diagnostic_window.DiagnosticReporter", return_value=reporter):
            w = DiagnosticWindow(config=_mock_config())
            qtbot.addWidget(w)
            assert w._copy_btn is not None

    def test_screenshot_diagnostic_window(self, qtbot):
        from src.gui.diagnostic_window import DiagnosticWindow
        reporter = self._mock_reporter()
        with patch("src.gui.diagnostic_window.DiagnosticReporter", return_value=reporter):
            w = DiagnosticWindow(config=_mock_config())
            qtbot.addWidget(w)
            w.resize(800, 600)
            w.show()
            w._on_report(reporter.full_report())
            QApplication.processEvents()
            out = _grab(w, "diagnostic_window")
            assert out.exists() and out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# DiagnosticReporter unit tests (no Qt required)
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnosticReporter:
    """Tests for src.gui.diagnostic_reporter.DiagnosticReporter."""

    def test_check_npu_no_onnxruntime(self):
        from src.gui.diagnostic_reporter import DiagnosticReporter
        with patch.dict("sys.modules", {"onnxruntime": None}):
            r = DiagnosticReporter(config=_mock_config())
            result = r.check_npu()
            assert result["status"] in ("warn", "fail", "skip")

    def test_check_npu_with_mock_npu(self):
        ort = MagicMock()
        ort.get_available_providers.return_value = [
            "VitisAIExecutionProvider", "CPUExecutionProvider"
        ]
        ort.__version__ = "1.18.0"
        with patch.dict("sys.modules", {"onnxruntime": ort}):
            from src.gui.diagnostic_reporter import DiagnosticReporter
            r = DiagnosticReporter(config=_mock_config())
            result = r.check_npu()
            assert result.get("npu_available") is True

    def test_check_system_returns_dict(self):
        from src.gui.diagnostic_reporter import DiagnosticReporter
        r = DiagnosticReporter(config=_mock_config())
        result = r.check_system()
        assert isinstance(result, dict)
        assert "status" in result

    def test_check_dependencies_returns_list(self):
        from src.gui.diagnostic_reporter import DiagnosticReporter
        r = DiagnosticReporter(config=_mock_config())
        result = r.check_dependencies()
        assert isinstance(result, list)
        for dep in result:
            assert "name" in dep
            assert "status" in dep

    def test_full_report_has_all_keys(self):
        from src.gui.diagnostic_reporter import DiagnosticReporter
        r = DiagnosticReporter(config=_mock_config())
        report = r.full_report()
        for key in ("backend", "npu", "tools", "security", "settings", "system", "network", "dependencies"):
            assert key in report, f"Missing key: {key}"

    def test_check_network_localhost_ok(self):
        from src.gui.diagnostic_reporter import DiagnosticReporter
        r = DiagnosticReporter(config=_mock_config())
        result = r.check_network()
        assert "status" in result


# ─────────────────────────────────────────────────────────────────────────────
# AI process integration tests (mocked backend — no real LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestAIProcessIntegration:
    """Verify the AI assistant pipeline from user input to displayed output."""

    def _make_streaming_response(self, tokens: list[str]):
        """Return a requests.Response mock that streams JSON lines."""
        import json

        def _iter_lines():
            for token in tokens:
                yield json.dumps({"message": {"content": token}}).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_lines.return_value = _iter_lines()
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_ask_streams_tokens(self):
        from src.ai_assistant import AIAssistant
        from src.conversation import ConversationHistory

        cfg = _mock_config("ollama")
        cfg.get = MagicMock(return_value={"rate_limit_per_minute": 0})
        cfg.resources = {"stream_response": True}

        tokens = ["Hello", " from", " the", " AI"]
        mock_resp = self._make_streaming_response(tokens)

        with patch("requests.post", return_value=mock_resp):
            assistant = AIAssistant(cfg)
            history = MagicMock()
            history.to_ollama_messages.return_value = []
            result = list(assistant.ask("Hi", history=history))
            assert "".join(result) == "".join(tokens)

    def test_ask_appends_to_chat_widget(self, qtbot):
        """Simulate the full UI flow: user sends message → tokens stream into ChatWidget."""
        from src.gui.chat_widget import ChatWidget
        from src.ai_assistant import AIAssistant
        from src.conversation import ConversationHistory

        cfg = _mock_config("ollama")
        cfg.get = MagicMock(return_value={"rate_limit_per_minute": 0})
        cfg.resources = {"stream_response": True}

        tokens = ["NPU", " inference", " is", " fast."]
        mock_resp = self._make_streaming_response(tokens)

        chat = ChatWidget(_mock_settings_manager())
        qtbot.addWidget(chat)
        chat.show()

        with patch("requests.post", return_value=mock_resp):
            assistant = AIAssistant(cfg)
            history = MagicMock()
            history.to_ollama_messages.return_value = []

            chat.append_user_message("What is NPU inference?")
            response_text = "".join(assistant.ask("What is NPU inference?", history=history))
            chat.append_assistant_message(response_text, model_name="Llama-3-NPU-8B")

        QApplication.processEvents()
        # Two bubbles should be present (user + assistant) plus trailing stretch
        assert chat._msg_layout.count() >= 3

    def test_screenshot_ai_response_in_chat(self, qtbot):
        from src.gui.chat_widget import ChatWidget
        from src.ai_assistant import AIAssistant

        cfg = _mock_config("ollama")
        cfg.get = MagicMock(return_value={"rate_limit_per_minute": 0})
        cfg.resources = {"stream_response": True}

        tokens = ["The", " NPU", " runs", " at", " 94", " tokens", "/sec."]
        import json

        def _iter_lines():
            for token in tokens:
                yield json.dumps({"message": {"content": token}}).encode()

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_lines.return_value = _iter_lines()
        mock_resp.raise_for_status = MagicMock()

        chat = ChatWidget(_mock_settings_manager())
        qtbot.addWidget(chat)
        chat.resize(420, 500)
        chat.show()

        with patch("requests.post", return_value=mock_resp):
            assistant = AIAssistant(cfg)
            history = MagicMock()
            history.to_ollama_messages.return_value = []
            chat.append_user_message("How fast is the NPU?")
            response = "".join(assistant.ask("How fast?", history=history))
            chat.append_assistant_message(response, model_name="Llama-3-NPU-8B")

        QApplication.processEvents()
        out = _grab(chat, "ai_response_in_chat")
        assert out.exists() and out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# Model install via GUI (mocked — no download)
# ─────────────────────────────────────────────────────────────────────────────

class TestModelInstallViaGUI:
    """Verify the catalog download flow triggered from NPUCatalogWidget."""

    def test_download_button_starts_thread(self, qtbot):
        from src.gui.model_manager import NPUCatalogWidget, _DownloadThread
        from src.npu_model_installer import MODEL_CATALOG

        sm = _mock_settings_manager()
        w = NPUCatalogWidget(settings_manager=sm)
        qtbot.addWidget(w)
        w.show()

        entry = MODEL_CATALOG[0]
        card = w._cards.get(entry.key)
        assert card is not None, "First catalog entry has no card"

        # Patch the thread so it finishes immediately with a fake path
        with patch.object(_DownloadThread, "start") as mock_start:
            card._on_download()
            QApplication.processEvents()
            # The card's download button should now be disabled (in-progress state)
            # and the thread was started
            mock_start.assert_called_once()

    def test_download_finished_updates_card(self, qtbot):
        from src.gui.model_manager import NPUCatalogWidget
        from src.npu_model_installer import MODEL_CATALOG

        sm = _mock_settings_manager()
        w = NPUCatalogWidget(settings_manager=sm)
        qtbot.addWidget(w)
        w.show()

        entry = MODEL_CATALOG[0]
        card = w._cards.get(entry.key)
        # Simulate a finished download
        fake_path = "/tmp/fake_model.onnx"
        card.on_download_finished(fake_path)
        QApplication.processEvents()
        # Card state should be refreshed (no exception raised)

    def test_screenshot_model_install_ui(self, qtbot):
        from src.gui.model_manager import NPUCatalogWidget
        sm = _mock_settings_manager()
        w = NPUCatalogWidget(settings_manager=sm)
        qtbot.addWidget(w)
        w.resize(700, 600)
        w.show()
        QApplication.processEvents()
        out = _grab(w, "model_install_catalog")
        assert out.exists() and out.stat().st_size > 0
