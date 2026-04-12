"""Model manager widget — browse, drag-and-drop, and delete AI models.

Embedded in the **Models** tab of :class:`~src.gui.settings_window.SettingsWindow`.
Can also be used as a standalone dialog.

Features
--------
- Live list of models fetched from the active backend (Ollama / OpenAI-compat)
- Per-model NPU compatibility badge (✅ OK / ⚠ Warn / ⛔ No)
- **Browse ONNX…** — opens a file dialog filtered to ``*.onnx``
- **Drag-and-drop** — drop ``.onnx`` or ``.gguf`` files from any file manager
- **Set as current model** — updates ``settings.json`` immediately
- **Delete** — removes Ollama model (``ollama rm``) or deregisters ONNX path;
  optionally deletes the ONNX file from disk
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QColor, QDropEvent
    from PyQt5.QtWidgets import (
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    _HAS_QT = True
except ImportError:
    _HAS_QT = False
    logger.warning("PyQt5 not installed — ModelManagerWidget unavailable.")

if _HAS_QT:

    # Colours for NPU compatibility badges
    _BADGE_OK   = "#27ae60"
    _BADGE_WARN = "#e67e22"
    _BADGE_FAIL = "#c0392b"
    _BADGE_SKIP = "#7f8c8d"

    class _FetchThread(QThread):
        """Background thread that calls ModelSelector.list_models()."""
        finished = pyqtSignal(list)
        error    = pyqtSignal(str)

        def __init__(self, selector: Any, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._selector = selector

        def run(self) -> None:
            try:
                models = self._selector.list_models(timeout=5)
                self.finished.emit(models)
            except Exception as exc:  # noqa: BLE001
                self.error.emit(str(exc))

    class ModelManagerWidget(QWidget):
        """Model browser widget with drag-and-drop and delete support.

        Parameters
        ----------
        manager:
            The application :class:`~src.settings.SettingsManager`.
        parent:
            Optional parent widget.
        """

        def __init__(self, manager: Any, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._manager   = manager
            self._selector  = None
            self._models: list[Any] = []

            self._build_ui()
            self._build_selector()
            self.refresh()

        def _build_selector(self) -> None:
            try:
                from src.model_selector import ModelSelector
                cfg = self._manager.to_config()
                self._selector = ModelSelector(cfg)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not build ModelSelector: %s", exc)

        def _build_ui(self) -> None:
            layout = QVBoxLayout(self)

            # Model list
            self._list = QListWidget()
            self._list.setAlternatingRowColors(True)
            self._list.setSelectionMode(QListWidget.SingleSelection)
            self._list.currentItemChanged.connect(self._on_selection_changed)
            self._list.setAcceptDrops(True)
            self._list.dragEnterEvent = self._drag_enter
            self._list.dragMoveEvent  = self._drag_move
            self._list.dropEvent      = self._drop_event
            layout.addWidget(self._list)

            # Drop hint label
            hint = QLabel("💡 Drop .onnx or .gguf files here to add a model")
            hint.setAlignment(Qt.AlignCenter)
            hint.setStyleSheet("color: grey; font-size: 11px;")
            layout.addWidget(hint)

            # Buttons row
            btn_row = QHBoxLayout()

            self._btn_refresh = QPushButton("🔄 Refresh")
            self._btn_refresh.setToolTip("Fetch the model list from the backend")
            self._btn_refresh.clicked.connect(self.refresh)
            btn_row.addWidget(self._btn_refresh)

            self._btn_browse = QPushButton("📂 Browse ONNX…")
            self._btn_browse.setToolTip("Open a file dialog to add an ONNX model file")
            self._btn_browse.clicked.connect(self._browse_onnx)
            btn_row.addWidget(self._btn_browse)

            self._btn_use = QPushButton("✔ Use this model")
            self._btn_use.setEnabled(False)
            self._btn_use.setToolTip("Set the selected model as the current model")
            self._btn_use.clicked.connect(self._use_model)
            btn_row.addWidget(self._btn_use)

            self._btn_delete = QPushButton("🗑 Delete")
            self._btn_delete.setEnabled(False)
            self._btn_delete.setToolTip("Remove the selected model from Ollama or deregister it")
            self._btn_delete.clicked.connect(self._delete_model)
            btn_row.addWidget(self._btn_delete)

            layout.addLayout(btn_row)

            # Status label
            self._status = QLabel("")
            self._status.setAlignment(Qt.AlignLeft)
            layout.addWidget(self._status)

        # ── List population ────────────────────────────────────────────────

        def refresh(self) -> None:
            """Fetch models from the backend in a background thread."""
            if self._selector is None:
                self._build_selector()
            if self._selector is None:
                self._set_status("⚠ Backend not configured.", error=True)
                return

            self._btn_refresh.setEnabled(False)
            self._set_status("Fetching models…")
            self._thread = _FetchThread(self._selector, parent=self)
            self._thread.finished.connect(self._on_models_fetched)
            self._thread.error.connect(self._on_fetch_error)
            self._thread.start()

        def _on_models_fetched(self, models: list) -> None:
            self._models = models
            self._list.clear()
            for m in models:
                self._add_list_item(m)
            self._btn_refresh.setEnabled(True)
            self._set_status(f"{len(models)} model(s) available.")

        def _on_fetch_error(self, msg: str) -> None:
            self._btn_refresh.setEnabled(True)
            self._set_status(f"⚠ Could not fetch models: {msg}", error=True)

        def _add_list_item(self, model: Any) -> None:
            """Add a model to the list widget with NPU badge."""
            if self._selector is not None:
                warning = self._selector.npu_warning(model)
            else:
                warning = None

            if warning is None:
                badge = "✅"
                colour = _BADGE_OK
            elif "⛔" in warning:
                badge = "⛔"
                colour = _BADGE_FAIL
            else:
                badge = "⚠"
                colour = _BADGE_WARN

            size_str = f"  {model.size_gb:.1f} GB" if model.size_gb else ""
            label    = f"{badge} {model.name}{size_str}"
            item     = QListWidgetItem(label)
            item.setData(Qt.UserRole, model)
            item.setToolTip(warning or "NPU compatible")
            item.setForeground(QColor(colour))
            self._list.addItem(item)

        # ── Selection ──────────────────────────────────────────────────────

        def _on_selection_changed(self, current: QListWidgetItem | None, _: Any) -> None:
            has_sel = current is not None
            self._btn_use.setEnabled(has_sel)
            self._btn_delete.setEnabled(has_sel)

        def _selected_model(self) -> Any | None:
            item = self._list.currentItem()
            return item.data(Qt.UserRole) if item else None

        # ── Use model ──────────────────────────────────────────────────────

        def _use_model(self) -> None:
            m = self._selected_model()
            if m is None:
                return
            if self._selector is not None:
                self._selector.set_model(m.name)
            # Also write through settings manager for persistence
            backend = self._manager.get("backend", "ollama")
            if backend == "ollama":
                self._manager.set("ollama.model", m.name)
            elif backend == "openai":
                self._manager.set("openai.model", m.name)
            elif backend == "npu":
                self._manager.set("npu.model_path", m.name)
            self._set_status(f"✔ Now using: {m.name}")
            logger.info("Model set to %r (backend=%r)", m.name, backend)

        # ── Browse ONNX ────────────────────────────────────────────────────

        def _browse_onnx(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select ONNX model file",
                str(Path.home()),
                "ONNX models (*.onnx);;All files (*)",
            )
            if path:
                self._register_file(path)

        def _register_file(self, path: str) -> None:
            """Add a file-path model to the list and select it."""
            from src.model_selector import ModelInfo
            m = ModelInfo(name=path)
            self._models.append(m)
            self._add_list_item(m)
            # Select the new item
            self._list.setCurrentRow(self._list.count() - 1)
            self._set_status(f"Added: {path}")

        # ── Drag-and-drop ──────────────────────────────────────────────────

        def _drag_enter(self, event: Any) -> None:
            if event.mimeData().hasUrls():
                paths = [u.toLocalFile() for u in event.mimeData().urls()]
                if any(p.endswith((".onnx", ".gguf")) for p in paths):
                    event.acceptProposedAction()
                    return
            event.ignore()

        def _drag_move(self, event: Any) -> None:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
            else:
                event.ignore()

        def _drop_event(self, event: Any) -> None:
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.endswith((".onnx", ".gguf")):
                    self._register_file(path)
            event.acceptProposedAction()

        # ── Delete model ───────────────────────────────────────────────────

        def _delete_model(self) -> None:
            m = self._selected_model()
            if m is None:
                return

            name    = m.name
            is_file = name.endswith((".onnx", ".gguf")) and Path(name).exists()

            if is_file:
                reply = QMessageBox.question(
                    self,
                    "Delete model file",
                    f"Remove <b>{name}</b> from the list?<br><br>"
                    "Also delete the file from disk?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.No,
                )
                if reply == QMessageBox.Cancel:
                    return
                if reply == QMessageBox.Yes:
                    try:
                        Path(name).unlink()
                        self._set_status(f"🗑 Deleted file: {name}")
                    except OSError as exc:
                        QMessageBox.critical(self, "Error", f"Could not delete file:\n{exc}")
                        return
                # Remove from list regardless
                self._remove_selected_item()
            else:
                # Ollama model
                reply = QMessageBox.question(
                    self,
                    "Remove model",
                    f"Remove Ollama model <b>{name}</b>?<br>"
                    "This will run <code>ollama rm {name}</code>.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
                try:
                    result = subprocess.run(
                        ["ollama", "rm", name],
                        capture_output=True, text=True, timeout=30,
                    )
                    if result.returncode == 0:
                        self._set_status(f"🗑 Removed model: {name}")
                        self._remove_selected_item()
                    else:
                        QMessageBox.critical(
                            self, "Error",
                            f"ollama rm failed:\n{result.stderr.strip()}"
                        )
                except FileNotFoundError:
                    QMessageBox.critical(
                        self, "Error",
                        "ollama command not found. Is Ollama installed and on PATH?"
                    )
                except subprocess.TimeoutExpired:
                    QMessageBox.critical(self, "Error", "ollama rm timed out.")

        def _remove_selected_item(self) -> None:
            row = self._list.currentRow()
            if row >= 0:
                self._list.takeItem(row)
                if row < len(self._models):
                    self._models.pop(row)

        # ── Status ─────────────────────────────────────────────────────────

        def _set_status(self, msg: str, error: bool = False) -> None:
            colour = "#c0392b" if error else "#27ae60"
            self._status.setStyleSheet(f"color: {colour};")
            self._status.setText(msg)
