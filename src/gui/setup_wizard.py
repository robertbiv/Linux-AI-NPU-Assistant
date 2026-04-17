# SPDX-License-Identifier: GPL-3.0-or-later
"""First-boot setup wizard for Neural Monolith.

Shown automatically on the very first launch (when no ``settings.json``
exists yet) so users can choose their AI backend without ever having to
open Settings manually.

Three options are presented:

* **NPU — Self-contained** — Use the built-in AMD NPU via ONNX Runtime.
  No Ollama or external service required.  This is the default choice and
  the recommended option for most Flatpak users.

* **Ollama — Connect to existing server** — Forward all requests to an
  Ollama server already running on the host machine.  The server URL can
  be customised (defaults to ``http://localhost:11434``).

* **Both — NPU + Ollama (hybrid)** — Uses the ``ollama+npu`` backend.
  Requests for ``.onnx`` models go to the NPU; everything else goes to
  Ollama.  Lets users keep their existing Ollama models while also trying
  NPU-accelerated inference.

The choice is persisted immediately via :class:`~src.settings.SettingsManager`
so the wizard is not shown again on subsequent launches.

Public API
----------
* :func:`needs_first_boot_setup` — returns ``True`` when the wizard should run.
* :func:`run_first_boot_wizard` — shows the dialog and saves the selection.
* :class:`FirstBootWizard` — the ``QDialog`` subclass (testable without Qt-show).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Backend identifier constants (mirror src/config.py)
BACKEND_NPU        = "npu"
BACKEND_OLLAMA     = "ollama"
BACKEND_OLLAMA_NPU = "ollama+npu"

DEFAULT_OLLAMA_URL = "http://localhost:11434"

try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtWidgets import (
        QButtonGroup,
        QDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QRadioButton,
        QVBoxLayout,
        QWidget,
    )
    _HAS_QT = True
except ImportError:
    _HAS_QT = False
    logger.warning("PyQt5 not installed — FirstBootWizard unavailable.")


# ── First-boot detection ──────────────────────────────────────────────────────

def needs_first_boot_setup(settings_path: "str | Path") -> bool:
    """Return ``True`` if the first-boot setup wizard should be shown.

    The wizard runs only once — on the very first launch when no
    ``settings.json`` file exists yet.  Users who already have a settings
    file (upgrades, or manual configuration) are unaffected.

    Parameters
    ----------
    settings_path:
        Path to the application's ``settings.json`` file.
    """
    return not Path(settings_path).exists()


# ── Qt classes ────────────────────────────────────────────────────────────────

if _HAS_QT:
    from src.gui import npu_theme as T

    class _OptionCard(QFrame):
        """Clickable card for selecting a backend option.

        Wraps a ``QRadioButton``, an emoji icon, and two lines of text inside
        a styled ``QFrame``.  Clicking anywhere on the card selects it.
        """

        selected = pyqtSignal()

        def __init__(
            self,
            icon: str,
            title: str,
            description: str,
            parent: "QWidget | None" = None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName("optionCard")
            self._is_selected = False
            self._normal_style = (
                f"QFrame#optionCard {{"
                f"  background-color: {T.BG_CARD};"
                f"  border: 1px solid {T.BORDER};"
                f"  border-radius: 12px;"
                f"}}"
            )
            self._selected_style = (
                f"QFrame#optionCard {{"
                f"  background-color: {T.BLUE_DIM};"
                f"  border: 2px solid {T.BLUE};"
                f"  border-radius: 12px;"
                f"}}"
            )
            self.setStyleSheet(self._normal_style)
            self.setCursor(Qt.PointingHandCursor)

            layout = QHBoxLayout(self)
            layout.setContentsMargins(16, 14, 16, 14)
            layout.setSpacing(14)

            # Radio button
            self._radio = QRadioButton()
            self._radio.setStyleSheet(
                f"QRadioButton::indicator {{"
                f"  width: 18px; height: 18px;"
                f"  border: 2px solid {T.BORDER};"
                f"  border-radius: 9px;"
                f"  background: {T.BG_INPUT};"
                f"}}"
                f"QRadioButton::indicator:checked {{"
                f"  background: {T.BLUE};"
                f"  border-color: {T.BLUE};"
                f"}}"
            )
            self._radio.toggled.connect(self._on_radio_toggled)
            layout.addWidget(self._radio)

            # Emoji icon
            icon_lbl = QLabel(icon)
            icon_lbl.setFixedSize(40, 40)
            icon_lbl.setAlignment(Qt.AlignCenter)
            icon_lbl.setStyleSheet(
                f"background: {T.BG_CARD2}; border: 1px solid {T.BORDER};"
                f"border-radius: 8px; font-size: 20px;"
            )
            layout.addWidget(icon_lbl)

            # Title + description
            text_col = QVBoxLayout()
            text_col.setSpacing(3)

            title_lbl = QLabel(title)
            title_lbl.setStyleSheet(
                f"color: {T.TEXT_PRIMARY}; font-size: 14px;"
                f"font-weight: bold; background: transparent;"
            )
            text_col.addWidget(title_lbl)

            desc_lbl = QLabel(description)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet(
                f"color: {T.TEXT_SECONDARY}; font-size: 12px; background: transparent;"
            )
            text_col.addWidget(desc_lbl)

            layout.addLayout(text_col, stretch=1)

        # ── Helpers ───────────────────────────────────────────────────────────

        def _on_radio_toggled(self, checked: bool) -> None:
            self._is_selected = checked
            self.setStyleSheet(
                self._selected_style if checked else self._normal_style
            )
            if checked:
                self.selected.emit()

        def mousePressEvent(self, event: "QMouseEvent") -> None:  # noqa: ANN001
            self._radio.setChecked(True)
            super().mousePressEvent(event)

        # ── Public API ────────────────────────────────────────────────────────

        @property
        def radio(self) -> QRadioButton:
            """The underlying ``QRadioButton`` (used for exclusive groups)."""
            return self._radio

        def set_checked(self, checked: bool) -> None:
            """Check or uncheck this card programmatically."""
            self._radio.setChecked(checked)

        def is_checked(self) -> bool:
            """Return whether this card is currently selected."""
            return self._radio.isChecked()

    # ─────────────────────────────────────────────────────────────────────────

    class FirstBootWizard(QDialog):
        """First-boot setup wizard dialog.

        Presents three backend options as clickable cards.  The dialog is
        modal; clicking **Get Started** (or closing the window) persists the
        selection to :class:`~src.settings.SettingsManager`.

        Attributes
        ----------
        chosen_backend:
            The backend identifier selected by the user (``"npu"``,
            ``"ollama"``, or ``"ollama+npu"``).  Defaults to ``"npu"``.
        chosen_ollama_url:
            The Ollama server URL entered by the user.  Relevant only when
            ``chosen_backend`` is ``"ollama"`` or ``"ollama+npu"``.
        """

        def __init__(self, parent: "QWidget | None" = None) -> None:
            super().__init__(parent)
            self.setWindowTitle("Welcome to Neural Monolith")
            self.setModal(True)
            self.setFixedWidth(520)
            self.setStyleSheet(T.STYLESHEET)

            self._chosen_backend: str = BACKEND_NPU
            self._chosen_ollama_url: str = DEFAULT_OLLAMA_URL

            self._setup_ui()

        # ── Public API ────────────────────────────────────────────────────────

        @property
        def chosen_backend(self) -> str:
            """Backend selected by the user."""
            return self._chosen_backend

        @property
        def chosen_ollama_url(self) -> str:
            """Ollama server URL entered by the user."""
            return self._chosen_ollama_url or DEFAULT_OLLAMA_URL

        # ── UI construction ───────────────────────────────────────────────────

        def _setup_ui(self) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(28, 28, 28, 24)
            root.setSpacing(16)

            # ── Header ────────────────────────────────────────────────────────
            title = QLabel("✦  Welcome to Neural Monolith")
            title.setStyleSheet(
                f"color: {T.GREEN}; font-size: 20px; font-weight: bold;"
                f"background: transparent;"
            )
            title.setAlignment(Qt.AlignCenter)
            root.addWidget(title)

            subtitle = QLabel("Choose how you'd like to run AI inference")
            subtitle.setStyleSheet(
                f"color: {T.TEXT_SECONDARY}; font-size: 13px; background: transparent;"
            )
            subtitle.setAlignment(Qt.AlignCenter)
            root.addWidget(subtitle)

            # ── Option cards ──────────────────────────────────────────────────

            self._radio_group = QButtonGroup(self)
            self._radio_group.setExclusive(True)

            # Card 1 — NPU (self-contained, default)
            self._card_npu = _OptionCard(
                "🧠",
                "NPU — Self-contained",
                "Use the built-in AMD NPU. No Ollama or internet required. "
                "Download ONNX models from Settings → Models.",
            )
            self._radio_group.addButton(self._card_npu.radio, 0)
            self._card_npu.selected.connect(
                lambda: self._on_selection(BACKEND_NPU)
            )
            root.addWidget(self._card_npu)

            # Card 2 — Ollama
            self._card_ollama = _OptionCard(
                "🦙",
                "Ollama — Connect to existing server",
                "Use models already installed in Ollama (GPU, CPU, ROCm…). "
                "Requires Ollama to be running on this machine.",
            )
            self._radio_group.addButton(self._card_ollama.radio, 1)
            self._card_ollama.selected.connect(
                lambda: self._on_selection(BACKEND_OLLAMA)
            )
            root.addWidget(self._card_ollama)

            # URL field — shown when Ollama or Both is selected
            self._url_row = QWidget()
            url_layout = QHBoxLayout(self._url_row)
            url_layout.setContentsMargins(58, 0, 12, 0)
            url_layout.setSpacing(8)

            url_lbl = QLabel("Server URL:")
            url_lbl.setFixedWidth(90)
            url_lbl.setStyleSheet(
                f"color: {T.TEXT_SECONDARY}; font-size: 12px; background: transparent;"
            )
            url_layout.addWidget(url_lbl)

            self._url_edit = QLineEdit(DEFAULT_OLLAMA_URL)
            self._url_edit.setPlaceholderText("http://localhost:11434")
            self._url_edit.textChanged.connect(self._on_url_changed)
            url_layout.addWidget(self._url_edit)

            self._url_row.setVisible(False)
            root.addWidget(self._url_row)

            # Card 3 — Both (hybrid)
            self._card_both = _OptionCard(
                "⚡",
                "Both — NPU + Ollama (hybrid)",
                "NPU for downloaded ONNX models, Ollama for GPU/CPU models. "
                "Switch between them anytime from the model picker.",
            )
            self._radio_group.addButton(self._card_both.radio, 2)
            self._card_both.selected.connect(
                lambda: self._on_selection(BACKEND_OLLAMA_NPU)
            )
            root.addWidget(self._card_both)

            # ── Footer note ───────────────────────────────────────────────────
            note = QLabel(
                "You can change this at any time in  Settings → AI Backend."
            )
            note.setStyleSheet(
                f"color: {T.TEXT_MUTED}; font-size: 11px; background: transparent;"
            )
            note.setAlignment(Qt.AlignCenter)
            root.addWidget(note)

            # ── Get Started button ────────────────────────────────────────────
            btn_row = QHBoxLayout()
            btn_row.setContentsMargins(0, 4, 0, 0)
            btn_row.addStretch()

            go_btn = QPushButton("Get Started  →")
            go_btn.setObjectName("sendBtn")
            go_btn.setFixedHeight(40)
            go_btn.setMinimumWidth(160)
            go_btn.clicked.connect(self.accept)
            btn_row.addWidget(go_btn)

            btn_row.addStretch()
            root.addLayout(btn_row)

            # Select NPU by default
            self._card_npu.set_checked(True)

        # ── Private slots ─────────────────────────────────────────────────────

        def _on_selection(self, backend: str) -> None:
            self._chosen_backend = backend
            needs_url = backend in (BACKEND_OLLAMA, BACKEND_OLLAMA_NPU)
            self._url_row.setVisible(needs_url)
            self.adjustSize()

        def _on_url_changed(self, text: str) -> None:
            self._chosen_ollama_url = text.strip() or DEFAULT_OLLAMA_URL


# ── Public convenience function ───────────────────────────────────────────────

def run_first_boot_wizard(app: Any, settings_manager: Any) -> None:
    """Show the first-boot wizard and persist the user's choice.

    The dialog is modal.  If the user dismisses it via the window-close
    button without clicking **Get Started**, the default selection (NPU) is
    still saved so the wizard is not shown again.

    Parameters
    ----------
    app:
        The running ``QApplication`` instance (unused, reserved for future
        parent-window resolution).
    settings_manager:
        Application :class:`~src.settings.SettingsManager`.  The chosen
        backend (and, when applicable, Ollama URL) are written immediately
        via :meth:`~src.settings.SettingsManager.set_many`.
    """
    if not _HAS_QT:
        logger.warning("PyQt5 unavailable; skipping first-boot wizard.")
        return

    dlg = FirstBootWizard()
    dlg.exec_()  # modal — result code is intentionally ignored

    backend = dlg.chosen_backend
    changes: dict = {"backend": backend}
    if backend in (BACKEND_OLLAMA, BACKEND_OLLAMA_NPU):
        changes["ollama.base_url"] = dlg.chosen_ollama_url

    settings_manager.set_many(changes)
    logger.info("First-boot setup complete: backend=%r", backend)
