"""Tests for src/gui/setup_wizard.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.gui.setup_wizard import (
    BACKEND_NPU,
    BACKEND_OLLAMA,
    BACKEND_OLLAMA_NPU,
    DEFAULT_OLLAMA_URL,
    needs_first_boot_setup,
    run_first_boot_wizard,
)


# ── needs_first_boot_setup ────────────────────────────────────────────────────


class TestNeedsFirstBootSetup:
    def test_true_when_file_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "settings.json"
        assert needs_first_boot_setup(missing) is True

    def test_false_when_file_exists(self, tmp_path: Path) -> None:
        p = tmp_path / "settings.json"
        p.write_text("{}")
        assert needs_first_boot_setup(p) is False

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "settings.json")
        assert needs_first_boot_setup(missing) is True


# ── FirstBootWizard (logic only, no display) ──────────────────────────────────


try:
    from src.gui.setup_wizard import FirstBootWizard
    _HAS_QT = True
except Exception:
    _HAS_QT = False


@pytest.mark.skipif(not _HAS_QT, reason="PyQt5 not installed")
class TestFirstBootWizardDefaults:
    """Test wizard initial state without showing the GUI."""

    def test_default_backend_is_npu(self, qapp) -> None:
        dlg = FirstBootWizard()
        assert dlg.chosen_backend == BACKEND_NPU
        dlg.destroy()

    def test_default_ollama_url(self, qapp) -> None:
        dlg = FirstBootWizard()
        assert dlg.chosen_ollama_url == DEFAULT_OLLAMA_URL
        dlg.destroy()

    def test_npu_card_selected_by_default(self, qapp) -> None:
        dlg = FirstBootWizard()
        assert dlg._card_npu.is_checked() is True
        assert dlg._card_ollama.is_checked() is False
        assert dlg._card_both.is_checked() is False
        dlg.destroy()

    def test_url_row_hidden_by_default(self, qapp) -> None:
        dlg = FirstBootWizard()
        assert dlg._url_row.isHidden() is True
        dlg.destroy()

    def test_selecting_ollama_shows_url_row(self, qapp) -> None:
        dlg = FirstBootWizard()
        dlg._card_ollama.set_checked(True)
        assert dlg._url_row.isHidden() is False
        assert dlg.chosen_backend == BACKEND_OLLAMA
        dlg.destroy()

    def test_selecting_both_shows_url_row(self, qapp) -> None:
        dlg = FirstBootWizard()
        dlg._card_both.set_checked(True)
        assert dlg._url_row.isHidden() is False
        assert dlg.chosen_backend == BACKEND_OLLAMA_NPU
        dlg.destroy()

    def test_selecting_npu_hides_url_row(self, qapp) -> None:
        dlg = FirstBootWizard()
        dlg._card_ollama.set_checked(True)
        dlg._card_npu.set_checked(True)
        assert dlg._url_row.isHidden() is True
        assert dlg.chosen_backend == BACKEND_NPU
        dlg.destroy()

    def test_custom_ollama_url_stored(self, qapp) -> None:
        dlg = FirstBootWizard()
        dlg._url_edit.setText("http://192.168.1.10:11434")
        assert dlg.chosen_ollama_url == "http://192.168.1.10:11434"
        dlg.destroy()

    def test_blank_url_falls_back_to_default(self, qapp) -> None:
        dlg = FirstBootWizard()
        dlg._url_edit.setText("")
        assert dlg.chosen_ollama_url == DEFAULT_OLLAMA_URL
        dlg.destroy()


# ── run_first_boot_wizard ─────────────────────────────────────────────────────


class TestRunFirstBootWizard:
    """Test run_first_boot_wizard without a real GUI event loop."""

    def _make_settings(self):
        sm = MagicMock()
        sm.set_many = MagicMock()
        return sm

    def test_no_qt_is_silent(self) -> None:
        """Without PyQt5 the function must not raise."""
        sm = self._make_settings()
        with patch("src.gui.setup_wizard._HAS_QT", False):
            run_first_boot_wizard(None, sm)
        sm.set_many.assert_not_called()

    @pytest.mark.skipif(not _HAS_QT, reason="PyQt5 not installed")
    def test_npu_selection_saved(self) -> None:
        sm = self._make_settings()
        mock_dlg = MagicMock()
        mock_dlg.exec_.return_value = 1
        mock_dlg.chosen_backend = BACKEND_NPU
        mock_dlg.chosen_ollama_url = DEFAULT_OLLAMA_URL

        with patch("src.gui.setup_wizard.FirstBootWizard", return_value=mock_dlg):
            run_first_boot_wizard(None, sm)

        sm.set_many.assert_called_once()
        call_args = sm.set_many.call_args[0][0]
        assert call_args["backend"] == BACKEND_NPU
        assert "ollama.base_url" not in call_args

    @pytest.mark.skipif(not _HAS_QT, reason="PyQt5 not installed")
    def test_ollama_selection_saves_url(self) -> None:
        sm = self._make_settings()
        mock_dlg = MagicMock()
        mock_dlg.exec_.return_value = 1
        mock_dlg.chosen_backend = BACKEND_OLLAMA
        mock_dlg.chosen_ollama_url = "http://192.168.1.5:11434"

        with patch("src.gui.setup_wizard.FirstBootWizard", return_value=mock_dlg):
            run_first_boot_wizard(None, sm)

        call_args = sm.set_many.call_args[0][0]
        assert call_args["backend"] == BACKEND_OLLAMA
        assert call_args["ollama.base_url"] == "http://192.168.1.5:11434"

    @pytest.mark.skipif(not _HAS_QT, reason="PyQt5 not installed")
    def test_hybrid_selection_saves_url(self) -> None:
        sm = self._make_settings()
        mock_dlg = MagicMock()
        mock_dlg.exec_.return_value = 1
        mock_dlg.chosen_backend = BACKEND_OLLAMA_NPU
        mock_dlg.chosen_ollama_url = DEFAULT_OLLAMA_URL

        with patch("src.gui.setup_wizard.FirstBootWizard", return_value=mock_dlg):
            run_first_boot_wizard(None, sm)

        call_args = sm.set_many.call_args[0][0]
        assert call_args["backend"] == BACKEND_OLLAMA_NPU
        assert "ollama.base_url" in call_args

    @pytest.mark.skipif(not _HAS_QT, reason="PyQt5 not installed")
    def test_dismissed_dialog_still_saves_default(self) -> None:
        """Closing the dialog with X still persists the default backend."""
        sm = self._make_settings()
        mock_dlg = MagicMock()
        mock_dlg.exec_.return_value = 0  # Rejected / closed
        mock_dlg.chosen_backend = BACKEND_NPU
        mock_dlg.chosen_ollama_url = DEFAULT_OLLAMA_URL

        with patch("src.gui.setup_wizard.FirstBootWizard", return_value=mock_dlg):
            run_first_boot_wizard(None, sm)

        sm.set_many.assert_called_once()
