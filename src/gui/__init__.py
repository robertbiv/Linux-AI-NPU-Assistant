# SPDX-License-Identifier: GPL-3.0-or-later
"""GUI package for Neural Monolith (Linux AI NPU Assistant).

This package provides PyQt5-based GUI components.  All modules use
a conditional import pattern so that the rest of the application
remains importable even when PyQt5 is not installed.

## Modules
npu_theme
    NPU dark-mode colour palette and global QSS stylesheet constants.
main_window
    :class:`~src.gui.main_window.MainWindow` — entry-point window with
    compact overlay mode and full desktop mode, switchable at runtime.
full_window
    :class:`~src.gui.full_window.FullWindow` — wide desktop layout with
    left sidebar navigation (used by ``main_window`` in full mode).
chat_widget
    :class:`~src.gui.chat_widget.ChatWidget` — NPU-themed chat interface
    with streaming support and inline code-block rendering.
status_widget
    :class:`~src.gui.status_widget.StatusWidget` — live NPU performance
    dashboard with metrics, bar chart, and kernel list.
npu_settings_widget
    :class:`~src.gui.npu_settings_widget.NPUSettingsWidget` — settings
    page styled to match the Neural Monolith mockup design.
theme
    Desktop-environment detection and Qt style/palette application.
diagnostic_reporter
    Pure-Python (no Qt) system status collector — fully testable.
settings_window
    QDialog with tabbed settings pages (Backend, Models, Tools, Security, Appearance).
model_manager
    Model browser widget with file dialog, drag-and-drop, and delete support.
diagnostic_window
    QDialog showing live status of all subsystems plus a test runner.
"""
