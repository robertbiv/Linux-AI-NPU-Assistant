# SPDX-License-Identifier: GPL-3.0-or-later
"""Root pytest configuration.

Sets QT_QPA_PLATFORM=offscreen *before* any Qt code is imported so every test
runs headlessly without Xvfb or a physical display.  The offscreen platform
ships with PyQt5 and renders widgets into off-screen buffers; widget.grab()
still returns valid pixmaps so screenshots work.
"""
from __future__ import annotations

import os

# Must be set before the QApplication singleton is created (pytest-qt creates
# it inside the `qapp` fixture).  setdefault() lets CI override via the env.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Silence Qt accessibility warnings that flood test output on headless runs.
os.environ.setdefault("QT_ACCESSIBILITY", "0")
