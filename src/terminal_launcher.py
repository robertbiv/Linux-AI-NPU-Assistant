"""Terminal launcher — open the user's terminal pre-filled with a command.

The command is inserted into the terminal's readline buffer so the user can
see and edit it before pressing Enter to run it.  The assistant **never**
executes the command itself; it only opens the terminal.

Technique
---------
We write a tiny temporary shell script that uses ``read -e -i "<cmd>"`` to
pre-fill readline, then ``eval`` only after the user confirms.  The script
is passed to the terminal emulator's ``-e`` / ``--`` flag.  The temp file
is cleaned up automatically after the shell exits.

Supported terminal emulators (tried in order)
----------------------------------------------
- ``x-terminal-emulator``  (Debian/Ubuntu alternatives system)
- ``gnome-terminal``
- ``konsole``
- ``xfce4-terminal``
- ``mate-terminal``
- ``lxterminal``
- ``xterm``
- ``kitty``
- ``alacritty``
- ``wezterm``
- ``tilix``
"""

from __future__ import annotations

import logging
import os
import shlex
import stat
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Ordered list of (executable, arg_style) where arg_style is one of:
#   "dashe"   → terminal -e 'shell_cmd'          (xterm, most GTK terminals)
#   "dashdash"→ terminal -- shell_cmd args...    (gnome-terminal, kitty, alacritty)
#   "execute" → terminal --command='shell_cmd'   (konsole)
_TERMINALS: list[tuple[str, str]] = [
    ("x-terminal-emulator", "dashe"),
    ("gnome-terminal",      "dashdash"),
    ("konsole",             "execute"),
    ("xfce4-terminal",      "dashe"),
    ("mate-terminal",       "dashe"),
    ("lxterminal",          "dashe"),
    ("tilix",               "dashe"),
    ("kitty",               "dashdash"),
    ("alacritty",           "dashdash"),
    ("wezterm",             "dashdash"),
    ("xterm",               "dashe"),
]

# Shell script template.  $1 is the command string to pre-fill.
# Uses bash's read -e (readline) with -i (initial text) so the user can
# edit the command before pressing Enter.  After running, a new interactive
# shell is spawned so the window stays open.
_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
set -euo pipefail
_CMD={quoted_cmd}
printf '\\n  The assistant suggests this command:\\n\\n'
printf '  %s\\n\\n' "$_CMD"
printf 'You can edit it, then press Enter to run (Ctrl-C to cancel).\\n\\n'
read -r -e -p '$ ' -i "$_CMD" _CONFIRMED
if [ -n "$_CONFIRMED" ]; then
    eval "$_CONFIRMED"
fi
printf '\\n[Press Enter to close this window]'
read -r _
"""


def _find_terminal() -> tuple[str, str] | None:
    """Return (executable_path, arg_style) for the first available terminal."""
    import shutil  # lazy
    for exe, style in _TERMINALS:
        path = shutil.which(exe)
        if path:
            return path, style
    return None


def _build_launch_cmd(terminal: str, style: str, script_path: str) -> list[str]:
    """Build the argv list to launch *terminal* running the script."""
    if style == "dashdash":
        return [terminal, "--", "bash", script_path]
    if style == "execute":
        return [terminal, f"--command=bash {shlex.quote(script_path)}"]
    # "dashe" default
    return [terminal, "-e", f"bash {shlex.quote(script_path)}"]


def open_with_command(command: str) -> tuple[bool, str]:
    """Open the user's default terminal pre-filled with *command*.

    The user sees the command in an editable readline buffer and must press
    Enter to execute it (or Ctrl-C to cancel).  This function returns
    immediately — it does **not** wait for the terminal to close.

    Parameters
    ----------
    command:
        The shell command string to pre-fill.

    Returns
    -------
    (success, message)
        ``success`` is ``True`` if the terminal was launched.  ``message``
        describes what happened (useful for UI display).
    """
    import subprocess  # lazy

    terminal_info = _find_terminal()
    if terminal_info is None:
        msg = (
            "No supported terminal emulator found. "
            "Install one of: gnome-terminal, konsole, xterm, kitty, alacritty."
        )
        logger.warning("terminal_launcher: %s", msg)
        return False, msg

    terminal, style = terminal_info

    # Write the pre-fill script to a temporary file.
    # The script self-cleans: the temp dir is deleted when the shell exits
    # because we set delete=False and clean up in the script itself — but
    # simpler: we let the OS clean /tmp on reboot.  The file is unlinked
    # after we've launched the terminal (the bash process has already opened
    # it by then).
    quoted = shlex.quote(command)
    script_body = _SCRIPT_TEMPLATE.format(quoted_cmd=quoted)

    try:
        fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="ai_helper_")
        try:
            os.write(fd, script_body.encode())
        finally:
            os.close(fd)
        os.chmod(script_path, stat.S_IRWXU)  # owner-only execute

        launch_cmd = _build_launch_cmd(terminal, style, script_path)
        logger.info("terminal_launcher: %s", " ".join(launch_cmd))

        subprocess.Popen(
            launch_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        # Give the terminal a moment to open the script before we unlink it
        # (Popen is async; the file needs to exist until bash reads it)
        # We schedule deletion via a background thread.
        _schedule_delete(script_path, delay=5.0)

        return True, f"Opened terminal with command: {command}"

    except Exception as exc:  # noqa: BLE001
        logger.error("terminal_launcher error: %s", exc)
        return False, str(exc)


def _schedule_delete(path: str, delay: float) -> None:
    """Delete *path* after *delay* seconds in a daemon thread."""
    import threading
    import time

    def _delete() -> None:
        time.sleep(delay)
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass

    t = threading.Thread(target=_delete, daemon=True)
    t.start()
