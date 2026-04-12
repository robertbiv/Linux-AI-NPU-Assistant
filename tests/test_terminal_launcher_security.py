import sys
from unittest.mock import MagicMock, patch

# Mock yaml if it's not available (to support environments without it during testing)
try:
    import yaml
except ImportError:
    sys.modules['yaml'] = MagicMock()

import pytest
from src import terminal_launcher

@patch("src.config.load")
@patch("src.terminal_launcher._find_terminal")
def test_open_with_command_blocked(mock_find_terminal, mock_load):
    mock_config = MagicMock()
    mock_config.safety = {"blocked_commands": [r"rm\s+-rf\s+/"]}
    mock_load.return_value = mock_config
    mock_find_terminal.return_value = ("/usr/bin/xterm", "dashe")

    blocked_command = "rm -rf /"
    success, msg = terminal_launcher.open_with_command(blocked_command)

    assert success is False
    assert "blocked by safety policy" in msg

@patch("src.config.load")
@patch("src.terminal_launcher._find_terminal")
@patch("src.shell_detector.detect")
@patch("subprocess.Popen")
def test_open_with_command_sanitization(mock_popen, mock_detect, mock_find_terminal, mock_load):
    mock_config = MagicMock()
    mock_config.safety = {"blocked_commands": []}
    mock_load.return_value = mock_config
    mock_find_terminal.return_value = ("/usr/bin/xterm", "dashe")
    mock_detect.return_value = MagicMock(family="bash", path="/bin/bash", name="bash")

    command = "echo hello\x07world"
    success, msg = terminal_launcher.open_with_command(command)

    assert success is True
    assert "helloworld" in msg
    assert "\x07" not in msg

def test_zsh_template_no_eval():
    from src.terminal_launcher import _ZSH_SCRIPT
    assert "eval" not in _ZSH_SCRIPT
    assert "print -z" in _ZSH_SCRIPT
    assert "exec zsh -i" in _ZSH_SCRIPT

def test_fish_template_robustness():
    from src.terminal_launcher import _FISH_WRAPPER
    assert "export _CMD" in _FISH_WRAPPER
    assert "commandline -- \"$_CMD\"" in _FISH_WRAPPER
