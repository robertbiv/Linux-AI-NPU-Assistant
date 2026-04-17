import pytest
from src.utils import _deep_merge, is_running_in_flatpak

def test_deep_merge():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 4, "e": 5}, "f": 6}
    merged = _deep_merge(base, override)
    assert merged == {"a": 1, "b": {"c": 4, "d": 3, "e": 5}, "f": 6}
    assert base == {"a": 1, "b": {"c": 2, "d": 3}}


class TestIsRunningInFlatpak:
    def test_false_when_no_env_and_no_file(self, monkeypatch):
        monkeypatch.delenv("FLATPAK_ID", raising=False)
        from unittest.mock import patch
        with patch("src.utils.Path.exists", return_value=False):
            assert is_running_in_flatpak() is False

    def test_true_when_flatpak_id_set(self, monkeypatch):
        monkeypatch.setenv("FLATPAK_ID", "io.github.robertbiv.LinuxAiNpuAssistant")
        from unittest.mock import patch
        with patch("src.utils.Path.exists", return_value=False):
            assert is_running_in_flatpak() is True

    def test_true_when_flatpak_info_file_exists(self, monkeypatch):
        monkeypatch.delenv("FLATPAK_ID", raising=False)
        from unittest.mock import patch
        with patch("src.utils.Path.exists", return_value=True):
            assert is_running_in_flatpak() is True
