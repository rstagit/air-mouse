import sys
from unittest import mock

import pytest

import platform_actions as actions


@pytest.fixture
def fake_pyautogui(monkeypatch):
    fake = mock.MagicMock()
    monkeypatch.setattr(actions, "pyautogui", fake)
    return fake


def test_switch_desktop_windows(monkeypatch, fake_pyautogui):
    monkeypatch.setattr(actions, "OS_NAME", "Windows")
    actions.switch_desktop("right")
    fake_pyautogui.hotkey.assert_called_once_with("ctrl", "win", "right")


def test_switch_desktop_macos(monkeypatch, fake_pyautogui):
    monkeypatch.setattr(actions, "OS_NAME", "Darwin")
    actions.switch_desktop("left")
    fake_pyautogui.hotkey.assert_called_once_with("ctrl", "left")


def test_switch_desktop_linux(monkeypatch, fake_pyautogui):
    monkeypatch.setattr(actions, "OS_NAME", "Linux")
    actions.switch_desktop("right")
    fake_pyautogui.hotkey.assert_called_once_with("ctrl", "alt", "right")


def test_navigate_slide_maps_direction_to_page_keys(fake_pyautogui):
    actions.navigate_slide("right")
    fake_pyautogui.press.assert_called_once_with("pagedown")
    fake_pyautogui.reset_mock()
    actions.navigate_slide("left")
    fake_pyautogui.press.assert_called_once_with("pageup")


def test_lock_screen_windows_calls_lock_workstation(monkeypatch):
    monkeypatch.setattr(actions, "OS_NAME", "Windows")
    fake_ctypes = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    actions.lock_screen()
    fake_ctypes.windll.user32.LockWorkStation.assert_called_once()


def test_lock_screen_macos(monkeypatch, fake_pyautogui):
    monkeypatch.setattr(actions, "OS_NAME", "Darwin")
    actions.lock_screen()
    fake_pyautogui.hotkey.assert_called_once_with("ctrl", "cmd", "q")


def test_lock_screen_linux_tries_commands_until_one_succeeds(monkeypatch):
    monkeypatch.setattr(actions, "OS_NAME", "Linux")
    calls = []

    def fake_system(cmd):
        calls.append(cmd)
        return 0 if "xdg-screensaver" in cmd else 1

    monkeypatch.setattr(actions.os, "system", fake_system)
    actions.lock_screen()
    # Should have tried the first two (failing) commands before succeeding
    # on xdg-screensaver, and stopped there rather than trying dm-tool too.
    assert len(calls) == 3
    assert "xdg-screensaver" in calls[-1]


def test_toggle_media_playback_sends_configured_key(fake_pyautogui):
    actions.toggle_media_playback("playpause")
    fake_pyautogui.press.assert_called_once_with("playpause")


def test_toggle_media_playback_falls_back_to_space_on_failure(fake_pyautogui):
    fake_pyautogui.press.side_effect = [Exception("no media key support"), None]
    actions.toggle_media_playback("playpause")
    assert fake_pyautogui.press.call_args_list == [mock.call("playpause"), mock.call("space")]


def test_zoom_holds_ctrl_and_scrolls(fake_pyautogui):
    actions.zoom("in", step=3)
    fake_pyautogui.keyDown.assert_called_once_with("ctrl")
    fake_pyautogui.scroll.assert_called_once_with(3)
    fake_pyautogui.keyUp.assert_called_once_with("ctrl")

    fake_pyautogui.reset_mock()
    actions.zoom("out", step=3)
    fake_pyautogui.scroll.assert_called_once_with(-3)


def test_zoom_releases_ctrl_even_if_scroll_raises(fake_pyautogui):
    fake_pyautogui.scroll.side_effect = Exception("boom")
    with pytest.raises(Exception):
        actions.zoom("in")
    fake_pyautogui.keyUp.assert_called_once_with("ctrl")
