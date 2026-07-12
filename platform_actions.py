import os
import platform

import pyautogui

OS_NAME = platform.system()  # 'Windows', 'Darwin' (macOS), or 'Linux'


def switch_desktop(direction):
    """direction: 'left' or 'right'. Hotkeys are OS-dependent; Linux varies
    by desktop environment (this targets GNOME's default binding)."""
    key = "right" if direction == "right" else "left"
    if OS_NAME == "Windows":
        pyautogui.hotkey("ctrl", "win", key)
    elif OS_NAME == "Darwin":
        pyautogui.hotkey("ctrl", key)
    else:
        pyautogui.hotkey("ctrl", "alt", key)


def navigate_slide(direction):
    pyautogui.press("pagedown" if direction == "right" else "pageup")


def lock_screen():
    if OS_NAME == "Windows":
        import ctypes
        ctypes.windll.user32.LockWorkStation()
    elif OS_NAME == "Darwin":
        pyautogui.hotkey("ctrl", "cmd", "q")
    else:
        for cmd in (
            "loginctl lock-session",
            "gnome-screensaver-command -l",
            "xdg-screensaver lock",
            "dm-tool lock",
        ):
            if os.system(cmd + " >/dev/null 2>&1") == 0:
                break


def toggle_media_playback(key="playpause"):
    try:
        pyautogui.press(key)
    except Exception as e:
        print(f"Warning: media key '{key}' failed ({e}); trying space instead.")
        try:
            pyautogui.press("space")
        except Exception as e2:
            print(f"Warning: fallback space key also failed ({e2}).")


def zoom(direction, step=2):
    pyautogui.keyDown("ctrl")
    try:
        pyautogui.scroll(step if direction == "in" else -step)
    finally:
        pyautogui.keyUp("ctrl")
