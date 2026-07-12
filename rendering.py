
import cv2

from gesture_engine import GESTURE_LEGEND


def draw_hud(frame, cam_height, status, color, fps, mode_flags):
    """status/color: current gesture label and its display color.
    mode_flags: list of short strings for any active toggle modes
    (Presentation, Laser, Dwell) shown next to the gesture line."""
    cv2.putText(frame, f"Gesture: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if mode_flags:
        cv2.putText(frame, "  ".join(mode_flags), (10, 88), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 165, 255), 2)
    cv2.putText(
        frame,
        "SPACE=pause  h=help  q/ESC=quit",
        (10, cam_height - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )


def draw_legend(frame, cam_width, cam_height):
    """Semi-transparent panel listing every gesture and what it does,
    toggled on/off with the 'h' key."""
    lines = [f"{name}: {shape} -> {action}" for name, shape, action in GESTURE_LEGEND]
    lines += [
        "",
        "p = toggle presentation mode (sweep -> slide nav)",
        "l = toggle laser pointer mode",
        "d = toggle dwell-click (accessibility)",
        "c = re-run calibration",
    ]

    pad = 10
    line_h = 22
    box_w = min(cam_width - 20, 560)
    box_h = min(cam_height - 20, line_h * len(lines) + 2 * pad)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 100), (10 + box_w, 100 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    y = 100 + pad + 16
    for line in lines:
        if y > 100 + box_h - 4:
            break
        cv2.putText(frame, line, (10 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += line_h


class LaserPointerOverlay:


    def __init__(self, size=22, color="#ff2d2d"):
        self.size = size
        self._visible = False
        self._available = False
        try:
            import tkinter as tk
            self._tk = tk
            self.root = tk.Tk()
            self.root.withdraw()
            self.root.overrideredirect(True)
            self.root.attributes("-topmost", True)
            try:
                self.root.attributes("-alpha", 0.85)
            except tk.TclError:
                pass
            self.canvas = tk.Canvas(self.root, width=size, height=size,
                                     highlightthickness=0, bg="black")
            self.canvas.pack()
            pad = 2
            self.canvas.create_oval(pad, pad, size - pad, size - pad, fill=color, outline="")
            try:
                # Per-pixel color-key transparency -- supported on Windows,
                # not on X11/macOS Tk builds (falls back to the whole-window
                # -alpha translucency set above, which still reads fine as
                # a small floating dot).
                self.root.attributes("-transparentcolor", "black")
            except tk.TclError:
                pass
            self._available = True
        except Exception as e:
            print(f"Note: laser pointer overlay unavailable ({e}); "
                  f"laser mode will suspend mouse control without showing a dot.")

    def show(self):
        if self._available and not self._visible:
            self.root.deiconify()
            self._visible = True

    def hide(self):
        if self._available and self._visible:
            self.root.withdraw()
            self._visible = False

    def move_to(self, x, y):
        if self._available and self._visible:
            half = self.size // 2
            try:
                self.root.geometry(f"+{int(x - half)}+{int(y - half)}")
            except self._tk.TclError:
                pass

    def pump(self):
        if self._available:
            try:
                self.root.update_idletasks()
                self.root.update()
            except self._tk.TclError:
                self._available = False

    def close(self):
        if self._available:
            try:
                self.root.destroy()
            except Exception:
                pass
