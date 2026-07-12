
import time
from collections import deque, Counter

import numpy as np


WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle (+ palm)
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring (+ palm)
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky (+ palm)
    (0, 17),                                 # wrist to pinky base
]


MOVE = "MOVE"
GRAB = "GRAB"
RIGHT_CLICK = "RIGHT_CLICK"
DOUBLE_CLICK = "DOUBLE_CLICK"
SCROLL_UP = "SCROLL_UP"
SCROLL_DOWN = "SCROLL_DOWN"
MEDIA_TOGGLE = "MEDIA_TOGGLE"
LOCK_SCREEN = "LOCK_SCREEN"
PRECISION = "PRECISION"
THREE_SWEEP = "THREE_SWEEP"
TWO_HAND_ZOOM = "TWO_HAND_ZOOM"
LASER = "LASER"


GESTURE_LEGEND = [
    (MOVE, "default / open hand", "move cursor"),
    (GRAB, "thumb + index pinch", "click / drag"),
    (RIGHT_CLICK, "thumb + middle pinch", "right-click"),
    (DOUBLE_CLICK, "thumb + index + middle pinch", "double-click"),
    (f"{SCROLL_UP}/{SCROLL_DOWN}", "thumbs up / down", "scroll"),
    (MEDIA_TOGGLE, "shaka: thumb + pinky out", "play / pause"),
    (PRECISION, "peace sign, held", "slow / precise move"),
    (THREE_SWEEP, "3 fingers, swept sideways", "switch desktop / slide"),
    (LOCK_SCREEN, "middle finger alone, held", "lock screen"),
    (TWO_HAND_ZOOM, "both hands pinch, move apart", "zoom in / out"),
]



def lm_xy(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y], dtype=np.float32)


def dist(a, b):
    return float(np.linalg.norm(a - b))


def finger_extended(landmarks, mcp_idx, pip_idx, tip_idx, wrist, margin):
    """Scale/rotation-robust check: a finger counts as extended if its tip is
    meaningfully farther from the wrist than its own pip joint is -- this
    works regardless of hand rotation or distance from the camera, unlike a
    plain 'tip.y < pip.y' check."""
    tip = lm_xy(landmarks, tip_idx)
    pip = lm_xy(landmarks, pip_idx)
    return dist(wrist, tip) > dist(wrist, pip) * margin


def hand_scale_of(landmarks):
    """Wrist-to-middle-mcp distance, used throughout to normalize distance
    thresholds for hand size / camera distance."""
    return dist(lm_xy(landmarks, WRIST), lm_xy(landmarks, MIDDLE_MCP)) + 1e-6


def is_pinching(landmarks, pinch_ratio):
    """True if this single hand's thumb and index tip are pinched together.
    Used by the two-hand zoom check in the main loop (each hand is tested
    independently before the pair is treated as a zoom gesture)."""
    wrist = lm_xy(landmarks, WRIST)
    scale = hand_scale_of(landmarks)
    thumb_tip = lm_xy(landmarks, THUMB_TIP)
    index_tip = lm_xy(landmarks, INDEX_TIP)
    return dist(thumb_tip, index_tip) / scale < pinch_ratio


def pinch_center(landmarks):
    """Midpoint between thumb tip and index tip -- the anchor point used to
    measure the distance between two pinching hands for zoom."""
    return (lm_xy(landmarks, THUMB_TIP) + lm_xy(landmarks, INDEX_TIP)) / 2.0


def inter_hand_pinch_distance(hand_a, hand_b):
    return dist(pinch_center(hand_a), pinch_center(hand_b))



def classify_gesture(landmarks, cfg):
    """Returns (gesture_name, cursor_tracking_point) for a single hand.

    Priority order matters: the most geometrically distinctive shapes (the
    pinches) are checked first, from most- to least-specific, and MOVE is
    the fallback for anything that doesn't match a specific gesture --
    including an open palm.
    """
    wrist = lm_xy(landmarks, WRIST)
    hand_scale = hand_scale_of(landmarks)
    margin = cfg.finger_extended_margin

    index_ext = finger_extended(landmarks, INDEX_MCP, INDEX_PIP, INDEX_TIP, wrist, margin)
    middle_ext = finger_extended(landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, wrist, margin)
    ring_ext = finger_extended(landmarks, RING_MCP, RING_PIP, RING_TIP, wrist, margin)
    pinky_ext = finger_extended(landmarks, PINKY_MCP, PINKY_PIP, PINKY_TIP, wrist, margin)
    thumb_ext = finger_extended(landmarks, THUMB_MCP, THUMB_IP, THUMB_TIP, wrist, margin)

    thumb_tip = lm_xy(landmarks, THUMB_TIP)
    index_tip = lm_xy(landmarks, INDEX_TIP)
    middle_tip = lm_xy(landmarks, MIDDLE_TIP)

    pinch_index = dist(thumb_tip, index_tip) / hand_scale < cfg.pinch_ratio
    pinch_middle = dist(thumb_tip, middle_tip) / hand_scale < cfg.pinch_ratio

    tripod = (
        dist(thumb_tip, index_tip) / hand_scale < cfg.double_click_pinch_ratio
        and dist(thumb_tip, middle_tip) / hand_scale < cfg.double_click_pinch_ratio
    )


    if cfg.enable_double_click and tripod:
        return DOUBLE_CLICK, index_tip

    
    if pinch_index:
        return GRAB, index_tip

    
    if cfg.enable_right_click and pinch_middle:
        return RIGHT_CLICK, middle_tip

    other_four_curled = not index_ext and not middle_ext and not ring_ext and not pinky_ext


    if thumb_ext and other_four_curled:
        vertical_ratio = (wrist[1] - thumb_tip[1]) / hand_scale
        if vertical_ratio > cfg.thumb_vertical_ratio:
            return SCROLL_UP, thumb_tip
        if vertical_ratio < -cfg.thumb_vertical_ratio:
            return SCROLL_DOWN, thumb_tip

    
    if cfg.enable_media_control and thumb_ext and pinky_ext and not index_ext and not middle_ext and not ring_ext:
        return MEDIA_TOGGLE, thumb_tip


    if middle_ext and not index_ext and not ring_ext and not pinky_ext:
        return LOCK_SCREEN, middle_tip


    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return PRECISION, index_tip


    if index_ext and middle_ext and ring_ext and not pinky_ext:
        return THREE_SWEEP, middle_tip


    return MOVE, index_tip


def draw_hand_overlay(frame, landmarks, frame_w, frame_h, color=(0, 200, 0)):
    """Manual landmark/skeleton overlay (no dependency on the legacy
    mp.solutions.drawing_utils module)."""
    import cv2  # local import: keeps this module importable/testable without a display
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def normalized_to_screen(x, y, screen_w, screen_h, margin):
    x_eff = (x - margin) / (1 - 2 * margin)
    y_eff = (y - margin) / (1 - 2 * margin)
    x_eff = min(max(x_eff, 0.0), 1.0)
    y_eff = min(max(y_eff, 0.0), 1.0)
    return x_eff * screen_w, y_eff * screen_h



class VideoClock:


    def __init__(self):
        self.start = time.time()
        self.last_ms = -1

    def next_ms(self):
        ms = int((time.time() - self.start) * 1000)
        if ms <= self.last_ms:
            ms = self.last_ms + 1
        self.last_ms = ms
        return ms



class GestureStabilizer:
    def __init__(self, window=5):
        self.history = deque(maxlen=window)

    def push(self, gesture):
        self.history.append(gesture)
        return Counter(self.history).most_common(1)[0][0]

    def reset(self):
        self.history.clear()



class DeltaTracker:
    def __init__(self, min_delta, max_window_s, pos_label, neg_label):
        self.min_delta = min_delta
        self.max_window_s = max_window_s
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.samples = deque()

    def reset(self):
        self.samples.clear()

    def update(self, value, t):
        """Returns pos_label, neg_label, or None."""
        self.samples.append((t, value))
        while self.samples and t - self.samples[0][0] > self.max_window_s:
            self.samples.popleft()
        if len(self.samples) < 2:
            return None
        d = self.samples[-1][1] - self.samples[0][1]
        if d > self.min_delta:
            self.reset()
            return self.pos_label
        if d < -self.min_delta:
            self.reset()
            return self.neg_label
        return None
