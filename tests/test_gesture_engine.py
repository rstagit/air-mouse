import dataclasses

import pytest

from config import Config
from gesture_engine import (
    classify_gesture, normalized_to_screen, is_pinching, inter_hand_pinch_distance,
    GestureStabilizer, DeltaTracker, VideoClock,
    MOVE, GRAB, RIGHT_CLICK, DOUBLE_CLICK, SCROLL_UP, SCROLL_DOWN,
    MEDIA_TOGGLE, LOCK_SCREEN, PRECISION, THREE_SWEEP,
)
from hand_fixtures import make_hand, DIR_THUMB_UP, DIR_THUMB_DOWN, DIR_THUMB_SHAKA


@pytest.fixture
def cfg():
    return Config()


# ---------------------------------------------------------------------------
# One shape -> one gesture, for every gesture in the table.
# ---------------------------------------------------------------------------

def test_open_hand_falls_back_to_move(cfg):
    hand = make_hand(index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True, thumb_ext=True)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture == MOVE


def test_closed_fist_falls_back_to_move(cfg):
    hand = make_hand()  # everything curled, default thumb_dir (side), not pinching
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture == MOVE


def test_thumb_index_pinch_is_grab(cfg):
    # middle_ext=True keeps the middle fingertip away from the thumb/index
    # pinch point -- two CURLED fingertips naturally sit close to each
    # other, which would otherwise also satisfy pinch_middle by accident
    # and misclassify this as DOUBLE_CLICK. This mirrors a real
    # consideration: a loosely-curled GRAB should keep the middle finger
    # clear of the thumb to avoid an accidental double-click reading.
    hand = make_hand(pinch_index=True, middle_ext=True)
    gesture, track_pt = classify_gesture(hand, cfg)
    assert gesture == GRAB
    # GRAB tracks the index fingertip
    assert track_pt[0] == hand[8].x and track_pt[1] == hand[8].y


def test_thumb_middle_pinch_is_right_click(cfg):
    # index_ext=True for the same reason as above, mirrored: keeps the
    # index fingertip away from the thumb/middle pinch point.
    hand = make_hand(pinch_middle=True, index_ext=True)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture == RIGHT_CLICK


def test_tripod_pinch_is_double_click_not_grab(cfg):
    hand = make_hand(pinch_index=True, pinch_middle=True)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture == DOUBLE_CLICK


def test_thumbs_up_is_scroll_up(cfg):
    hand = make_hand(thumb_ext=True, thumb_dir=DIR_THUMB_UP)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture == SCROLL_UP


def test_thumbs_down_is_scroll_down(cfg):
    hand = make_hand(thumb_ext=True, thumb_dir=DIR_THUMB_DOWN)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture == SCROLL_DOWN


def test_shaka_is_media_toggle(cfg):
    hand = make_hand(thumb_ext=True, pinky_ext=True, thumb_dir=DIR_THUMB_SHAKA)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture == MEDIA_TOGGLE


def test_middle_alone_is_lock_screen(cfg):
    hand = make_hand(middle_ext=True)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture == LOCK_SCREEN


def test_peace_sign_is_precision(cfg):
    hand = make_hand(index_ext=True, middle_ext=True)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture == PRECISION


def test_three_fingers_is_sweep(cfg):
    hand = make_hand(index_ext=True, middle_ext=True, ring_ext=True)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture == THREE_SWEEP


# ---------------------------------------------------------------------------
# Priority / overlap: the three pinch gestures must never collide, and
# disabling a feature must fall back sanely instead of misfiring.
# ---------------------------------------------------------------------------

def test_double_click_takes_priority_over_grab_and_right_click(cfg):
    hand = make_hand(pinch_index=True, pinch_middle=True)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture == DOUBLE_CLICK
    assert gesture != GRAB
    assert gesture != RIGHT_CLICK


def test_double_click_disabled_falls_back_to_grab(cfg):
    cfg = dataclasses.replace(cfg, enable_double_click=False)
    hand = make_hand(pinch_index=True, pinch_middle=True)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture == GRAB


def test_right_click_disabled_does_not_right_click(cfg):
    cfg = dataclasses.replace(cfg, enable_right_click=False)
    hand = make_hand(pinch_middle=True)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture != RIGHT_CLICK


def test_media_control_disabled_does_not_fire(cfg):
    cfg = dataclasses.replace(cfg, enable_media_control=False)
    hand = make_hand(thumb_ext=True, pinky_ext=True, thumb_dir=DIR_THUMB_SHAKA)
    gesture, _ = classify_gesture(hand, cfg)
    assert gesture != MEDIA_TOGGLE


def test_sweep_requires_ring_precision_requires_no_ring(cfg):
    """Precision (index+middle) and sweep (index+middle+ring) must be
    mutually exclusive based on the ring finger alone."""
    precision_hand = make_hand(index_ext=True, middle_ext=True, ring_ext=False)
    sweep_hand = make_hand(index_ext=True, middle_ext=True, ring_ext=True)
    assert classify_gesture(precision_hand, cfg)[0] == PRECISION
    assert classify_gesture(sweep_hand, cfg)[0] == THREE_SWEEP


def test_pinch_ratio_is_tunable(cfg):
    """A borderline pinch (thumb-index distance = exactly 0.30x hand
    scale) should register as GRAB under the default threshold (0.35) but
    not under a stricter one (0.20) -- proves the threshold is actually
    load-bearing, not just trivially satisfied."""
    from types import SimpleNamespace
    from gesture_engine import hand_scale_of, INDEX_TIP, THUMB_TIP

    hand = make_hand(index_ext=True)
    scale = hand_scale_of(hand)
    index_tip = hand[INDEX_TIP]
    hand[THUMB_TIP] = SimpleNamespace(x=index_tip.x + 0.30 * scale, y=index_tip.y, z=0.0)

    loose_cfg = dataclasses.replace(cfg, pinch_ratio=0.35)
    strict_cfg = dataclasses.replace(cfg, pinch_ratio=0.20)

    assert classify_gesture(hand, loose_cfg)[0] == GRAB
    assert classify_gesture(hand, strict_cfg)[0] != GRAB


# ---------------------------------------------------------------------------
# Two-hand pinch-zoom helpers
# ---------------------------------------------------------------------------

def test_is_pinching_true_and_false(cfg):
    pinching = make_hand(pinch_index=True)
    open_hand = make_hand(index_ext=True)
    assert is_pinching(pinching, cfg.pinch_ratio) is True
    assert is_pinching(open_hand, cfg.pinch_ratio) is False


def test_inter_hand_pinch_distance_changes_with_separation():
    hand_a = make_hand(pinch_index=True)
    hand_b = make_hand(pinch_index=True)
    # Same fixture pose placed at the same coordinates -> distance ~0.
    assert inter_hand_pinch_distance(hand_a, hand_b) == pytest.approx(0.0, abs=1e-6)

    # Shift hand_b's pinch center by a known offset -> distance should
    # match that offset exactly, proving the function actually measures
    # separation rather than always returning 0.
    from types import SimpleNamespace
    from gesture_engine import THUMB_TIP, INDEX_TIP
    for idx in (THUMB_TIP, INDEX_TIP):
        p = hand_b[idx]
        hand_b[idx] = SimpleNamespace(x=p.x + 0.2, y=p.y, z=0.0)
    assert inter_hand_pinch_distance(hand_a, hand_b) == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# GestureStabilizer: majority vote over a rolling window.
# ---------------------------------------------------------------------------

def test_stabilizer_requires_majority_before_switching():
    stab = GestureStabilizer(window=5)
    for _ in range(5):
        assert stab.push(MOVE) == MOVE
    # A single flicker frame shouldn't flip the confirmed gesture.
    assert stab.push(GRAB) == MOVE
    # Once GRAB has a majority in the window, it should win.
    for _ in range(4):
        result = stab.push(GRAB)
    assert result == GRAB


def test_stabilizer_reset_clears_history():
    stab = GestureStabilizer(window=3)
    stab.push(GRAB)
    stab.push(GRAB)
    stab.reset()
    assert stab.push(MOVE) == MOVE


# ---------------------------------------------------------------------------
# DeltaTracker: generic rolling-window delta firing, used for both sweep
# (x-position) and zoom (inter-hand distance).
# ---------------------------------------------------------------------------

def test_delta_tracker_fires_positive_and_resets():
    tracker = DeltaTracker(min_delta=0.1, max_window_s=0.5, pos_label="right", neg_label="left")
    assert tracker.update(0.0, t=0.0) is None
    assert tracker.update(0.05, t=0.1) is None
    result = tracker.update(0.2, t=0.2)
    assert result == "right"
    # Fired -> internal samples cleared -> shouldn't immediately re-fire
    # from stale data.
    assert tracker.update(0.21, t=0.21) is None


def test_delta_tracker_fires_negative():
    tracker = DeltaTracker(min_delta=0.1, max_window_s=0.5, pos_label="in", neg_label="out")
    tracker.update(0.5, t=0.0)
    result = tracker.update(0.3, t=0.1)
    assert result == "out"


def test_delta_tracker_drops_stale_samples_outside_window():
    tracker = DeltaTracker(min_delta=0.1, max_window_s=0.2, pos_label="right", neg_label="left")
    tracker.update(0.0, t=0.0)
    # Big gap: the first sample should age out of the window, so a small
    # move afterward must NOT fire just because of the old baseline.
    result = tracker.update(0.05, t=5.0)
    assert result is None


def test_delta_tracker_reset():
    tracker = DeltaTracker(min_delta=0.1, max_window_s=0.5, pos_label="right", neg_label="left")
    tracker.update(0.0, t=0.0)
    tracker.reset()
    assert tracker.update(0.05, t=0.01) is None  # no baseline left to compare against


# ---------------------------------------------------------------------------
# normalized_to_screen: margin-aware mapping + clamping.
# ---------------------------------------------------------------------------

def test_normalized_to_screen_center():
    x, y = normalized_to_screen(0.5, 0.5, 1920, 1080, margin=0.12)
    assert x == pytest.approx(960, abs=1)
    assert y == pytest.approx(540, abs=1)


def test_normalized_to_screen_clamps_to_edges():
    x, y = normalized_to_screen(-1.0, 2.0, 1920, 1080, margin=0.12)
    assert x == 0.0
    assert y == 1080.0


def test_normalized_to_screen_margin_expands_range():
    # At x = margin, effective x should be 0 (left edge of the usable range).
    x, _ = normalized_to_screen(0.12, 0.5, 1000, 1000, margin=0.12)
    assert x == pytest.approx(0.0, abs=1e-4)
    # At x = 1 - margin, effective x should be the right edge.
    x, _ = normalized_to_screen(0.88, 0.5, 1000, 1000, margin=0.12)
    assert x == pytest.approx(1000.0, abs=1e-4)


# ---------------------------------------------------------------------------
# VideoClock: strictly increasing timestamps, as required by MediaPipe's
# VIDEO running mode.
# ---------------------------------------------------------------------------

def test_video_clock_strictly_increasing():
    clock = VideoClock()
    values = [clock.next_ms() for _ in range(50)]
    assert all(b > a for a, b in zip(values, values[1:]))
