from calibration import _clamp, _fit_finger_extended_margin, _fit_pinch_ratio
from hand_fixtures import make_hand


def test_clamp_bounds_values():
    assert _clamp(5, 0, 10) == 5
    assert _clamp(-5, 0, 10) == 0
    assert _clamp(50, 0, 10) == 10


def test_fit_finger_extended_margin_uses_open_hand_samples():
    open_samples = [make_hand(index_ext=True) for _ in range(10)]
    margin = _fit_finger_extended_margin(open_samples, default=1.10)
    # The fixture's extended index gives tip/pip ratio = 0.48/0.30 = 1.6;
    # fitted margin is 85% of that (1.36), within the clamp range.
    assert 1.02 <= margin <= 1.4
    assert margin > 1.2


def test_fit_finger_extended_margin_falls_back_to_default_on_empty_samples():
    assert _fit_finger_extended_margin([], default=1.10) == 1.10


def test_fit_pinch_ratio_uses_pinch_samples():
    pinch_samples = [make_hand(pinch_index=True) for _ in range(10)]
    ratio = _fit_pinch_ratio(pinch_samples, default=0.35)
    # Exact-coincidence pinch in the fixture -> observed ratio ~0 -> fitted
    # value should sit at the lower clamp bound.
    assert ratio == 0.15


def test_fit_pinch_ratio_falls_back_to_default_on_empty_samples():
    assert _fit_pinch_ratio([], default=0.35) == 0.35


def test_fit_pinch_ratio_scales_with_observed_distance():
    """A looser observed pinch should fit to a looser (but still clamped)
    threshold than a tighter one."""
    from types import SimpleNamespace
    from gesture_engine import THUMB_TIP, INDEX_TIP, hand_scale_of

    loose_hand = make_hand(index_ext=True)
    scale = hand_scale_of(loose_hand)
    index_tip = loose_hand[INDEX_TIP]
    loose_hand[THUMB_TIP] = SimpleNamespace(x=index_tip.x + 0.10 * scale, y=index_tip.y, z=0.0)

    tight_hand = make_hand(index_ext=True)
    tight_hand[THUMB_TIP] = SimpleNamespace(x=index_tip.x + 0.02 * scale, y=index_tip.y, z=0.0)

    loose_ratio = _fit_pinch_ratio([loose_hand] * 5, default=0.35)
    tight_ratio = _fit_pinch_ratio([tight_hand] * 5, default=0.35)
    assert loose_ratio > tight_ratio
