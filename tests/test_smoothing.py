import pytest

from smoothing import OneEuroFilter


def test_first_sample_passes_through_unchanged():
    f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.012, dcutoff=1.0)
    assert f.filter(5.0, t=0.0) == 5.0


def test_smooths_toward_a_step_input_without_overshoot():
    f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.012, dcutoff=1.0)
    f.filter(0.0, t=0.0)
    t = 0.0
    prev = 0.0
    for _ in range(60):
        t += 1 / 30.0
        val = f.filter(100.0, t=t)
        # Monotonically approaching the target, never overshooting past it.
        assert prev - 1e-6 <= val <= 100.0 + 1e-6
        prev = val
    # After 2 seconds at 30fps it should have converged close to the target.
    assert val == pytest.approx(100.0, abs=1.0)


def test_constant_input_stays_constant():
    f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.012, dcutoff=1.0)
    t = 0.0
    for _ in range(10):
        val = f.filter(42.0, t=t)
        t += 1 / 30.0
    assert val == pytest.approx(42.0, abs=1e-6)


def test_reset_clears_history_so_next_sample_passes_through():
    f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.012, dcutoff=1.0)
    f.filter(0.0, t=0.0)
    f.filter(10.0, t=0.033)
    f.reset()
    # Behaves like a brand-new filter: first sample after reset passes
    # through unchanged instead of being smoothed against the pre-reset
    # history.
    assert f.filter(999.0, t=5.0) == 999.0


def test_higher_beta_reacts_faster_to_fast_movement():
    """Beta scales how much the filter 'loosens up' for fast movement --
    a higher beta should track a fast-moving signal more closely than a
    lower beta after the same number of samples."""
    slow = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.001, dcutoff=1.0)
    fast = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=1.0, dcutoff=1.0)
    t = 0.0
    slow.filter(0.0, t=t)
    fast.filter(0.0, t=t)
    for _ in range(5):
        t += 1 / 30.0
        slow_val = slow.filter(100.0, t=t)
        fast_val = fast.filter(100.0, t=t)
    assert fast_val > slow_val
