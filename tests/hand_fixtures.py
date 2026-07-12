
import math
from types import SimpleNamespace

from gesture_engine import (
    WRIST, THUMB_MCP, THUMB_IP, THUMB_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_TIP,
)

WRIST_XY = (0.5, 0.9)

# Radii (distance from wrist) used to place MCP/PIP/TIP for an extended vs
# curled finger. Extended gives tip/pip ~1.6 (comfortably above any
# reasonable finger_extended_margin); curled gives tip/pip ~0.67
# (comfortably below 1.0).
MCP_R = 0.15
PIP_R_EXT, TIP_R_EXT = 0.30, 0.48
PIP_R_CURL, TIP_R_CURL = 0.24, 0.16


def _p(x, y):
    return SimpleNamespace(x=float(x), y=float(y), z=0.0)


def _along(direction, radius, base=WRIST_XY):
    dx, dy = direction
    n = math.hypot(dx, dy)
    dx, dy = dx / n, dy / n
    return (base[0] + dx * radius, base[1] + dy * radius)


def _finger(lm, mcp_i, pip_i, tip_i, direction, extended):
    pip_r, tip_r = (PIP_R_EXT, TIP_R_EXT) if extended else (PIP_R_CURL, TIP_R_CURL)
    lm[mcp_i] = _p(*_along(direction, MCP_R))
    lm[pip_i] = _p(*_along(direction, pip_r))
    lm[tip_i] = _p(*_along(direction, tip_r))


# Roughly spread finger directions (unit-ish vectors from the wrist,
# pointing "up" in image space i.e. negative y).
DIR_INDEX = (-0.3, -1.0)
DIR_MIDDLE = (0.0, -1.0)
DIR_RING = (0.3, -1.0)
DIR_PINKY = (0.55, -0.9)
DIR_THUMB_SIDE = (-1.0, 0.0)
DIR_THUMB_UP = (0.0, -1.0)
DIR_THUMB_DOWN = (0.0, 1.0)
DIR_THUMB_SHAKA = (-1.0, -0.35)


def make_hand(index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False,
              thumb_ext=False, thumb_dir=DIR_THUMB_SIDE,
              pinch_index=False, pinch_middle=False):
    """Returns a 21-element landmark list matching the requested shape.

    pinch_index / pinch_middle place the thumb tip exactly on top of the
    index / middle tip (distance 0), which is always well inside any
    reasonable pinch_ratio threshold. Both True places the thumb tip at
    the midpoint of both tips (tripod pinch).
    """
    lm = [None] * 21
    lm[WRIST] = _p(*WRIST_XY)

    _finger(lm, INDEX_MCP, INDEX_PIP, INDEX_TIP, DIR_INDEX, index_ext)
    _finger(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, DIR_MIDDLE, middle_ext)
    _finger(lm, RING_MCP, RING_PIP, RING_TIP, DIR_RING, ring_ext)
    _finger(lm, PINKY_MCP, PINKY_PIP, PINKY_TIP, DIR_PINKY, pinky_ext)

    if pinch_index or pinch_middle:
        lm[THUMB_MCP] = _p(*_along(DIR_THUMB_SIDE, MCP_R))
        lm[THUMB_IP] = _p(*_along(DIR_THUMB_SIDE, 0.22))
        if pinch_index and pinch_middle:
            ix, iy = lm[INDEX_TIP].x, lm[INDEX_TIP].y
            mx, my = lm[MIDDLE_TIP].x, lm[MIDDLE_TIP].y
            lm[THUMB_TIP] = _p((ix + mx) / 2, (iy + my) / 2)
        elif pinch_index:
            lm[THUMB_TIP] = _p(lm[INDEX_TIP].x, lm[INDEX_TIP].y)
        else:
            lm[THUMB_TIP] = _p(lm[MIDDLE_TIP].x, lm[MIDDLE_TIP].y)
    else:
        _finger(lm, THUMB_MCP, THUMB_IP, THUMB_TIP, thumb_dir, thumb_ext)

    for i in range(21):
        if lm[i] is None:
            lm[i] = _p(*WRIST_XY)  # unused joints (CMC/DIP) -- never read by classify_gesture
    return lm
