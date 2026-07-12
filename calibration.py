
import dataclasses
import statistics
import time

import cv2
import mediapipe as mp

from gesture_engine import (
    WRIST, MIDDLE_MCP, INDEX_PIP, INDEX_TIP, THUMB_TIP,
    lm_xy, dist, hand_scale_of, draw_hand_overlay,
)

SAMPLE_SECONDS = 1.5
INSTRUCTION_COLOR = (0, 200, 255)


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _collect_samples(cam, landmarker, clock, instruction, seconds, cam_width, cam_height, window_name):

    samples = []
    start = time.time()
    while time.time() - start < seconds:
        frame = cam.read()
        if frame is None:
            time.sleep(0.005)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, clock.next_ms())

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            draw_hand_overlay(frame, landmarks, cam_width, cam_height)
            samples.append(landmarks)

        remaining = max(0.0, seconds - (time.time() - start))
        cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INSTRUCTION_COLOR, 2)
        cv2.putText(frame, f"{remaining:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INSTRUCTION_COLOR, 2)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    return samples


def run_calibration(cam, landmarker, clock, cfg, cam_width, cam_height, window_name="Hand Mouse Control"):
    """Runs the two-step wizard and returns an updated Config. If a hand
    wasn't reliably detected during either step, returns cfg unchanged."""
    print("Calibration: hold your hand where the camera can see it clearly.")

    open_samples = _collect_samples(
        cam, landmarker, clock,
        "Calibration 1/2: hold hand OPEN, fingers spread",
        SAMPLE_SECONDS, cam_width, cam_height, window_name,
    )
    pinch_samples = _collect_samples(
        cam, landmarker, clock,
        "Calibration 2/2: PINCH thumb and index together",
        SAMPLE_SECONDS, cam_width, cam_height, window_name,
    )

    if not open_samples or not pinch_samples:
        print("Calibration: hand not detected reliably enough; keeping current settings.")
        return cfg

    new_margin = _fit_finger_extended_margin(open_samples, cfg.finger_extended_margin)
    new_pinch = _fit_pinch_ratio(pinch_samples, cfg.pinch_ratio)

    print(f"Calibration done: finger_extended_margin={new_margin:.3f} (was {cfg.finger_extended_margin:.3f}), "
          f"pinch_ratio={new_pinch:.3f} (was {cfg.pinch_ratio:.3f})")

    return dataclasses.replace(cfg, finger_extended_margin=new_margin, pinch_ratio=new_pinch)


def _fit_finger_extended_margin(open_samples, default):
    """Uses the index finger's tip/wrist vs pip/wrist distance ratio while
    the hand is held open. The new margin is set a bit below the observed
    ratio (so the same open finger reliably still counts as extended),
    clamped to a sane range."""
    ratios = []
    for lm in open_samples:
        wrist = lm_xy(lm, WRIST)
        tip = lm_xy(lm, INDEX_TIP)
        pip = lm_xy(lm, INDEX_PIP)
        pip_d = dist(wrist, pip)
        if pip_d > 1e-6:
            ratios.append(dist(wrist, tip) / pip_d)
    if not ratios:
        return default
    return _clamp(statistics.median(ratios) * 0.85, 1.02, 1.4)


def _fit_pinch_ratio(pinch_samples, default):
    """Uses the observed thumb-to-index distance (normalized by hand
    scale) while actively pinching. The new threshold is set above the
    observed distance (headroom for an imperfect pinch), clamped to a
    sane range."""
    ratios = [
        dist(lm_xy(lm, THUMB_TIP), lm_xy(lm, INDEX_TIP)) / hand_scale_of(lm)
        for lm in pinch_samples
    ]
    if not ratios:
        return default
    return _clamp(statistics.median(ratios) * 1.6, 0.15, 0.55)
