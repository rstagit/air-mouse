
import argparse
import math
import os
import sys
import time
import urllib.request

import cv2
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import Config, CONFIG_PATH
from camera import ThreadedCamera, list_available_cameras
from smoothing import OneEuroFilter
from gesture_engine import (
    classify_gesture, normalized_to_screen, draw_hand_overlay,
    GestureStabilizer, DeltaTracker, VideoClock,
    is_pinching, inter_hand_pinch_distance, lm_xy, INDEX_TIP,
    MOVE, GRAB, RIGHT_CLICK, DOUBLE_CLICK, SCROLL_UP, SCROLL_DOWN,
    MEDIA_TOGGLE, LOCK_SCREEN, PRECISION, THREE_SWEEP, TWO_HAND_ZOOM, LASER,
)
import platform_actions as actions
from rendering import draw_hud, draw_legend, LaserPointerOverlay
from calibration import run_calibration

WINDOW_NAME = "Hand Mouse Control"


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

pyautogui.FAILSAFE = True   
pyautogui.PAUSE = 0.0       


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return MODEL_PATH

    print(f"Hand landmark model not found. Downloading to:\n  {MODEL_PATH}")

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(100, block_num * block_size * 100 // total_size)
        sys.stdout.write(f"\r  {pct}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, _progress)
        print("\nDownload complete.")
    except Exception as e:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)  # don't leave a partial/corrupt file behind
        raise RuntimeError(
            f"Failed to download hand landmark model from {MODEL_URL}.\n"
            f"Check your internet connection, or download it manually and "
            f"place it at {MODEL_PATH}.\nOriginal error: {e}"
        )
    return MODEL_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Hand-controlled desktop mouse.")
    parser.add_argument("--camera", type=int, default=None,
                         help="Camera index to use (overrides config.json)")
    parser.add_argument("--list-cameras", action="store_true",
                         help="List available camera indices and exit")
    parser.add_argument("--calibrate", action="store_true",
                         help="Run the calibration wizard even if already calibrated")
    parser.add_argument("--config", type=str, default=CONFIG_PATH,
                         help="Path to config.json")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_cameras:
        print("Probing camera indices 0-7 (this can take a few seconds)...")
        found = list_available_cameras()
        print("Available camera indices:", found if found else "none found")
        return

    cfg = Config.load(args.config)
    if args.camera is not None:
        cfg.camera_index = args.camera

    screen_w, screen_h = pyautogui.size()

    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,  # the primary hand drives the cursor; a second hand enables two-hand zoom
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
    clock = VideoClock()

    try:
        cam = ThreadedCamera(cfg.camera_index, cfg.cam_width, cfg.cam_height, cfg.cam_fps).start()
    except RuntimeError as e:
        print(f"Error: {e}\nTip: run with --list-cameras to see available camera indices.")
        landmarker.close()
        return

    cv2.namedWindow(WINDOW_NAME)

    first_run = not os.path.exists(args.config)
    if first_run or args.calibrate:
        cfg = run_calibration(cam, landmarker, clock, cfg, cfg.cam_width, cfg.cam_height, WINDOW_NAME)
        try:
            cfg.save(args.config)
        except OSError as e:
            print(f"Warning: could not save {args.config} ({e}); calibration won't persist next run.")

    laser_overlay = LaserPointerOverlay()

    stabilizer = GestureStabilizer(cfg.stability_frames)
    sweep_tracker = DeltaTracker(cfg.sweep_min_dx, cfg.sweep_max_window_s, "right", "left")
    zoom_tracker = DeltaTracker(cfg.zoom_min_ddist, cfg.zoom_max_window_s, "in", "out")
    filter_x = OneEuroFilter(freq=cfg.cam_fps, mincutoff=cfg.oneeuro_mincutoff,
                              beta=cfg.oneeuro_beta, dcutoff=cfg.oneeuro_dcutoff)
    filter_y = OneEuroFilter(freq=cfg.cam_fps, mincutoff=cfg.oneeuro_mincutoff,
                              beta=cfg.oneeuro_beta, dcutoff=cfg.oneeuro_dcutoff)

    
    prev_gesture = MOVE
    last_desktop_switch_time = 0.0
    last_scroll_time = 0.0
    last_zoom_time = 0.0
    dragging = False
    grab_start_time = None
    lock_gesture_start_time = None
    last_lock_time = 0.0
    last_cursor = None
    paused = False

    presentation_mode = False
    laser_mode = False
    dwell_enabled = cfg.enable_dwell_click
    show_legend = False

    dwell_anchor = None
    dwell_start_time = None
    dwell_fired = False

    prev_t = time.time()
    fps_smooth = 0.0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)  # mirror view so movement feels natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, clock.next_ms())

            confirmed = MOVE
            now = time.time()
            mode_flags = []
            if presentation_mode:
                mode_flags.append("PRESENTATION")
            if dwell_enabled:
                mode_flags.append("DWELL")

            if result.hand_landmarks:
                hands = result.hand_landmarks
                primary = hands[0]
                draw_hand_overlay(frame, primary, cfg.cam_width, cfg.cam_height)
                if len(hands) > 1:
                    draw_hand_overlay(frame, hands[1], cfg.cam_width, cfg.cam_height, color=(200, 120, 0))

                if laser_mode:
                    mode_flags.append("LASER")
                    stabilizer.reset()
                    sweep_tracker.reset()
                    zoom_tracker.reset()
                    lock_gesture_start_time = None
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                        grab_start_time = None

                    idx_pt = lm_xy(primary, INDEX_TIP)
                    sx, sy = normalized_to_screen(idx_pt[0], idx_pt[1], screen_w, screen_h, cfg.frame_margin)
                    sx = filter_x.filter(sx, t=now)
                    sy = filter_y.filter(sy, t=now)
                    laser_overlay.show()
                    laser_overlay.move_to(sx, sy)
                    confirmed = LASER
                else:
                    laser_overlay.hide()

                    two_hand_zoom_active = (
                        cfg.enable_zoom and len(hands) == 2
                        and is_pinching(hands[0], cfg.pinch_ratio)
                        and is_pinching(hands[1], cfg.pinch_ratio)
                    )

                    if two_hand_zoom_active:
                        confirmed = TWO_HAND_ZOOM
                        stabilizer.reset()
                        sweep_tracker.reset()
                        lock_gesture_start_time = None
                        if not paused:
                            if dragging:
                                pyautogui.mouseUp()
                                dragging = False
                                grab_start_time = None
                            d = inter_hand_pinch_distance(hands[0], hands[1])
                            direction = zoom_tracker.update(d, now)
                            if direction is not None and (now - last_zoom_time) > cfg.zoom_cooldown_s:
                                actions.zoom(direction, cfg.zoom_scroll_step)
                                last_zoom_time = now
                    else:
                        zoom_tracker.reset()
                        raw_gesture, track_pt = classify_gesture(primary, cfg)
                        confirmed = stabilizer.push(raw_gesture)

                        sx, sy = normalized_to_screen(track_pt[0], track_pt[1], screen_w, screen_h, cfg.frame_margin)
                        sx = filter_x.filter(sx, t=now)
                        sy = filter_y.filter(sy, t=now)

                        if confirmed != THREE_SWEEP:
                            sweep_tracker.reset()

                        if not paused:

                            held_long_enough = (
                                confirmed == GRAB and grab_start_time is not None
                                and (now - grab_start_time) >= cfg.drag_activation_delay_s
                            )
                            if confirmed in (MOVE, PRECISION) or held_long_enough:
                                target_x, target_y = sx, sy
                                if confirmed == PRECISION and last_cursor is not None:

                                    target_x = last_cursor[0] + cfg.precision_factor * (sx - last_cursor[0])
                                    target_y = last_cursor[1] + cfg.precision_factor * (sy - last_cursor[1])
                                if last_cursor is None or math.hypot(target_x - last_cursor[0], target_y - last_cursor[1]) > cfg.cursor_deadzone_px:
                                    pyautogui.moveTo(target_x, target_y)
                                    last_cursor = (target_x, target_y)


                            if confirmed == GRAB and not dragging:
                                pyautogui.mouseDown()
                                dragging = True
                                grab_start_time = now
                            elif confirmed != GRAB and dragging:
                                pyautogui.mouseUp()
                                dragging = False
                                grab_start_time = None


                            if dragging and grab_start_time is not None and (now - grab_start_time) > cfg.grab_max_hold_s:
                                pyautogui.mouseUp()
                                dragging = False
                                grab_start_time = None


                            if confirmed == RIGHT_CLICK and prev_gesture != RIGHT_CLICK:
                                pyautogui.click(button="right")
                            if confirmed == DOUBLE_CLICK and prev_gesture != DOUBLE_CLICK:
                                pyautogui.doubleClick()
                            if confirmed == MEDIA_TOGGLE and prev_gesture != MEDIA_TOGGLE:
                                actions.toggle_media_playback(cfg.media_playpause_key)

                            if confirmed in (SCROLL_UP, SCROLL_DOWN) and (now - last_scroll_time) > cfg.scroll_interval_s:
                                pyautogui.scroll(cfg.scroll_step if confirmed == SCROLL_UP else -cfg.scroll_step)
                                last_scroll_time = now

                            if confirmed == THREE_SWEEP:
                                direction = sweep_tracker.update(track_pt[0], now)
                                if direction is not None and (now - last_desktop_switch_time) > cfg.desktop_switch_cooldown_s:
                                    if presentation_mode:
                                        actions.navigate_slide(direction)
                                    else:
                                        actions.switch_desktop(direction)
                                    last_desktop_switch_time = now

                            if confirmed == LOCK_SCREEN:
                                if lock_gesture_start_time is None:
                                    lock_gesture_start_time = now
                                elif (now - lock_gesture_start_time) >= cfg.lock_hold_s \
                                        and (now - last_lock_time) > cfg.lock_cooldown_s:
                                    actions.lock_screen()
                                    last_lock_time = now
                                    lock_gesture_start_time = None
                            else:
                                lock_gesture_start_time = None


                            if dwell_enabled and confirmed in (MOVE, PRECISION):
                                if dwell_anchor is None or math.hypot(sx - dwell_anchor[0], sy - dwell_anchor[1]) > cfg.dwell_radius_px:
                                    dwell_anchor = (sx, sy)
                                    dwell_start_time = now
                                    dwell_fired = False
                                elif not dwell_fired and (now - dwell_start_time) >= cfg.dwell_time_s:
                                    pyautogui.click()
                                    dwell_fired = True
                            else:
                                dwell_anchor = None
                                dwell_fired = False

                prev_gesture = confirmed
            else:
                # No hand visible: release any held button so we never get stuck dragging.
                laser_overlay.hide()
                if dragging and not paused:
                    pyautogui.mouseUp()
                    dragging = False
                    grab_start_time = None
                sweep_tracker.reset()
                zoom_tracker.reset()
                lock_gesture_start_time = None
                dwell_anchor = None
                dwell_fired = False
                prev_gesture = MOVE

            laser_overlay.pump()

            # --- HUD ---
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)
            status = "PAUSED" if paused else confirmed
            color = (0, 0, 255) if paused else (0, 200, 0)
            draw_hud(frame, cfg.cam_height, status, color, fps_smooth, mode_flags)
            if show_legend:
                draw_legend(frame, cfg.cam_width, cfg.cam_height)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                paused = not paused
                if paused and dragging:
                    pyautogui.mouseUp()
                    dragging = False
                    grab_start_time = None
            elif key == ord('h'):
                show_legend = not show_legend
            elif key == ord('p'):
                presentation_mode = not presentation_mode
            elif key == ord('l'):
                laser_mode = not laser_mode
                if not laser_mode:
                    laser_overlay.hide()
                    filter_x.reset()
                    filter_y.reset()
                    last_cursor = None
            elif key == ord('d'):
                dwell_enabled = not dwell_enabled
                dwell_anchor = None
                dwell_fired = False
            elif key == ord('c'):
                cfg = run_calibration(cam, landmarker, clock, cfg, cfg.cam_width, cfg.cam_height, WINDOW_NAME)
                try:
                    cfg.save(args.config)
                except OSError as e:
                    print(f"Warning: could not save {args.config} ({e}).")
                stabilizer.reset()

    finally:
        if dragging:
            pyautogui.mouseUp()
        cam.stop()
        landmarker.close()
        laser_overlay.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
