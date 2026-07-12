
import dataclasses
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


@dataclasses.dataclass
class Config:
    
    camera_index: int = 0
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30


    frame_margin: float = 0.12


    pinch_ratio: float = 0.35            
    double_click_pinch_ratio: float = 0.22  
    finger_extended_margin: float = 1.10  
    thumb_vertical_ratio: float = 0.5     

    # --- Stabilizer / cursor ---
    stability_frames: int = 5
    cursor_deadzone_px: int = 2

    # --- Click / drag timing ---
    drag_activation_delay_s: float = 0.25
    grab_max_hold_s: float = 6.0

    # --- Scroll ---
    scroll_step: int = 18
    scroll_interval_s: float = 0.06

    
    desktop_switch_cooldown_s: float = 1.0
    sweep_min_dx: float = 0.16
    sweep_max_window_s: float = 0.5

    
    lock_hold_s: float = 1.5
    lock_cooldown_s: float = 3.0

    
    zoom_min_ddist: float = 0.06     
    zoom_max_window_s: float = 0.6
    zoom_cooldown_s: float = 0.15
    zoom_scroll_step: int = 2

    
    precision_factor: float = 0.35   # cursor moves at this fraction of normal speed while held

    
    media_playpause_key: str = "playpause"  # pyautogui key name; try "space" if your player ignores media keys

    
    enable_dwell_click: bool = False
    dwell_time_s: float = 1.2
    dwell_radius_px: int = 12

    
    oneeuro_mincutoff: float = 1.0
    oneeuro_beta: float = 0.012
    oneeuro_dcutoff: float = 1.0

    
    enable_right_click: bool = True
    enable_double_click: bool = True
    enable_media_control: bool = True
    enable_zoom: bool = True

    def save(self, path=CONFIG_PATH):
        """Writes this config to disk as JSON. Raises OSError on failure
        (caller decides whether that should be fatal)."""
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path=CONFIG_PATH):
        """Loads config.json, merged over the defaults. Never raises --
        any problem (missing file, bad JSON, unknown/invalid keys) falls
        back to defaults with a printed warning, so a broken config file
        can never stop the program from starting."""
        defaults = cls()
        if not os.path.exists(path):
            return defaults

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not read {path} ({e}); using defaults.")
            return defaults

        if not isinstance(data, dict):
            print(f"Warning: {path} does not contain a JSON object; using defaults.")
            return defaults

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - valid_fields)
        if unknown:
            print(f"Warning: ignoring unknown config keys in {path}: {unknown}")
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        try:
            return dataclasses.replace(defaults, **filtered)
        except TypeError as e:
            print(f"Warning: invalid value in {path} ({e}); using defaults.")
            return defaults
