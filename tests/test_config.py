import json

import pytest

from config import Config


def test_defaults_match_original_script_constants():
    cfg = Config()
    assert cfg.pinch_ratio == 0.35
    assert cfg.finger_extended_margin == 1.10
    assert cfg.stability_frames == 5
    assert cfg.drag_activation_delay_s == 0.25
    assert cfg.scroll_step == 18
    assert cfg.enable_dwell_click is False  # accessibility features opt-in, not opt-out


def test_load_missing_file_returns_defaults(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    cfg = Config.load(str(missing))
    assert cfg == Config()


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "config.json"
    original = Config(pinch_ratio=0.4, camera_index=2, enable_dwell_click=True)
    original.save(str(path))

    loaded = Config.load(str(path))
    assert loaded == original


def test_corrupt_json_falls_back_to_defaults(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text("{ this is not valid json ]")

    cfg = Config.load(str(path))
    assert cfg == Config()
    assert "Warning" in capsys.readouterr().out


def test_non_object_json_falls_back_to_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("[1, 2, 3]")
    cfg = Config.load(str(path))
    assert cfg == Config()


def test_unknown_keys_are_ignored_not_fatal(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"pinch_ratio": 0.4, "totally_made_up_key": 123}))

    cfg = Config.load(str(path))
    assert cfg.pinch_ratio == 0.4
    assert "unknown" in capsys.readouterr().out.lower()


def test_invalid_value_type_does_not_crash(tmp_path):
    path = tmp_path / "config.json"
    # pinch_ratio should be a float; dataclasses.replace() doesn't
    # type-check its keyword values, so this doesn't raise and doesn't
    # coerce -- the real contract under test is just that a malformed
    # single field can never crash the whole program on startup, and that
    # every OTHER field still holds its correct default.
    path.write_text(json.dumps({"pinch_ratio": {"nested": "object"}}))

    cfg = Config.load(str(path))  # must not raise
    assert cfg.camera_index == 0
    assert cfg.scroll_step == 18


def test_partial_calibration_style_update_preserves_other_fields(tmp_path):
    """Mirrors what calibration.py does: load, tweak two fields via
    dataclasses.replace, save, reload."""
    import dataclasses
    path = tmp_path / "config.json"
    cfg = Config.load(str(path))  # file doesn't exist yet -> defaults
    calibrated = dataclasses.replace(cfg, pinch_ratio=0.28, finger_extended_margin=1.15)
    calibrated.save(str(path))

    reloaded = Config.load(str(path))
    assert reloaded.pinch_ratio == 0.28
    assert reloaded.finger_extended_margin == 1.15
    assert reloaded.scroll_step == cfg.scroll_step  # everything else untouched
