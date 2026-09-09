"""Camera config round-trip and spec parsing. No hardware needed."""

import json

from so101.camera.config import CameraSpec, load, save


def test_load_returns_empty_when_file_is_missing(tmp_path):
    assert load(tmp_path / "nope.json") == {}


def test_round_trip_drops_unset_fields(tmp_path):
    fpath = tmp_path / "cameras.json"
    save({"overhead": CameraSpec(name="icspring", width=1280, height=720,
                                 fps=20, fourcc="MJPG"),
          "wrist": CameraSpec(name="EMEET")}, fpath)

    raw = json.loads(fpath.read_text(encoding="utf-8"))
    assert raw["wrist"] == {"name": "EMEET"}, "unset fields should not be written"
    assert raw["overhead"]["fourcc"] == "MJPG"

    assert load(fpath) == {
        "overhead": CameraSpec("icspring", 1280, 720, 20, "MJPG"),
        "wrist": CameraSpec("EMEET"),
    }
