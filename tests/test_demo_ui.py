"""The parts of the screen that can be checked without a screen.

Not much of a GUI is testable, but the two things that would embarrass the
exhibition are: a phase with no Japanese for it, so a visitor reads CAN_ABOVE
off the status line, and an overlay that draws nothing.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="the app refuses to import off Windows")


@pytest.fixture(scope="module")
def ui():
    spec = importlib.util.spec_from_file_location(
        "demo_ui_under_test", ROOT / "app" / "demo_ui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_taught_phase_has_words_for_a_visitor(ui):
    """The status line is read by people who do not know what PREGRASP is."""
    from so101.demo.trajectory import Trajectory

    taught = set()
    for path in sorted((ROOT / "data" / "demo_slots").glob("slot?.json")):
        taught.update(Trajectory.load(path).phases())
    missing = sorted(taught - set(ui.PHASE_WORDS))
    assert not missing, f"no Japanese for {missing}"


def test_every_colour_the_detector_knows_has_a_button(ui):
    from so101.dataset.scenes import CLASS_NAMES

    assert set(ui.COLOURS) == set(CLASS_NAMES), \
        "a colour the model can find but nobody can press is a support call"
    for colour in ui.COLOURS:
        assert colour in ui.JAPANESE and colour in ui.SWATCH
        assert colour in ui.BGR


def test_the_overlay_draws_the_slots_and_the_detections(ui):
    from so101.demo.slots import SlotMap

    class Detection:
        box = (340, 330, 360, 350)
        colour = "blue"
        confidence = 0.86

        @property
        def pixel(self):
            return (350.0, 340.0)

    slots = SlotMap(slots={"slot1": (200.0, 300.0), "slot2": (350.0, 340.0)},
                    acceptance_radius_px=18.0)
    blank = np.zeros((600, 800, 3), np.uint8)

    bare = ui.annotate(blank, [], slots)
    assert (bare != blank).any(), "the slots should be visible on their own"

    full = ui.annotate(blank, [Detection()], slots, chosen="slot2")
    assert (full != bare).any(), "a detection should add to it"


def test_the_overlay_survives_having_nothing_to_draw(ui):
    blank = np.zeros((600, 800, 3), np.uint8)
    assert ui.annotate(blank, None, None).shape == blank.shape
