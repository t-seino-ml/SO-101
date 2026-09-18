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


def test_a_panel_fills_exactly_the_space_it_was_given(ui):
    """A placeholder that is not panel-sized leaves a hole or overruns the card.

    It also has to exist at all: the panels used to be Labels, which size
    themselves in *characters* until they hold an image, and a 470-wide one
    asked for 470 characters and swallowed the window for as long as the
    detector took to load.
    """
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    try:
        root.withdraw()
        for size in (ui.TELEOP_VIEW, ui.PICK_VIEW):
            image = ui.placeholder(size)
            assert (image.width(), image.height()) == size
    finally:
        root.destroy()


@pytest.mark.parametrize("width,height", [(1560, 940), (1366, 768),
                                          (1920, 1080), (1280, 720)])
def test_the_pick_screen_fits_the_window_it_is_given(ui, width, height):
    """Three panels, the colours and STOP, all on screen at every size we meet.

    A panel pushed off the edge is not a cosmetic fault: the colour buttons go
    with it, and the only way to run the demo is then the command line.
    """
    plan = ui.pick_layout(width, height)
    cards = plan["row"]["cards"]
    assert len(cards) == 3
    assert cards[0][0] >= 0 and cards[-1][2] <= width
    for rect in (plan["select"], plan["stop"], *cards):
        x0, y0, x1, y1 = rect
        assert 0 <= x0 < x1 <= width, rect
        assert 0 <= y0 < y1 <= height, rect
    assert cards[0][3] <= plan["select"][1], "the cards overlap the colour bar"
    assert plan["select"][2] <= plan["stop"][0], "the colours overlap STOP"
    image_w, image_h = plan["row"]["image"]
    assert image_w >= 300 and image_h >= 220, \
        f"panels shrank to {image_w}x{image_h}"


@pytest.mark.parametrize("width,height", [(1560, 940), (1366, 768),
                                          (1920, 1080), (1280, 720)])
def test_the_teleop_screen_fits_the_window_it_is_given(ui, width, height):
    plan = ui.teleop_layout(width, height)
    cards = plan["row"]["cards"]
    assert len(cards) == 2
    for rect in (plan["panel"], *cards):
        x0, y0, x1, y1 = rect
        assert 0 <= x0 < x1 <= width, rect
        assert 0 <= y0 < y1 <= height, rect
    assert cards[0][3] <= plan["panel"][1]


def test_the_panels_take_the_space_rather_than_leaving_it_blank(ui):
    """The complaint that started the redesign: 情報量のない空白の部分.

    Whatever is left after the header and the lower card belongs to the camera
    views, so the check is that they actually claim most of the window.
    """
    plan = ui.pick_layout(1560, 940)
    used = sum((x1 - x0) for x0, _, x1, _ in plan["row"]["cards"])
    assert used >= 1560 * 0.9, f"the cards only use {used:.0f} of 1560 px"


def test_the_view_loop_reschedules_itself_even_when_it_throws(ui):
    """A display loop that stops rescheduling does not fail visibly.

    It happened: `clear()` destroyed the widgets, the 60 ms timer fired before
    the replacements existed, one TclError ended the loop, and every camera
    view on every screen stopped updating for the rest of the session with
    nothing on screen to say so.
    """
    import tkinter as tk

    class Exploding:
        """Enough of the app for the loop, with a panel that always throws."""

        views = {"side": object()}          # not a widget: `_show` will throw
        rescheduled = 0

        def after(self, _ms, _fn):
            self.rescheduled += 1

        def frame(self, _role):
            return None

        _show = ui.DemoApp._show
        _refresh_views = ui.DemoApp._refresh_views
        updates = None

    app = Exploding()
    app.updates = type("Q", (), {"put": staticmethod(lambda _u: None)})()
    app._refresh_views()
    assert app.rescheduled == 1, "the loop must survive its own failure"


def test_clearing_a_screen_drops_the_panels_it_owned(ui):
    """The race that killed the loop: views outliving the items they name."""
    import inspect

    source = inspect.getsource(ui.DemoApp.clear)
    before = source.index("self.views = {}")
    after = source.index('self.canvas.delete("all")')
    assert before < after, \
        "views must be dropped before the items are deleted, not after"


def test_a_colour_press_is_ignored_while_the_arm_is_moving(ui):
    """Canvas items have no disabled state, so the guard has to be in code.

    Greying the tiles is only paint. Without `colour_enabled` checked here, a
    second press mid-pick would start a second worker on the same serial port.
    """
    class Pressed:
        colour_enabled = False
        worker = None
        started = []
        start_pick = ui.DemoApp.start_pick

        def _set_buttons(self, _enabled):
            pass

        def start_worker(self, target):
            self.started.append(target)

    app = Pressed()
    app.start_pick("red")
    assert app.started == [], "a greyed tile must not start anything"

    app.colour_enabled = True
    app.start_pick("red")
    assert len(app.started) == 1


def test_stop_does_nothing_when_there_is_nothing_to_stop(ui):
    """Otherwise an idle press latches the flag and kills the next run."""
    class Idle:
        colour_enabled = True
        request_stop = ui.DemoApp.request_stop

        class stop_flag:
            set_called = False

            @classmethod
            def set(cls):
                cls.set_called = True

    Idle().request_stop()
    assert not Idle.stop_flag.set_called
