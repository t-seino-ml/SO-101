"""The exhibition, on a screen instead of a command line.

Two modes, chosen from the home screen:

  リーダ機で動かす   the leader arm drives the follower and nothing else
                     happens. No model, no detector, no trajectory - just the
                     two arms and both camera views.

  ブロックをつかむ   today's slot system. Both camera views, a third showing
                     what the detector sees, and six colour buttons. Press one
                     and the arm fetches that block and drops it in the can,
                     saying what it is doing as it goes.

Tkinter, because it is in the standard library and talks to the cameras and the
serial ports directly. There is no server and no browser: one fewer thing to be
down when a room full of school students is waiting.

Each screen is drawn onto a single canvas rather than built out of widgets. Tk
has no rounded corners, no gradients and no icons, and a wall of grey Frames is
not what a visitor should walk up to. `theme` holds the drawing; this file holds
the layout and the machinery.

Everything that moves the arm runs on a worker thread and reports back through a
queue; Tk is touched only from the main thread. The robot is connected when a
mode needs it and disconnected when the mode is left, so the arm is never live
while nobody is looking at it.

    uv run app/demo_ui.py

*** STOP stops at the end of the waypoint in progress, which is at most three
seconds. It is not the emergency stop. Those are unchanged and physical:
scripts/torque_off.py in a terminal, and the power. ***
"""

from so101.platform import require_windows

require_windows()

import queue  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import tkinter as tk  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from tkinter import font as tkfont  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Running this as a script would put `app/` on the path anyway; spelt out so the
# tests, which load this file by path, get the same `theme`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import theme  # noqa: E402
from so101.demo.slots import SlotMap, steady_centre  # noqa: E402
from so101.demo.trajectory import (  # noqa: E402
    Trajectory,
    Unsafe,
    connect,
    freeze,
    joints_of,
    play,
    read_limits,
)

COLOURS = ("red", "orange", "yellow", "green", "blue", "purple")
#: What each colour looks like on screen, and in the overlay. BGR for OpenCV.
SWATCH = dict(theme.BLOCK)
BGR = {"red": (40, 40, 200), "orange": (30, 110, 220), "yellow": (40, 190, 230),
       "green": (70, 160, 70), "blue": (190, 110, 40), "purple": (150, 70, 110)}
JAPANESE = {"red": "あか", "orange": "オレンジ", "yellow": "きいろ",
            "green": "みどり", "blue": "あお", "purple": "むらさき"}

#: What each waypoint is, in words a visitor can follow.
PHASE_WORDS = {
    "HOME": "はじめの位置へ",
    "PREGRASP": "ブロックの上まで移動しています",
    "GRASP": "ブロックまで下ろしています",
    "CLOSE": "つかんでいます",
    "LIFT": "持ち上げています",
    "TRANSFER": "缶へ運んでいます",
    "CAN_ABOVE": "缶の上まで来ました",
    "DROP": "缶に入れています",
    "RETURN": "はじめの位置へ戻ります",
}

#: Fallback panel sizes, used only before the window knows how big it is. The
#: real ones come out of the layout functions below.
TELEOP_VIEW = (600, 450)
PICK_VIEW = (470, 352)
DETECT_HZ = 3.0

MARGIN = 28
GAP = 18


@dataclass
class Update:
    """Something a worker wants the screen to say."""

    kind: str          # "state", "detail", "done", "failed", "slot"
    text: str = ""
    value: object = None


# -------------------------------------------------------------------------
# layout
# -------------------------------------------------------------------------

def camera_row(width, top, bottom, count, margin=MARGIN, gap=GAP, pad=12,
               head=46, ratio=0.75):
    """Rects for `count` camera cards: as large as the space allows, centred.

    The panels are the reason anybody is looking at the screen, so they take
    whatever room is left over rather than a fixed size - the first version had
    a fixed size and a window full of nothing around it.
    """
    card_w = (width - 2 * margin - (count - 1) * gap) / count
    image_w = card_w - 2 * pad
    image_h = image_w * ratio
    room = (bottom - top) - head - 2 * pad
    if image_h > room:                      # short window: height decides
        image_h = max(120, room)
        image_w = image_h / ratio
    image_w, image_h = int(image_w), int(image_h)
    card_w, card_h = image_w + 2 * pad, head + image_h + 2 * pad
    total = count * card_w + (count - 1) * gap
    left = (width - total) / 2
    cards = [(left + index * (card_w + gap), top,
              left + index * (card_w + gap) + card_w, top + card_h)
             for index in range(count)]
    return {"cards": cards, "image": (image_w, image_h), "head": head,
            "pad": pad}


def _bottom_split(height, row, wanted, margin=MARGIN, gap=GAP):
    """Put the lower panel at the foot of the window and share out the slack.

    If the cards leave more room than the panel wants, the leftover is split
    above and below rather than left as a band of empty navy under the buttons.
    """
    row_bottom = row["cards"][0][3]
    space = height - margin - (row_bottom + gap)
    if space <= wanted:
        return row, (row_bottom + gap, height - margin)
    shift = (space - wanted) / 2
    row = dict(row, cards=[(x0, y0 + shift, x1, y1 + shift)
                           for x0, y0, x1, y1 in row["cards"]])
    return row, (height - margin - wanted, height - margin)


def pick_layout(width, height, header=78, panel=260, stop=250):
    """Three camera cards over a colour bar and a stop button."""
    top = MARGIN + header
    row = camera_row(width, top, height - MARGIN - panel - GAP, 3)
    row, (y0, y1) = _bottom_split(height, row, panel)
    return {"row": row,
            "header": (MARGIN, MARGIN, width - MARGIN, MARGIN + header),
            "select": (MARGIN, y0, width - MARGIN - stop - GAP, y1),
            "stop": (width - MARGIN - stop, y0, width - MARGIN, y1)}


def teleop_layout(width, height, header=78, panel=150):
    """Two camera cards over one instruction panel."""
    top = MARGIN + header
    row = camera_row(width, top, height - MARGIN - panel - GAP, 2)
    row, (y0, y1) = _bottom_split(height, row, panel)
    return {"row": row,
            "header": (MARGIN, MARGIN, width - MARGIN, MARGIN + header),
            "panel": (MARGIN, y0, width - MARGIN, y1)}


def spaced(text):
    """Letter-spacing, which Tk fonts do not have, done by hand.

    Only for the small uppercase tags on the cards; anywhere else it would be
    unreadable.
    """
    return " ".join(text)


# -------------------------------------------------------------------------
# drawing the camera frames
# -------------------------------------------------------------------------

def to_photo(image, size):
    """An OpenCV BGR frame as something Tk can put on the canvas."""
    import cv2
    from PIL import Image, ImageTk

    if image is None:
        return None
    frame = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


def placeholder(size, text="カメラを準備しています…"):
    """A panel-sized image to hold the panel open until a frame arrives."""
    import cv2
    import numpy as np

    canvas = np.full((size[1], size[0], 3), 12, np.uint8)
    cv2.putText(canvas, text, (18, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (120, 150, 180), 1)
    return to_photo(canvas, size)


_FONTS = {}


def _japanese_font(size):
    """A TrueType face for the overlay labels, or None if Windows has moved them.

    OpenCV's putText cannot draw かな, and `red 0.86` is the wrong label for a
    screen aimed at high-school students. If no face loads the caller falls back
    to the English name rather than drawing nothing.
    """
    if size in _FONTS:
        return _FONTS[size]
    from PIL import ImageFont

    _FONTS[size] = None
    for name in ("YuGothM.ttc", "meiryo.ttc", "msgothic.ttc", "YuGothR.ttc"):
        try:
            _FONTS[size] = ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
            break
        except OSError:
            continue
    return _FONTS[size]


def _plates(canvas, plates, size):
    """Japanese name plates on the detection boxes, drawn through PIL."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    font = _japanese_font(size)
    if font is None:
        for x, y, text, colour in plates:
            cv2.putText(canvas, text, (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, size / 34, colour, 2)
        return canvas
    picture = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(picture)
    pad = max(4, size // 3)
    for x, y, text, colour in plates:
        box = draw.textbbox((0, 0), text, font=font)
        width, height = box[2] - box[0], box[3] - box[1]
        top = max(0, y - height - 2 * pad - 4)
        draw.rounded_rectangle(
            (x, top, x + width + 2 * pad, top + height + 2 * pad),
            radius=pad, fill=(colour[2], colour[1], colour[0]))
        draw.text((x + pad, top + pad - box[1]), text, font=font, fill="white")
    return cv2.cvtColor(np.array(picture), cv2.COLOR_RGB2BGR)


def annotate(image, detections, slot_map, chosen=None, shown_width=None):
    """The side view with what the detector sees drawn on it.

    The slots are drawn as the circles the run actually judges against, so a
    block sitting outside one is visibly outside one rather than mysteriously
    refused.

    Everything is drawn on the full-size frame and the panel then shrinks it, so
    `shown_width` says how wide it will end up and the lettering is scaled to
    survive the trip. Drawn at frame scale, an 800-wide camera in a 380-wide
    panel put the colour names on screen at eight pixels.
    """
    import cv2

    canvas = image.copy()
    scale = max(1.0, image.shape[1] / (shown_width or image.shape[1]))
    thin = max(1, int(round(scale)))
    thick = max(2, int(round(3 * scale)))
    if slot_map is not None:
        for name, (u, v) in sorted(slot_map.slots.items()):
            picked = (name == chosen)
            colour = (120, 220, 120) if picked else (110, 110, 120)
            radius = int(slot_map.acceptance_radius_px)
            cv2.circle(canvas, (int(u), int(v)), radius, colour,
                       thick if picked else thin)
            cv2.putText(canvas, name.replace("slot", "S"),
                        (int(u) - int(12 * scale),
                         int(v) - radius - int(8 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, colour,
                        thick if picked else thin)
    plates = []
    for detection in detections or []:
        x0, y0, x1, y1 = (int(value) for value in detection.box)
        colour = BGR.get(detection.colour, (200, 200, 200))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thick)
        plates.append((x0, y0,
                       JAPANESE.get(detection.colour, detection.colour), colour))
    return _plates(canvas, plates, int(17 * scale)) if plates else canvas


# -------------------------------------------------------------------------
# the app
# -------------------------------------------------------------------------

class DemoApp(tk.Tk):
    def __init__(self, follower_port="COM4", leader_port="COM3"):
        super().__init__()
        self.title("SO-101 デモ")
        self.configure(bg=theme.DEEP)
        width = min(1560, self.winfo_screenwidth() - 80)
        height = min(940, self.winfo_screenheight() - 140)
        self.geometry(f"{width}x{height}+40+20")
        self.bind("<F11>", self._fullscreen)
        self.bind("<Escape>",
                  lambda _event: self.attributes("-fullscreen", False))
        self.follower_port = follower_port
        self.leader_port = leader_port

        self.f_title = tkfont.Font(family="Yu Gothic UI", size=25, weight="bold")
        self.f_state = tkfont.Font(family="Yu Gothic UI", size=19, weight="bold")
        self.f_card = tkfont.Font(family="Yu Gothic UI", size=12, weight="bold")
        self.f_body = tkfont.Font(family="Yu Gothic UI", size=11)
        self.f_tile = tkfont.Font(family="Yu Gothic UI", size=12, weight="bold")
        self.f_tag = tkfont.Font(family="Consolas", size=8)
        self.f_stop = tkfont.Font(family="Arial", size=20, weight="bold")

        self.canvas = tk.Canvas(self, bg=theme.DEEP, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.updates = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = None
        self.cameras = None
        self.detector = None
        self.slot_map = None
        self.detections = []
        self.views = {}            # role -> canvas image item
        self.items = {}            # name -> canvas item, for the text we change
        self.tiles = {}
        self.view_size = PICK_VIEW
        self.colour_enabled = False
        self.mode = "home"
        self.detail_text = ""
        self.pill_at = self.detail_at = self.pill_left = None
        self._photos = {}          # kept alive; Tk drops images it cannot see
        self._last_detect = 0.0
        self._live = set()         # panels that have shown a real frame
        self._size = (width, height)
        self._resize_job = None

        self.show_home()
        self.canvas.bind("<Configure>", self._resized)
        self.after(50, self._drain)
        self.after(60, self._refresh_views)
        self.protocol("WM_DELETE_WINDOW", self.close)
        threading.Thread(target=self._start_up, daemon=True).start()

    def _fullscreen(self, _event=None):
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))

    def _resized(self, event):
        """Redraw at the new size, once the dragging has stopped.

        Never while the arm is moving: redrawing goes through `clear()`, and
        `clear()` stops the worker. A window nudged mid-pick would otherwise
        abandon the block in the jaws.
        """
        if (event.width, event.height) == self._size or event.width < 400:
            return
        self._size = (event.width, event.height)
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(180, self._redraw)

    def _redraw(self):
        self._resize_job = None
        if self.worker is not None and self.worker.is_alive():
            return
        {"home": self.show_home, "teleop": self.show_teleop,
         "pick": self.show_pick}[self.mode]()

    def _start_up(self):
        """Open the cameras, then load the detector, before either is asked for.

        The detector takes several seconds the first time - CUDA context, graph
        setup, the warmup pass - and doing it when the pick screen opens meant
        the pick screen opened onto nothing. It is loaded here instead, while
        somebody is still reading the home screen.
        """
        self._open_cameras()
        try:
            self._load_detector()
        except Exception as error:  # noqa: BLE001
            self.updates.put(Update("detail", f"検出モデルを読めません: {error}"))

    # -- shared resources -------------------------------------------------

    def _open_cameras(self):
        from so101.camera import CameraSet

        try:
            cameras = CameraSet.from_config()
            cameras.start()
            cameras.wait_for_frames(timeout=25)
            self.cameras = cameras
            self.updates.put(Update("detail", "カメラ準備完了"))
        except Exception as error:  # noqa: BLE001
            self.updates.put(Update("detail", f"カメラを開けません: {error}"))

    def _load_detector(self):
        if self.detector is not None:
            return self.detector
        from so101.policy import BlockDetector

        self.updates.put(Update("detail", "検出モデルを読み込んでいます…"))
        detector = BlockDetector(table_frame=None)
        detector.warmup()
        self.detector = detector
        self.updates.put(Update("detail", "検出モデル準備完了"))
        return detector

    def frame(self, role):
        if self.cameras is None or role not in self.cameras.streams:
            return None
        got = self.cameras.streams[role].read()
        return None if got is None else got.image

    # -- the canvas -------------------------------------------------------

    def clear(self):
        """Blank the canvas and lay the background back down.

        `views` is emptied before the items go, not after. The refresh timer
        fires every 60 ms and would otherwise find it pointing at items that no
        longer exist.
        """
        self.stop_worker()
        self.views = {}
        self.items = {}
        self.tiles = {}
        self.detections = []
        self.colour_enabled = False
        self.canvas.delete("all")
        self._photos.clear()
        self._live.clear()
        self.pill_at = self.detail_at = self.pill_left = None
        width, height = self.size()
        theme.gradient(self.canvas, 0, 0, width, height, theme.DEEP, theme.NIGHT)
        theme.starfield(self.canvas, 0, 0, width, height,
                        count=max(60, width * height // 16000))
        return width, height

    def size(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 400 or height < 300:     # before the window is mapped
            width, height = self._size
        return width, height

    def _header(self, rect, title, instruction, back=True):
        """Title, a rule, the instruction line, the status pill and the way out.

        Everything sits on one row. The back button used to hang below it and
        landed on top of the first camera card, which is what happens when a
        header is given a height and then something is put outside it.
        """
        x0, y0, x1, y1 = rect
        middle = (y0 + y1) / 2
        self.canvas.create_text(x0, middle, text=title, anchor="w",
                                font=self.f_title, fill=theme.TEXT)
        rule = x0 + self.f_title.measure(title) + 26
        self.canvas.create_line(rule, middle - 17, rule, middle + 17,
                                fill=theme.EDGE)
        right = x1
        if back:
            self._button("back", x1 - 108, middle - 17, 108, 34, "← もどる",
                         self.f_body, self.show_home, text_colour=theme.SUBTLE)
            right = x1 - 108 - 14
        self.pill_at = (right, middle - 16)
        self.detail_at = (rule + 22, middle)
        self.items["detail"] = self.canvas.create_text(
            rule + 22, middle, text="", anchor="w", font=self.f_body,
            fill=theme.SUBTLE)
        self._pill("準備しています…", theme.WARN)
        self._detail(self.detail_text or instruction)

    def _pill(self, text, colour):
        """The status badge, redrawn rather than edited - it changes width."""
        self.canvas.delete("pill")
        if self.pill_at is None:
            return
        x, y = self.pill_at
        text = self._fit(text, self.f_body, 340)
        width = self.f_body.measure(text) + 44
        self.pill_left = x - width
        theme.pill(self.canvas, x - width, y, text, self.f_body, dot=colour,
                   colour=theme.TEXT, tags="pill")

    def _fit(self, text, font, pixels):
        """As much of `text` as fits, with an ellipsis if it did not.

        The detail line carries whatever a worker last said - a SlotMap
        repr, a driver traceback - and unclipped it runs straight under the
        status pill.
        """
        if font.measure(text) <= pixels:
            return text
        while text and font.measure(text + "…") > pixels:
            text = text[:-1]
        return text + "…"

    def _camera_card(self, rect, image_size, head, pad, role, caption, tag):
        x0, y0, x1, y1 = rect
        theme.rounded(self.canvas, x0, y0, x1, y1, 16, fill=theme.CARD,
                      outline=theme.EDGE)
        self.canvas.create_line(x0 + 1, y0 + head, x1 - 1, y0 + head,
                                fill=theme.EDGE_SOFT)
        theme.camera_glyph(self.canvas, x0 + pad + 2, y0 + (head - 22) / 2)
        self.canvas.create_text(x0 + pad + 34, y0 + head / 2, text=caption,
                                anchor="w", font=self.f_card, fill=theme.TEXT)
        self.canvas.create_text(x1 - pad - 2, y0 + head / 2, text=spaced(tag),
                                anchor="e", font=self.f_tag, fill=theme.FAINT)
        blank = placeholder(image_size)
        self._photos[role] = blank
        self.views[role] = self.canvas.create_image(
            x0 + pad, y0 + head + pad, image=blank, anchor="nw")
        self.items[f"live:{role}"] = self.canvas.create_text(
            x0 + pad + 10, y0 + head + pad + image_size[1] - 14,
            text=spaced("NO SIGNAL"), anchor="w", font=self.f_tag,
            fill=theme.FAINT)

    def _button(self, name, x, y, width, height, text, font, command,
                fill="#16213c", outline=theme.EDGE, text_colour=None,
                radius=None):
        """A rounded rectangle that answers to the mouse."""
        tag = f"btn:{name}"
        theme.rounded(self.canvas, x, y, x + width, y + height,
                      height / 2 if radius is None else radius, fill=fill,
                      outline=outline, tags=tag)
        self.canvas.create_text(x + width / 2, y + height / 2, text=text,
                                font=font, fill=text_colour or theme.TEXT,
                                tags=tag)
        self.canvas.tag_bind(tag, "<Button-1>", lambda _event: command())
        self.canvas.tag_bind(
            tag, "<Enter>", lambda _event: self.canvas.configure(cursor="hand2"))
        self.canvas.tag_bind(
            tag, "<Leave>", lambda _event: self.canvas.configure(cursor=""))
        return tag

    # -- home -------------------------------------------------------------

    def show_home(self):
        width, height = self.clear()
        self.mode = "home"
        self.canvas.create_text(width / 2, height * 0.19, text="SO-101 デモ",
                                font=self.f_title, fill=theme.TEXT)
        self.canvas.create_text(width / 2, height * 0.19 + 34,
                                text=spaced("ROBOT ARM DEMONSTRATION"),
                                font=self.f_tag, fill=theme.FAINT)

        card_w = min(430, (width - 3 * MARGIN) / 2)
        card_h = min(300, height * 0.38)
        top = height * 0.33
        left = (width - 2 * card_w - GAP * 2) / 2
        self._mode_card((left, top, left + card_w, top + card_h),
                        "teleop", "01 / TELEOP", "リーダ機で動かす",
                        "もう一方のアームを手で動かすと\n"
                        "ロボットが同じ形について来ます",
                        theme.ACCENT, self.show_teleop)
        second = left + card_w + GAP * 2
        self._mode_card((second, top, second + card_w, top + card_h),
                        "pick", "02 / PICK & PLACE", "ブロックをつかむ",
                        "色を選ぶと、カメラがその色を探して\n"
                        "ロボットが缶に入れます",
                        theme.GOOD, self.show_pick)

        self.items["detail"] = self.canvas.create_text(
            width / 2, height - 36, text=self.detail_text, font=self.f_body,
            fill=theme.FAINT)

    def _mode_card(self, rect, key, tag, title, blurb, accent, command):
        """One of the two choices on the home screen.

        `key` is the canvas tag and `tag` is the label, and they are separate
        arguments because the label has spaces in it. A canvas tag is a Tcl
        list: tagging six items "mode:01 / TELEOP" tagged them "mode:01", "/"
        and "TELEOP", `tag_bind` on the whole string matched nothing, and
        neither card could be clicked.
        """
        x0, y0, x1, y1 = rect
        name = f"mode:{key}"
        box = theme.rounded(self.canvas, x0, y0, x1, y1, 18, fill=theme.CARD,
                            outline=theme.EDGE, tags=name)
        self.canvas.create_rectangle(x0 + 28, y0 + 32, x0 + 60, y0 + 35,
                                     fill=accent, width=0, tags=name)
        self.canvas.create_text(x0 + 28, y0 + 56, text=spaced(tag), anchor="w",
                                font=self.f_tag, fill=theme.FAINT, tags=name)
        self.canvas.create_text(x0 + 28, y0 + 96, text=title, anchor="w",
                                font=self.f_state, fill=theme.TEXT, tags=name)
        self.canvas.create_text(x0 + 28, y0 + 132, text=blurb, anchor="nw",
                                font=self.f_body, fill=theme.SUBTLE,
                                justify="left", tags=name)
        self.canvas.create_text(x1 - 28, y1 - 26, text="えらぶ  →", anchor="se",
                                font=self.f_card, fill=accent, tags=name)
        self.canvas.tag_bind(name, "<Button-1>", lambda _event: command())
        self.canvas.tag_bind(
            name, "<Enter>",
            lambda _event: (self.canvas.itemconfigure(box, outline=accent),
                            self.canvas.configure(cursor="hand2")))
        self.canvas.tag_bind(
            name, "<Leave>",
            lambda _event: (self.canvas.itemconfigure(box, outline=theme.EDGE),
                            self.canvas.configure(cursor="")))

    # -- teleoperation ----------------------------------------------------

    def show_teleop(self):
        width, height = self.clear()
        self.mode = "teleop"
        plan = teleop_layout(width, height)
        self._header(plan["header"], "リーダ機で動かす",
                     "もう一方のアームを手で動かしてください")
        row = plan["row"]
        self.view_size = row["image"]
        for rect, (role, caption, tag) in zip(
                row["cards"], (("side", "外付けカメラ", "CAM 01 / EXTERNAL"),
                               ("wrist", "アームのカメラ", "CAM 02 / ARM"))):
            self._camera_card(rect, row["image"], row["head"], row["pad"],
                              role, caption, tag)

        x0, y0, x1, y1 = plan["panel"]
        theme.rounded(self.canvas, x0, y0, x1, y1, 16, fill=theme.CARD,
                      outline=theme.EDGE)
        self.canvas.create_text(x0 + 26, y0 + 26, text=spaced("01 / TELEOP"),
                                anchor="w", font=self.f_tag, fill=theme.FAINT)
        self.items["state"] = self.canvas.create_text(
            x0 + 26, y0 + 62, text="準備しています…", anchor="w",
            font=self.f_state, fill=theme.WARN)
        self.canvas.create_text(
            x0 + 26, y0 + 100,
            text="ロボットは手で持っているアームと同じ形について来ます。"
                 "ゆっくり動かしてください。",
            anchor="w", font=self.f_body, fill=theme.SUBTLE)
        self._button("release", x1 - 200, y0 + 40, 170, 58, "もどって解放",
                     self.f_card, self.show_home, fill="#2c1b23",
                     outline=theme.BAD, text_colour=theme.BAD, radius=14)
        self.start_worker(self._teleop_worker)

    def _teleop_worker(self):
        from so101.hardware import bus_patch  # noqa: F401
        from so101.hardware import resolve as resolve_port
        from so101.hardware import tuning
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.teleoperators import make_teleoperator_from_config
        from lerobot.teleoperators.so_leader import SO101LeaderConfig

        tuning.install(verbose=False)
        follower = resolve_port([self.follower_port])[0]
        leader = resolve_port([self.leader_port])[0]
        robot = teleop = None
        try:
            self.updates.put(Update("state", "つないでいます…"))
            teleop = make_teleoperator_from_config(
                SO101LeaderConfig(port=leader, id="leader"))
            # The leader first: if it will not talk there is no session, and no
            # reason for the follower's torque to have come on to find out.
            for attempt in range(3):
                try:
                    teleop.connect()
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 2:
                        raise
                    time.sleep(1.5)
            robot = make_robot_from_config(
                SO101FollowerConfig(port=follower, id="follower"))
            connect(robot, follower, log=lambda line: None)
            self.updates.put(Update("done", "動かせます"))
            self.updates.put(Update(
                "detail", "もう一方のアームを手で動かしてください"))

            misses = 0
            while not self.stop_flag.is_set():
                started = time.perf_counter()
                try:
                    robot.send_action(teleop.get_action())
                    misses = 0
                except Exception as error:  # noqa: BLE001
                    misses += 1
                    if misses >= 5:
                        raise
                    self.updates.put(Update(
                        "detail", f"通信が乱れました（{misses}/5）"))
                    time.sleep(0.1)
                    continue
                time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - started)))
        except Exception as error:  # noqa: BLE001
            self.updates.put(Update("failed", f"{type(error).__name__}: {error}"))
        finally:
            for device in (teleop, robot):
                try:
                    if device is not None:
                        device.disconnect()
                except Exception:  # noqa: BLE001
                    pass
            self.updates.put(Update("detail", "アームを解放しました"))

    # -- picking ----------------------------------------------------------

    def show_pick(self):
        width, height = self.clear()
        self.mode = "pick"
        plan = pick_layout(width, height)
        self._header(plan["header"], "ブロックをつかむ",
                     "色を選ぶと、カメラがその色を探してロボットが缶に入れます")
        row = plan["row"]
        self.view_size = row["image"]
        for rect, (role, caption, tag) in zip(
                row["cards"], (("side", "外付けカメラ", "CAM 01 / EXTERNAL"),
                               ("wrist", "アームのカメラ", "CAM 02 / ARM"),
                               ("detect", "カメラが見ているもの",
                                "DETECTION"))):
            self._camera_card(rect, row["image"], row["head"], row["pad"],
                              role, caption, tag)

        self._select_card(plan["select"])
        self._stop_card(plan["stop"])
        self._set_buttons(True)
        self._state("色の選択待ち", theme.ACCENT)

        try:
            self.slot_map = SlotMap.load()
            self.updates.put(Update(
                "detail", f"Slot を {len(self.slot_map.slots)} か所"
                          f"読み込みました（半径 "
                          f"{self.slot_map.acceptance_radius_px:.0f} px）"))
        except Exception as error:  # noqa: BLE001
            self.slot_map = None
            self.updates.put(Update("detail", f"Slot が未登録です: {error}"))
        threading.Thread(target=self._load_detector, daemon=True).start()

    def _select_card(self, rect):
        x0, y0, x1, y1 = rect
        theme.rounded(self.canvas, x0, y0, x1, y1, 16, fill=theme.CARD,
                      outline=theme.EDGE)
        self.canvas.create_text(x0 + 26, y0 + 26,
                                text=spaced("01 / SELECT COLOR"), anchor="w",
                                font=self.f_tag, fill=theme.FAINT)
        self.items["state"] = self.canvas.create_text(
            x0 + 26, y0 + 62, text="色をえらんでください", anchor="w",
            font=self.f_state, fill=theme.TEXT)

        top = y0 + 92
        pad, gap = 26, 12
        tile_w = (x1 - x0 - 2 * pad - 5 * gap) / 6
        tile_h = max(78, y1 - 18 - top)
        for index, colour in enumerate(COLOURS):
            self.tiles[colour] = self._colour_tile(
                x0 + pad + index * (tile_w + gap), top, tile_w, tile_h, colour)

    def _colour_tile(self, x, y, width, height, colour):
        """One block button: the cube, then its name.

        A cube rather than a flat swatch, because the thing on the table is a
        cube and a visitor should be able to match the button to the object
        without reading anything.
        """
        tag = f"tile:{colour}"
        swatch = theme.BLOCK[colour]
        box = theme.rounded(self.canvas, x, y, x + width, y + height, 14,
                            fill="#151e37", outline=theme.EDGE, tags=tag)
        theme.cube(self.canvas, x + width / 2, y + height * 0.40,
                   min(46, height * 0.42, width * 0.42), swatch, tags=tag)
        label = self.canvas.create_text(
            x + width / 2, y + height - 22, text=JAPANESE[colour],
            font=self.f_tile, fill=theme.TEXT, tags=tag)
        self.canvas.tag_bind(tag, "<Button-1>",
                             lambda _event, c=colour: self.start_pick(c))
        self.canvas.tag_bind(
            tag, "<Enter>",
            lambda _event: (
                self.canvas.itemconfigure(
                    box, outline=swatch if self.colour_enabled else theme.EDGE),
                self.canvas.configure(
                    cursor="hand2" if self.colour_enabled else "")))
        self.canvas.tag_bind(
            tag, "<Leave>",
            lambda _event: (self.canvas.itemconfigure(box, outline=theme.EDGE),
                            self.canvas.configure(cursor="")))
        return {"box": box, "label": label}

    def _stop_card(self, rect):
        x0, y0, x1, y1 = rect
        bottom = y1 - 46          # room under it for what STOP does not do
        middle = (x0 + x1) / 2
        self.stop_box = theme.rounded(
            self.canvas, x0, y0, x1, bottom, 16, fill=theme.STOP_BOTTOM,
            outline=theme.STOP_TOP, tags="stop")
        theme.stop_glyph(self.canvas, middle - 13,
                         y0 + (bottom - y0) * 0.22, 26, tags="stop")
        self.canvas.create_text(middle, y0 + (bottom - y0) * 0.56, text="STOP",
                                font=self.f_stop, fill="white", tags="stop")
        self.canvas.create_text(middle, y0 + (bottom - y0) * 0.78,
                                text="動作を停止", font=self.f_body,
                                fill="#ffd9d6", tags="stop")
        self.canvas.tag_bind("stop", "<Button-1>",
                             lambda _event: self.request_stop())
        self.canvas.tag_bind(
            "stop", "<Enter>", lambda _event: self.canvas.configure(
                cursor="" if self.colour_enabled else "hand2"))
        self.canvas.tag_bind(
            "stop", "<Leave>", lambda _event: self.canvas.configure(cursor=""))
        self.canvas.create_text(
            middle, bottom + 16,
            text="STOP はいまの動作の区切りで止まります（最大 3 秒）",
            font=self.f_tag, fill=theme.FAINT)
        self.canvas.create_text(middle, bottom + 34,
                                text="すぐ止めるときは電源です",
                                font=self.f_tag, fill=theme.FAINT)

    def _set_buttons(self, enabled):
        """Colours live or greyed, and STOP banked or lit.

        A canvas item has no disabled state, so this is the tiles' own fill plus
        `colour_enabled`, which is what the click handler actually checks. STOP
        goes dull while nothing is running for the same reason a lit STOP on an
        idle machine is a lie.
        """
        self.colour_enabled = enabled
        for tile in self.tiles.values():
            self.canvas.itemconfigure(tile["box"],
                                      fill="#151e37" if enabled else "#101830")
            self.canvas.itemconfigure(
                tile["label"], fill=theme.TEXT if enabled else theme.FAINT)
        if hasattr(self, "stop_box"):
            self.canvas.itemconfigure(
                self.stop_box,
                fill="#3a2029" if enabled else theme.STOP_BOTTOM,
                outline="#54303a" if enabled else theme.STOP_TOP)

    def start_pick(self, colour):
        if not self.colour_enabled:
            return
        if self.worker is not None and self.worker.is_alive():
            return
        self._set_buttons(False)
        self.chosen_slot = None
        self.start_worker(lambda: self._pick_worker(colour))

    def request_stop(self):
        if self.colour_enabled:         # nothing is running to stop
            return
        self.stop_flag.set()
        self.updates.put(Update("state", "止めています…"))

    def _pick_worker(self, colour):
        from so101.hardware import bus_patch  # noqa: F401
        from so101.hardware import resolve as resolve_port
        from so101.hardware import tuning

        robot = None
        try:
            if self.slot_map is None:
                raise Unsafe("Slot が登録されていません。"
                             "calibrate_demo_slots.py を先に実行してください")
            detector = self._load_detector()
            if self.cameras is None:
                raise Unsafe("カメラが開いていません")

            self.updates.put(Update("state", f"{JAPANESE[colour]}を探しています"))
            centre, detail = steady_centre(
                detector, self.cameras.streams["side"], colour,
                log=lambda line: None, slot_map=self.slot_map)
            if centre is None:
                self.updates.put(Update("failed", detail["why"]))
                return
            if self.slot_map.ambiguous(centre):
                order = self.slot_map.ranked(centre)
                self.updates.put(Update(
                    "failed", f"{order[0][0]} と {order[1][0]} の"
                              f"ちょうど中間にあります"))
                return
            name, distance = self.slot_map.nearest(centre)
            if name is None:
                self.updates.put(Update(
                    "failed", f"いちばん近い Slot まで {distance:.0f} px です。"
                              f"ブロックが Slot の上にありません"))
                return
            self.updates.put(Update("slot", value=name))
            self.updates.put(Update(
                "detail", f"{JAPANESE[colour]}を {name} で見つけました"))

            trajectory = Trajectory.for_slot(self.slot_map.number(name))
            port = resolve_port([self.follower_port])[0]
            limits = read_limits(port)
            trajectory.check(limits,
                             start={n: limits[n]["deg"] for n in limits
                                    if n != "gripper"},
                             log=lambda line: None)

            self.updates.put(Update("state", "うごきます"))
            tuning.install(verbose=False)
            from lerobot.robots import make_robot_from_config
            from lerobot.robots.so_follower import SO101FollowerConfig

            robot = make_robot_from_config(
                SO101FollowerConfig(port=port, id="follower"))
            connect(robot, port, log=lambda line: None)

            def confirm(point):
                if self.stop_flag.is_set():
                    return False
                self.updates.put(Update(
                    "state", PHASE_WORDS.get(point.phase, point.phase)))
                return True

            def watch(record):
                self.updates.put(Update(
                    "detail",
                    f"{record['phase']}  ずれ {record['worst_error_deg']:+.1f}°"
                    f"  負荷 {record['load']}  {record['temperature_c']}℃"))

            play(robot, trajectory, log=lambda line: None, confirm=confirm,
                 on_sample=watch, limits=limits)
            self.updates.put(Update("done", "できました"))
        except Unsafe as error:
            self.updates.put(Update("failed", str(error)))
            if robot is not None:
                freeze(robot, log=lambda line: None)
        except Exception as error:  # noqa: BLE001
            self.updates.put(Update(
                "failed", f"{type(error).__name__}: {error}"))
            if robot is not None:
                freeze(robot, log=lambda line: None)
        finally:
            if robot is not None:
                try:
                    robot.disconnect()
                except Exception:  # noqa: BLE001
                    pass
            self.updates.put(Update("detail", "アームを解放しました"))

    # -- the loops the screen runs on -------------------------------------

    def start_worker(self, target):
        self.stop_flag.clear()
        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()

    def stop_worker(self, timeout=6.0):
        self.stop_flag.set()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=timeout)
        self.worker = None

    def _state(self, text, colour):
        """The pill in the header and the big line in the lower card, together.

        One call, because they must never disagree: a header saying it is
        holding a block while the card still says to choose a colour is worse
        than either on its own.
        """
        self._pill(text, colour)
        if "state" in self.items:
            self.canvas.itemconfigure(self.items["state"], text=text,
                                      fill=colour)

    def _detail(self, text):
        self.detail_text = text
        if "detail" not in self.items:
            return
        if self.detail_at is not None and self.pill_left is not None:
            room = max(160, self.pill_left - self.detail_at[0] - 20)
        else:
            room = self.size()[0] - 2 * MARGIN
        self.canvas.itemconfigure(self.items["detail"],
                                  text=self._fit(text, self.f_body, room))

    def _drain(self):
        """Whatever the worker has said since last time. Main thread only.

        Rescheduled in a `finally`, like `_refresh_views`, and for the same
        reason: a loop that stops rescheduling itself when something goes wrong
        does not fail visibly, it just quietly stops being a program.
        """
        try:
            self._drain_once()
        except Exception:  # noqa: BLE001 - never let the loop die
            pass
        finally:
            self.after(50, self._drain)

    def _drain_once(self):
        try:
            while True:
                update = self.updates.get_nowait()
                if update.kind == "state":
                    self._state(update.text, theme.ACCENT)
                elif update.kind == "detail":
                    self._detail(update.text)
                elif update.kind == "slot":
                    self.chosen_slot = update.value
                elif update.kind == "done":
                    self._state(update.text, theme.GOOD)
                    if self.mode == "pick":
                        self._set_buttons(True)
                        self.chosen_slot = None
                elif update.kind == "failed":
                    self._state(update.text, theme.BAD)
                    if self.mode == "pick":
                        self._set_buttons(True)
                        self.chosen_slot = None
        except queue.Empty:
            pass

    def _refresh_views(self):
        """Put the newest frame in every panel.

        The reschedule is in a `finally` because it used to be the last
        statement: one TclError from an item deleted a moment earlier ended the
        loop, and every camera view on every screen stopped updating for the
        rest of the session with no error anybody would see.
        """
        try:
            for role in ("side", "wrist"):
                if role in self.views:
                    self._show(role, self.frame(role))
            if "detect" in self.views:
                self._show_detection()
        except tk.TclError:
            pass       # an item went away mid-update; the next screen has its own
        except Exception as error:  # noqa: BLE001
            self.updates.put(Update("detail", f"表示: {error}"))
        finally:
            self.after(60, self._refresh_views)

    def _show(self, name, image):
        if name not in self.views:
            return
        photo = to_photo(image, self.view_size)
        if photo is None:
            return
        self._photos[name] = photo         # Tk drops what it cannot see
        self.canvas.itemconfigure(self.views[name], image=photo)
        badge = self.items.get(f"live:{name}")
        if badge is not None and name not in self._live:
            self._live.add(name)
            self.canvas.itemconfigure(badge, text=spaced("LIVE"),
                                      fill="#cfe0ff")

    def _show_detection(self):
        image = self.frame("side")
        if image is None:
            return
        now = time.perf_counter()
        if self.detector is not None and now - self._last_detect > 1 / DETECT_HZ:
            self._last_detect = now
            try:
                self.detections = self.detector.detect(image)
            except Exception:  # noqa: BLE001 - a frame is not worth a crash
                pass
        self._show("detect", annotate(image, self.detections, self.slot_map,
                                      getattr(self, "chosen_slot", None),
                                      shown_width=self.view_size[0]))

    def close(self):
        self.stop_worker()
        if self.cameras is not None:
            try:
                self.cameras.stop()
            except Exception:  # noqa: BLE001
                pass
        self.destroy()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SO-101 展示デモの画面")
    parser.add_argument("--follower-port", default="COM4")
    parser.add_argument("--leader-port", default="COM3")
    args = parser.parse_args()
    DemoApp(args.follower_port, args.leader_port).mainloop()


if __name__ == "__main__":
    main()
