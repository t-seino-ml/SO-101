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
from tkinter import font as tkfont  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

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
SWATCH = {"red": "#c62828", "orange": "#ef6c00", "yellow": "#f9a825",
          "green": "#2e7d32", "blue": "#1565c0", "purple": "#6a1b9a"}
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

BACKGROUND = "#101418"
PANEL = "#1b2229"
TEXT = "#e8eef4"
MUTED = "#8b9bab"
ACCENT = "#4fa3ff"
GOOD = "#57c785"
BAD = "#ff6b6b"

VIEW_W, VIEW_H = 384, 288
DETECT_HZ = 3.0


@dataclass
class Update:
    """Something a worker wants the screen to say."""

    kind: str          # "state", "detail", "done", "failed", "slot"
    text: str = ""
    value: object = None


# -------------------------------------------------------------------------
# drawing
# -------------------------------------------------------------------------

def to_photo(image, width=VIEW_W, height=VIEW_H):
    """An OpenCV BGR frame as something Tk can put in a Label."""
    import cv2
    from PIL import Image, ImageTk

    if image is None:
        return None
    frame = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


def annotate(image, detections, slot_map, chosen=None):
    """The side view with what the detector sees drawn on it.

    The slots are drawn as the circles the run actually judges against, so a
    block sitting outside one is visibly outside one rather than mysteriously
    refused.
    """
    import cv2

    canvas = image.copy()
    if slot_map is not None:
        for name, (u, v) in sorted(slot_map.slots.items()):
            picked = (name == chosen)
            colour = (120, 220, 120) if picked else (120, 120, 120)
            radius = int(slot_map.acceptance_radius_px)
            cv2.circle(canvas, (int(u), int(v)), radius, colour,
                       3 if picked else 1)
            cv2.putText(canvas, name.replace("slot", "S"),
                        (int(u) - 12, int(v) - radius - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2 if picked else 1)
    for detection in detections or []:
        x0, y0, x1, y1 = (int(v) for v in detection.box)
        colour = BGR.get(detection.colour, (200, 200, 200))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(canvas, f"{detection.colour} {detection.confidence:.2f}",
                    (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1)
    return canvas


# -------------------------------------------------------------------------
# the app
# -------------------------------------------------------------------------

class DemoApp(tk.Tk):
    def __init__(self, follower_port="COM4", leader_port="COM3"):
        super().__init__()
        self.title("SO-101 デモ")
        self.configure(bg=BACKGROUND)
        self.geometry("1180x760")
        self.follower_port = follower_port
        self.leader_port = leader_port

        self.big = tkfont.Font(family="Yu Gothic UI", size=22, weight="bold")
        self.mid = tkfont.Font(family="Yu Gothic UI", size=14)
        self.small = tkfont.Font(family="Yu Gothic UI", size=11)

        self.updates = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = None
        self.cameras = None
        self.detector = None
        self.slot_map = None
        self.detections = []
        self._photos = {}          # kept alive; Tk drops images it cannot see
        self._last_detect = 0.0

        self.screen = None
        self.show_home()
        self.after(50, self._drain)
        self.after(60, self._refresh_views)
        self.protocol("WM_DELETE_WINDOW", self.close)
        threading.Thread(target=self._open_cameras, daemon=True).start()

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

    # -- screens ----------------------------------------------------------

    def clear(self):
        self.stop_worker()
        if self.screen is not None:
            self.screen.destroy()
        self.screen = tk.Frame(self, bg=BACKGROUND)
        self.screen.pack(fill="both", expand=True, padx=18, pady=18)
        self._photos.clear()
        return self.screen

    def show_home(self):
        page = self.clear()
        self.mode = "home"
        tk.Label(page, text="SO-101 デモ", font=self.big, bg=BACKGROUND,
                 fg=TEXT).pack(pady=(40, 6))
        tk.Label(page, text="どちらのモードで動かしますか", font=self.mid,
                 bg=BACKGROUND, fg=MUTED).pack(pady=(0, 40))

        row = tk.Frame(page, bg=BACKGROUND)
        row.pack()
        self._mode_button(
            row, "リーダ機で動かす",
            "もう一方のアームを手で動かすと\nロボットが同じ形について来ます",
            self.show_teleop).pack(side="left", padx=18)
        self._mode_button(
            row, "ブロックをつかむ",
            "色を選ぶと、カメラがその色を探して\nロボットが缶に入れます",
            self.show_pick).pack(side="left", padx=18)

        self.status = tk.Label(page, text="", font=self.small, bg=BACKGROUND,
                               fg=MUTED)
        self.status.pack(side="bottom", pady=10)

    def _mode_button(self, parent, title, blurb, command):
        box = tk.Frame(parent, bg=PANEL, highlightthickness=2,
                       highlightbackground="#2b3947", cursor="hand2")
        tk.Label(box, text=title, font=self.big, bg=PANEL, fg=TEXT).pack(
            padx=46, pady=(34, 10))
        tk.Label(box, text=blurb, font=self.small, bg=PANEL, fg=MUTED,
                 justify="center").pack(padx=46, pady=(0, 34))
        for widget in (box, *box.winfo_children()):
            widget.bind("<Button-1>", lambda _event: command())
        return box

    def _views(self, parent, roles, extra=None):
        """A row of live camera panels. Returns {name: Label}."""
        row = tk.Frame(parent, bg=BACKGROUND)
        row.pack(pady=(4, 10))
        labels = {}
        for name, caption in list(roles.items()) + list((extra or {}).items()):
            column = tk.Frame(row, bg=PANEL)
            column.pack(side="left", padx=8)
            tk.Label(column, text=caption, font=self.small, bg=PANEL,
                     fg=MUTED).pack(pady=(6, 2))
            view = tk.Label(column, bg="#000000", width=VIEW_W, height=VIEW_H)
            view.pack(padx=6, pady=(0, 6))
            labels[name] = view
        return labels

    def _header(self, page, title, back=True):
        bar = tk.Frame(page, bg=BACKGROUND)
        bar.pack(fill="x")
        tk.Label(bar, text=title, font=self.big, bg=BACKGROUND,
                 fg=TEXT).pack(side="left")
        if back:
            tk.Button(bar, text="← もどる", font=self.small, bg=PANEL, fg=TEXT,
                      relief="flat", padx=14, pady=6,
                      command=self.show_home).pack(side="right")
        return bar

    # -- teleoperation ----------------------------------------------------

    def show_teleop(self):
        page = self.clear()
        self.mode = "teleop"
        self._header(page, "リーダ機で動かす")
        self.views = self._views(page, {"side": "外付けカメラ",
                                        "wrist": "アームのカメラ"})
        self.state = tk.Label(page, text="準備しています…", font=self.mid,
                              bg=BACKGROUND, fg=ACCENT)
        self.state.pack(pady=(6, 2))
        self.status = tk.Label(page, text="", font=self.small, bg=BACKGROUND,
                               fg=MUTED)
        self.status.pack()
        tk.Label(page, text="もう一方のアームを手で動かしてください。"
                           "ロボットが同じ形について来ます。",
                 font=self.small, bg=BACKGROUND, fg=MUTED).pack(pady=(14, 0))
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
            self.updates.put(Update("state", "動かせます"))
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
        page = self.clear()
        self.mode = "pick"
        self._header(page, "ブロックをつかむ")
        self.views = self._views(
            page, {"side": "外付けカメラ", "wrist": "アームのカメラ"},
            extra={"detect": "カメラが見ているもの"})

        self.state = tk.Label(page, text="色をえらんでください", font=self.big,
                              bg=BACKGROUND, fg=ACCENT)
        self.state.pack(pady=(10, 2))
        self.status = tk.Label(page, text="", font=self.small, bg=BACKGROUND,
                               fg=MUTED)
        self.status.pack(pady=(0, 10))

        self.buttons = tk.Frame(page, bg=BACKGROUND)
        self.buttons.pack()
        self.colour_buttons = {}
        for colour in COLOURS:
            button = tk.Button(
                self.buttons, text=JAPANESE[colour], font=self.mid,
                bg=SWATCH[colour], fg="white", relief="flat", width=8,
                padx=10, pady=14, activebackground=SWATCH[colour],
                command=lambda c=colour: self.start_pick(c))
            button.pack(side="left", padx=6)
            self.colour_buttons[colour] = button

        self.stop_button = tk.Button(
            page, text="STOP", font=self.big, bg=BAD, fg="white",
            relief="flat", padx=40, pady=10, state="disabled",
            command=self.request_stop)
        self.stop_button.pack(pady=14)
        tk.Label(page, text="STOP はいまの動作の区切りで止まります（最大 3 秒）。"
                            "すぐ止めるときは電源です。",
                 font=self.small, bg=BACKGROUND, fg=MUTED).pack()

        try:
            self.slot_map = SlotMap.load()
            self.updates.put(Update("detail", str(self.slot_map)))
        except Exception as error:  # noqa: BLE001
            self.slot_map = None
            self.updates.put(Update("detail", f"Slot が未登録です: {error}"))
        threading.Thread(target=self._load_detector, daemon=True).start()

    def _set_buttons(self, enabled):
        for button in self.colour_buttons.values():
            button.configure(state="normal" if enabled else "disabled")
        self.stop_button.configure(state="disabled" if enabled else "normal")

    def start_pick(self, colour):
        if self.worker is not None and self.worker.is_alive():
            return
        self._set_buttons(False)
        self.chosen_slot = None
        self.start_worker(lambda: self._pick_worker(colour))

    def request_stop(self):
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

    def _drain(self):
        """Whatever the worker has said since last time. Main thread only."""
        try:
            while True:
                update = self.updates.get_nowait()
                if not hasattr(self, "state"):
                    continue
                if update.kind == "state":
                    self.state.configure(text=update.text, fg=ACCENT)
                elif update.kind == "detail":
                    self.status.configure(text=update.text)
                elif update.kind == "slot":
                    self.chosen_slot = update.value
                elif update.kind == "done":
                    self.state.configure(text=update.text, fg=GOOD)
                    if self.mode == "pick":
                        self._set_buttons(True)
                        self.chosen_slot = None
                elif update.kind == "failed":
                    self.state.configure(text=update.text, fg=BAD)
                    if self.mode == "pick":
                        self._set_buttons(True)
                        self.chosen_slot = None
        except queue.Empty:
            pass
        self.after(50, self._drain)

    def _refresh_views(self):
        if getattr(self, "views", None):
            for role in ("side", "wrist"):
                if role in self.views:
                    self._show(role, self.frame(role))
            if "detect" in self.views:
                self._show_detection()
        self.after(60, self._refresh_views)

    def _show(self, name, image):
        photo = to_photo(image)
        if photo is None:
            return
        self._photos[name] = photo         # Tk drops what it cannot see
        self.views[name].configure(image=photo, width=VIEW_W, height=VIEW_H)

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
                                      getattr(self, "chosen_slot", None)))

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
