"""Watch a training run's curves as they are written.

Two kinds of run in this project write two different things, so this reads both:

- the block detector (Ultralytics) appends a row to results.csv every epoch
- the grasp policy (LeRobot) prints one line to its log every log_freq steps

Either way it only reads what training already writes down, so nothing here can
slow a run or interfere with it, and it works just as well on a finished run.

    uv run scripts/watch_training.py                  # newest run, live window
    uv run scripts/watch_training.py --terminal       # text, no window
    uv run scripts/watch_training.py --source runs/policy/grasp_train.log
    uv run scripts/watch_training.py --save outputs/curves.png --once
"""

import argparse
import csv
import re
import time
from pathlib import Path

BAR_WIDTH = 28
SPARK = "_.-~^"

# (title, log scale, ((key, label), ...))
DETECTOR_PANELS = (
    ("metric", False, (("metrics/mAP50(B)", "mAP50"),
                       ("metrics/mAP50-95(B)", "mAP50-95"),
                       ("metrics/precision(B)", "precision"),
                       ("metrics/recall(B)", "recall"))),
    ("loss", False, (("train/box_loss", "train box"),
                     ("train/cls_loss", "train cls"),
                     ("val/box_loss", "val box"),
                     ("val/cls_loss", "val cls"))),
)
POLICY_PANELS = (
    ("loss", True, (("loss", "loss"),)),
    ("gradient norm", True, (("grdn", "grad norm"),)),
)

# INFO ... step:6K smpl:48K ep:94 epch:3.15 loss:0.291 grdn:22.207 lr:1.0e-05
POLICY_LINE = re.compile(r"\bstep:(\S+)\s+.*?\bloss:([0-9.eE+-]+)")
POLICY_FIELD = re.compile(r"\b(loss|grdn|lr|epch|updt_s|data_s):([0-9.eE+-]+)")
# tqdm's own counter, which is exact but restarts at zero on a resumed run.
PROGRESS_LINE = re.compile(r"\b(\d+)/(\d+) \[")


def big_number(text):
    """LeRobot shortens large counts, so 6K comes back as 6000."""
    scale = {"K": 1e3, "M": 1e6, "B": 1e9}.get(text[-1:].upper())
    return float(text[:-1]) * scale if scale else float(text)


def falls(key):
    """Whether lower is better, which decides what counts as the best value."""
    return "loss" in key or "grdn" in key


class Source:
    """A training run's numbers, however that run happens to write them.

    A run that was resumed has its history split across the logs of each
    attempt, so several paths can stand for one run: their rows are merged and
    sorted, and a step recorded twice keeps the later reading.
    """

    def __init__(self, path, kind):
        paths = [Path(p) for p in (path if isinstance(path, (list, tuple))
                                   else [path])]
        self.paths = paths
        self.path = paths[-1]
        self.kind = kind
        self.panels = POLICY_PANELS if kind == "policy" else DETECTOR_PANELS
        self.x_key = "step" if kind == "policy" else "epoch"
        self.total = None

    def read(self):
        if self.kind != "policy":
            return self._read_csv()
        merged, total = {}, None
        for path in self.paths:
            for row in self._read_log(path):
                merged[row["step"]] = row
            total = max(total or 0, self.total or 0) or None
        self.total = total
        return [merged[step] for step in sorted(merged)]

    def _read_csv(self):
        """results.csv, tolerating a row half-written as we read it."""
        try:
            with open(self.path, newline="", encoding="utf-8") as handle:
                raw = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            return []
        rows = []
        for row in raw:
            try:
                rows.append({key.strip(): float(value) for key, value in row.items()
                             if value not in (None, "")})
            except ValueError:
                continue      # a partial final line; complete on the next pass
        return [row for row in rows if "epoch" in row]

    def _read_log(self, path=None):
        """LeRobot's step lines, placed on a true step axis.

        Neither number in the log is usable on its own. The step count in the
        text is rounded past a thousand - 20K, not 20200 - and the progress bar
        beside it counts from zero even when the run resumed at step 20,000, so
        a resumed run would either climb in jumps or restart from the origin.

        Together they pin it down: the bar is exact and its spacing gives the
        logging interval, while the rounded text supplies the offset the bar
        has lost.
        """
        try:
            text = (path or self.path).read_text(encoding="utf-8",
                                                 errors="replace")
        except OSError:
            return []

        rows, bars, logged = [], [], []
        bar = None
        totals = None
        for line in text.replace(chr(13), chr(10)).splitlines():
            progress = PROGRESS_LINE.search(line)
            if progress:
                bar = int(progress.group(1))
                totals = int(progress.group(2))
            match = POLICY_LINE.search(line)
            if not match:
                continue
            rows.append({key: float(value)
                         for key, value in POLICY_FIELD.findall(line)})
            bars.append(bar)
            logged.append(big_number(match.group(1)))

        if not rows:
            return []

        steps = self._step_axis(bars, logged)
        for row, step in zip(rows, steps):
            row["step"] = step
        offset = steps[-1] - (bars[-1] if bars[-1] is not None else steps[-1])
        self.total = (totals + offset) if totals is not None else None
        return rows

    @staticmethod
    def _step_axis(bars, logged):
        """Absolute step for each logged line.

        The shift between the two is one unknown constant. LeRobot rounds the
        number it prints to the nearest thousand, so every line says the shift
        lies within 500 of its own estimate - and the true shift is the one
        candidate that no line contradicts.
        """
        usable = [(bar, value) for bar, value in zip(bars, logged)
                  if bar is not None]
        if not usable:
            return logged

        gaps = [b - a for (a, _), (b, _) in zip(usable, usable[1:]) if b > a]
        interval = min(gaps) if gaps else 1
        candidates = {round((value - bar) / interval) * interval
                      for bar, value in usable}
        tolerance = 500 if max(logged) >= 1000 else 0.5

        def disagreement(shift):
            misses = sum(abs(bar + shift - value) > tolerance
                         for bar, value in usable)
            drift = sum(abs(bar + shift - value) for bar, value in usable)
            return misses, drift

        shift = min(candidates, key=disagreement)
        return [bar + shift if bar is not None else value
                for bar, value in zip(bars, logged)]


def find_source(explicit=None):
    """The run that was written to most recently, unless one was named."""
    if explicit:
        paths = [Path(p) for p in (explicit if isinstance(explicit, (list, tuple))
                                   else [explicit])]
        for path in paths:
            if not path.is_file():
                raise SystemExit(f"{path} not found")
        kind = "policy" if paths[0].suffix == ".log" else "detector"
        return Source(paths if kind == "policy" else paths[0], kind)

    candidates = [(p, "detector") for p in Path("runs").rglob("results.csv")]
    for path in Path("runs").rglob("*.log"):
        tail = path.read_text(encoding="utf-8", errors="replace")[-20000:]
        if POLICY_LINE.search(tail):
            candidates.append((path, "policy"))
    if not candidates:
        raise SystemExit("No training output under runs/. Is a run going?")
    path, kind = max(candidates, key=lambda pair: pair[0].stat().st_mtime)
    return Source(path, kind)


def sparkline(values):
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return SPARK[0] * len(values)
    span = len(SPARK) - 1
    return "".join(SPARK[int(round((value - low) / (high - low) * span))]
                   for value in values)


def render_terminal(source, rows):
    latest = rows[-1]
    position = f"{source.x_key} {latest[source.x_key]:,.0f}"
    if source.total:
        position += f"/{source.total:,}"
    if source.kind == "policy":
        position += f"   epoch {latest.get('epch', 0):.2f}"
    elif "time" in latest:
        position += f"   elapsed {latest['time'] / 60:.1f} min"
    print(f"\n  {position}")

    for _, _, series in source.panels:
        for key, label in series:
            history = [row[key] for row in rows if key in row]
            if not history:
                continue
            value = history[-1]
            best = min(history) if falls(key) else max(history)
            marker = "  <- best" if value == best else f"  (best {best:.4g})"
            if not falls(key) and 0 <= value <= 1:
                filled = int(round(value * BAR_WIDTH))
                shape = f"[{'#' * filled}{'.' * (BAR_WIDTH - filled)}]"
            else:
                shape = sparkline(history[-BAR_WIDTH:])
            print(f"    {label:<10} {value:<9.4g} {shape}{marker}")


def render_plot(source, rows, axes):
    for axis, (title, logscale, series) in zip(axes, source.panels):
        axis.clear()
        for key, label in series:
            points = [(row[source.x_key], row[key]) for row in rows if key in row]
            if points:
                axis.plot([p[0] for p in points], [p[1] for p in points],
                          marker="o", markersize=2.5, linewidth=1.2, label=label)
        axis.set_ylabel(title)
        if logscale:
            axis.set_yscale("log")
        elif title == "metric":
            axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3, which="both")
        axis.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel(source.x_key)
    if source.total:
        axes[-1].set_xlim(0, source.total)

    position = rows[-1][source.x_key]
    headline = f"{source.x_key} {position:,.0f}"
    if source.total:
        headline += f"/{source.total:,}  ({100 * position / source.total:.0f}%)"
    key, label = source.panels[0][2][0]
    history = [row[key] for row in rows if key in row]
    if history:
        best = min(history) if falls(key) else max(history)
        headline += f"   {label} {history[-1]:.4g}   best {best:.4g}"
    axes[0].set_title(headline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="+", default=None,
                        help="results.csv, or one or more LeRobot training logs "
                             "of the same run (default: whichever is newest)")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--terminal", action="store_true",
                        help="print instead of opening a window")
    parser.add_argument("--once", action="store_true", help="render once and exit")
    parser.add_argument("--save", default=None, help="also write the plot here")
    args = parser.parse_args()

    source = find_source(args.source)
    print(f"  watching {', '.join(str(p) for p in source.paths)}  "
          f"({source.kind})")

    if args.terminal:
        seen = 0
        while True:
            rows = source.read()
            if len(rows) > seen:
                render_terminal(source, rows)
                seen = len(rows)
            if args.once:
                return
            time.sleep(args.interval)

    import matplotlib
    import matplotlib.pyplot as plt

    if args.once and args.save:
        matplotlib.use("Agg")     # no window needed just to write a file
    plt.ion()
    figure, axes = plt.subplots(len(source.panels), 1, figsize=(9, 7), sharex=True)
    axes = list(axes) if hasattr(axes, "__len__") else [axes]
    figure.canvas.manager.set_window_title(f"training - {source.path.stem}")

    seen = 0
    try:
        while True:
            rows = source.read()
            if rows and len(rows) != seen:
                render_plot(source, rows, axes)
                figure.tight_layout()
                if args.save:
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    figure.savefig(args.save, dpi=110)
                seen = len(rows)
            if args.once:
                if args.save:
                    print(f"  saved {args.save}")
                return
            # plt.pause keeps the window responsive; a bare sleep would freeze it.
            plt.pause(args.interval)
            if not plt.fignum_exists(figure.number):
                return
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
