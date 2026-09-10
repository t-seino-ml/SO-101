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


def big_number(text):
    """LeRobot shortens large counts, so 6K comes back as 6000."""
    scale = {"K": 1e3, "M": 1e6, "B": 1e9}.get(text[-1:].upper())
    return float(text[:-1]) * scale if scale else float(text)


def falls(key):
    """Whether lower is better, which decides what counts as the best value."""
    return "loss" in key or "grdn" in key


class Source:
    """A training run's numbers, however that run happens to write them."""

    def __init__(self, path, kind):
        self.path = Path(path)
        self.kind = kind
        self.panels = POLICY_PANELS if kind == "policy" else DETECTOR_PANELS
        self.x_key = "step" if kind == "policy" else "epoch"
        self.total = None

    def read(self):
        return self._read_log() if self.kind == "policy" else self._read_csv()

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

    def _read_log(self):
        """LeRobot's step lines.

        The step count in them is rounded once it passes a thousand - 6K, not
        6000 - which would make the x axis climb in visible jumps. The lines are
        printed at a fixed interval, though, so the first one gives that interval
        and the rest follow from their position. The rounded value is kept as a
        check: if it disagrees by more than one interval, trust the log.
        """
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        rows = []
        interval = None
        for line in text.replace("\r", "\n").splitlines():
            match = POLICY_LINE.search(line)
            if not match:
                continue
            row = {key: float(value) for key, value in POLICY_FIELD.findall(line)}
            logged = big_number(match.group(1))
            if interval is None:
                interval = logged
            counted = interval * (len(rows) + 1)
            row["step"] = counted if abs(counted - logged) <= interval else logged
            rows.append(row)

        # The tqdm bar in the same log carries the total, which the step lines
        # do not, and it is there from the first step rather than the first log.
        totals = re.findall(r"/(\d+) \[", text)
        self.total = int(totals[-1]) if totals else None
        return rows


def find_source(explicit=None):
    """The run that was written to most recently, unless one was named."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        return Source(path, "policy" if path.suffix == ".log" else "detector")

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
    parser.add_argument("--source", default=None,
                        help="results.csv or a LeRobot training log "
                             "(default: whichever run is newest)")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--terminal", action="store_true",
                        help="print instead of opening a window")
    parser.add_argument("--once", action="store_true", help="render once and exit")
    parser.add_argument("--save", default=None, help="also write the plot here")
    args = parser.parse_args()

    source = find_source(args.source)
    print(f"  watching {source.path}  ({source.kind})")

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
