"""Watch a training run's curves as they are written.

Ultralytics appends a row to results.csv after every epoch, so this tails that
file rather than instrumenting the training loop: nothing here can slow the run
down or interfere with it, and it works just as well on a run that is already
finished.

    uv run scripts/watch_training.py                # live plot window
    uv run scripts/watch_training.py --terminal     # text, no window
    uv run scripts/watch_training.py --save outputs/curves.png --once
"""

import argparse
import csv
import time
from pathlib import Path

METRICS = (
    ("metrics/mAP50(B)", "mAP50"),
    ("metrics/mAP50-95(B)", "mAP50-95"),
    ("metrics/precision(B)", "precision"),
    ("metrics/recall(B)", "recall"),
)
LOSSES = (
    ("train/box_loss", "train box"),
    ("train/cls_loss", "train cls"),
    ("val/box_loss", "val box"),
    ("val/cls_loss", "val cls"),
)
BAR_WIDTH = 28


def find_results(explicit=None):
    """The newest results.csv under runs/, unless one was named."""
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        return path
    candidates = sorted(Path("runs").rglob("results.csv"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit("No results.csv under runs/. Is training running?")
    return candidates[0]


def read_rows(path):
    """Parse results.csv, tolerating a row half-written as we read it."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []
    parsed = []
    for row in rows:
        try:
            parsed.append({key.strip(): float(value)
                           for key, value in row.items()
                           if value not in (None, "")})
        except ValueError:
            continue          # a partial final line; it will be complete next time
    return [row for row in parsed if "epoch" in row]


def render_terminal(rows, total_epochs):
    latest = rows[-1]
    epoch = int(latest["epoch"])
    print(f"\n  epoch {epoch}" + (f"/{total_epochs}" if total_epochs else "")
          + f"   elapsed {latest.get('time', 0) / 60:.1f} min")
    for key, label in METRICS:
        if key not in latest:
            continue
        value = latest[key]
        filled = int(round(value * BAR_WIDTH))
        best = max(row[key] for row in rows if key in row)
        marker = "  <- best" if value >= best - 1e-9 else f"  (best {best:.3f})"
        print(f"    {label:<10} {value:.3f} "
              f"[{'#' * filled}{'.' * (BAR_WIDTH - filled)}]{marker}")
    losses = "   ".join(f"{label} {latest[key]:.3f}"
                        for key, label in LOSSES if key in latest)
    if losses:
        print(f"    {losses}")


def render_plot(rows, axes, total_epochs):
    epochs = [row["epoch"] for row in rows]
    metrics_axis, loss_axis = axes

    metrics_axis.clear()
    for key, label in METRICS:
        series = [(row["epoch"], row[key]) for row in rows if key in row]
        if series:
            metrics_axis.plot([p[0] for p in series], [p[1] for p in series],
                              marker="o", markersize=3, label=label)
    metrics_axis.set_ylabel("metric")
    metrics_axis.set_ylim(0, 1.02)
    metrics_axis.grid(alpha=0.3)
    metrics_axis.legend(loc="lower right", fontsize=8)
    best = max((row.get("metrics/mAP50(B)", 0) for row in rows), default=0)
    metrics_axis.set_title(f"epoch {int(epochs[-1])}"
                           + (f"/{total_epochs}" if total_epochs else "")
                           + f"   best mAP50 {best:.3f}")

    loss_axis.clear()
    for key, label in LOSSES:
        series = [(row["epoch"], row[key]) for row in rows if key in row]
        if series:
            loss_axis.plot([p[0] for p in series], [p[1] for p in series],
                           marker="o", markersize=3, label=label)
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("loss")
    loss_axis.grid(alpha=0.3)
    loss_axis.legend(loc="upper right", fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None,
                        help="path to results.csv (default: newest under runs/)")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--terminal", action="store_true",
                        help="print instead of opening a window")
    parser.add_argument("--once", action="store_true", help="render once and exit")
    parser.add_argument("--save", default=None, help="also write the plot here")
    parser.add_argument("--epochs", type=int, default=None,
                        help="total epochs, for the progress label")
    args = parser.parse_args()

    path = find_results(args.results)
    print(f"  watching {path}")

    if args.terminal:
        seen = 0
        while True:
            rows = read_rows(path)
            if len(rows) > seen:
                render_terminal(rows, args.epochs)
                seen = len(rows)
            if args.once:
                return
            time.sleep(args.interval)

    import matplotlib.pyplot as plt

    plt.ion()
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    figure.canvas.manager.set_window_title(f"training - {path.parent.name}")

    seen = 0
    try:
        while True:
            rows = read_rows(path)
            if rows and len(rows) != seen:
                render_plot(rows, axes, args.epochs)
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
