"""Drawing the exhibition's look on a Tk canvas.

Tk has no rounded corners, no gradients and no icons, so the screen is drawn
rather than assembled: a canvas, a few shapes, and these helpers. That is not a
workaround - it is the only way to get this look without a browser, and a
browser is a second thing to be down while a room full of school students
waits.

Colours are named for what they are for rather than what they are, so the whole
palette can be changed in one place without hunting for hex codes in layout
code.
"""

from __future__ import annotations

import math

# -- palette --------------------------------------------------------------

DEEP = "#080d1c"            # the top of the background
NIGHT = "#141c37"           # the bottom of it
CARD = "#111a30"
CARD_HEAD = "#16203a"
EDGE = "#2a3556"
EDGE_SOFT = "#202a47"

TEXT = "#ffffff"
SUBTLE = "#9aa8c7"
FAINT = "#63719b"
ACCENT = "#4da3ff"
GOOD = "#4ade80"
WARN = "#fbbf24"
BAD = "#f87171"

STOP_TOP = "#ef5350"
STOP_BOTTOM = "#c0392b"

#: The six block colours, as they are drawn on the tiles and the overlay.
BLOCK = {"red": "#e04a4a", "orange": "#e8892b", "yellow": "#edc233",
         "green": "#57bb4a", "blue": "#3f7fe0", "purple": "#9450d6"}


# -- colour arithmetic ----------------------------------------------------

def rgb(colour):
    colour = colour.lstrip("#")
    return tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))


def hex_of(values):
    return "#" + "".join(f"{max(0, min(255, int(v))):02x}" for v in values)


def mix(first, second, amount):
    """`amount` of the way from `first` to `second`."""
    a, b = rgb(first), rgb(second)
    return hex_of(a[i] + (b[i] - a[i]) * amount for i in range(3))


def lighter(colour, amount=0.25):
    return mix(colour, "#ffffff", amount)


def darker(colour, amount=0.25):
    return mix(colour, "#000000", amount)


# -- shapes ---------------------------------------------------------------

def gradient(canvas, x0, y0, x1, y1, top, bottom, steps=96, tags=()):
    """A vertical wash. Tk draws no gradients, so this draws one in bands."""
    height = max(1, y1 - y0)
    band = math.ceil(height / steps)
    for index in range(steps):
        y = y0 + index * band
        canvas.create_rectangle(
            x0, y, x1, min(y + band + 1, y1), width=0, tags=tags,
            fill=mix(top, bottom, index / max(1, steps - 1)))


def starfield(canvas, x0, y0, x1, y1, count=90, seed=7, tags=()):
    """A scatter of faint dots, so the background is not a flat slab.

    Deterministic: the same dots every run, because a background that shimmers
    between launches looks like a fault rather than a decoration.
    """
    state = seed
    for _ in range(count):
        state = (state * 1103515245 + 12345) % (1 << 31)
        x = x0 + (state >> 7) % max(1, x1 - x0)
        state = (state * 1103515245 + 12345) % (1 << 31)
        y = y0 + (state >> 7) % max(1, y1 - y0)
        state = (state * 1103515245 + 12345) % (1 << 31)
        size = 1 + (state >> 9) % 2
        shade = mix(NIGHT, "#7fa6ff", 0.12 + ((state >> 5) % 30) / 100)
        canvas.create_oval(x, y, x + size, y + size, fill=shade, width=0,
                           tags=tags)


def rounded(canvas, x0, y0, x1, y1, radius=14, fill=CARD, outline=EDGE,
            width=1, tags=()):
    """A rounded rectangle, as a smoothed polygon.

    Tk's `smooth` rounds a polygon's corners for us if the corner is given as
    three points close together; that is cheaper and crisper than compositing
    arcs with rectangles, and it takes an outline in one call.
    """
    r = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
    points = [
        x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
        x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
        x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
    ]
    return canvas.create_polygon(points, smooth=True, splinesteps=16,
                                 fill=fill, outline=outline, width=width,
                                 tags=tags)


def pill(canvas, x, y, text, font, dot=ACCENT, fill="#16213c", outline=EDGE,
         colour=SUBTLE, pad=(14, 7), tags=()):
    """A small status badge with a coloured dot. Returns (shape, dot, label)."""
    width = font.measure(text) + pad[0] * 2 + 16
    height = font.metrics("linespace") + pad[1] * 2
    shape = rounded(canvas, x, y, x + width, y + height, height / 2,
                    fill=fill, outline=outline, tags=tags)
    middle = y + height / 2
    marker = canvas.create_oval(x + 12, middle - 4, x + 20, middle + 4,
                                fill=dot, width=0, tags=tags)
    label = canvas.create_text(x + 28, middle, text=text, font=font,
                               fill=colour, anchor="w", tags=tags)
    return shape, marker, label


def cube(canvas, cx, cy, size, colour, tags=()):
    """An isometric block, drawn as its three visible faces.

    The blocks on the table are cubes and the buttons should look like them,
    so a visitor can match the button to the thing rather than to a word.
    """
    half = size / 2
    lift = size * 0.30
    top = [cx, cy - half, cx + half, cy - half + lift,
           cx, cy - half + lift * 2, cx - half, cy - half + lift]
    left = [cx - half, cy - half + lift, cx, cy - half + lift * 2,
            cx, cy + half, cx - half, cy + half - lift]
    right = [cx + half, cy - half + lift, cx, cy - half + lift * 2,
             cx, cy + half, cx + half, cy + half - lift]
    shapes = [
        canvas.create_polygon(top, fill=lighter(colour, 0.28), width=0,
                              tags=tags),
        canvas.create_polygon(left, fill=darker(colour, 0.28), width=0,
                              tags=tags),
        canvas.create_polygon(right, fill=colour, width=0, tags=tags),
    ]
    return shapes


def camera_glyph(canvas, x, y, size=22, colour=ACCENT, tags=()):
    """The little camera chip on each panel's header."""
    rounded(canvas, x, y, x + size, y + size, 6, fill=mix(colour, DEEP, 0.72),
            outline="", width=0, tags=tags)
    body = size * 0.46
    canvas.create_rectangle(x + (size - body) / 2, y + size * 0.36,
                            x + (size + body) / 2, y + size * 0.68,
                            fill=colour, width=0, tags=tags)
    canvas.create_rectangle(x + size * 0.40, y + size * 0.27,
                            x + size * 0.60, y + size * 0.36,
                            fill=colour, width=0, tags=tags)


def stop_glyph(canvas, x, y, size=20, colour="#ffffff", tags=()):
    canvas.create_rectangle(x, y, x + size, y + size, fill=colour, width=0,
                            tags=tags)
