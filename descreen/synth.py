"""Synthetic, ground-truth fixtures for the descreen tool.

All fixtures are generated deterministically from an explicit screen geometry so
that detection can be scored against known periods and angles.  They are input
images, never pre-fixed "after" images.
"""

from __future__ import annotations

from dataclasses import dataclass
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ScreenSpec:
    period_px: float
    angle_deg: float
    amplitude: float = 0.11


def dot_screen(
    shape: tuple[int, int],
    spec: ScreenSpec,
    ink: np.ndarray,
    *,
    dot_gamma: float = 1.0,
) -> np.ndarray:
    """Apply a rotated 2-D halftone screen whose local dot size follows tone.

    Real halftones are a 2-D dot lattice, not a 1-D cosine; the lattice puts
    energy on the fundamental, its harmonics, and lattice combinations, with
    nothing in between.  We build a square lattice rotated by ``angle_deg``,
    threshold it at a level derived from the local tone (preserving the local
    mean), and mix in the underlying detail at a reduced contrast to model ink
    opacity.  ``amplitude`` is the dot contrast.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    th = np.deg2rad(spec.angle_deg)
    p = max(float(spec.period_px), 1.5)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    # Square lattice of round-ish dots: separable raised cosines, 0..1.
    lattice = 0.5 + 0.25 * (np.cos(2 * np.pi * u / p) + np.cos(2 * np.pi * v / p))
    ink_img = np.clip(ink, 0.0, 1.0)
    threshold = np.clip(1.0 - ink_img, 0.02, 0.98)
    transition = 0.06
    coverage = np.clip((lattice - threshold) / transition + 0.5, 0.0, 1.0)
    coverage = coverage ** max(dot_gamma, 1e-6)
    depth = float(np.clip(spec.amplitude, 0.01, 0.9))
    # (1-depth) passes the continuous-tone signal; depth is carried by dots.
    screened = (1.0 - depth) * ink_img + depth * (1.0 - coverage)
    return np.clip(screened, 0.0, 1.0)


def _font(size: int) -> ImageFont.FreeTypeFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size)


def make_text_fixture(
    shape: tuple[int, int] = (960, 1280),
    *,
    screen: ScreenSpec | None = ScreenSpec(7.2, 15.0),
    seed: int = 7,
) -> tuple[np.ndarray, dict]:
    """Sharp text, fine lines, and hairline diagonals on a light background."""
    h, w = shape
    rng = np.random.default_rng(seed)
    img = Image.new("L", (w, h), 242)
    d = ImageDraw.Draw(img)
    title = _font(54)
    body = _font(25)
    small = _font(15)
    d.text((56, 42), "Halftone screen removal test", font=title, fill=18)
    lines = [
        "Sharp text must keep its edge slope after descreening.",
        "Fine serifs at 6 pt: The quick brown fox jumps over 13 lazy dogs.",
        "Hairlines below should remain resolved, not smeared:",
    ]
    y = 118
    for line in lines:
        d.text((56, y), line, font=body, fill=24)
        y += 36
    for i in range(12):
        x = 60 + i * 90
        d.line([(x, y + 8), (x + 66, y + 8)], fill=12, width=1)
        d.line([(x, y + 22), (x + 66, y + 22)], fill=12, width=2)
        d.line([(x, y + 38), (x + 66, y + 38)], fill=12, width=3)
    y += 70
    d.text((56, y), "Small print 9 pt: ABCDEFGHIJKLM 0123456789 .,:;", font=small, fill=30)
    y += 28
    # Hairline diagonal and a few tone patches.
    for k in range(6):
        off = k * 34
        d.line([(70 + off, y + 10), (190 + off, y + 150)], fill=10, width=1)
    for k in range(5):
        v = int(30 + k * 50)
        d.rectangle([(560 + k * 70, y + 8), (620 + k * 70, y + 140)], fill=v)
    # Slight paper grain to avoid an unrealistically clean spectrum.
    tone = np.asarray(img, dtype=np.float32) / 255.0
    tone = tone + rng.normal(0.0, 0.004, tone.shape).astype(np.float32)
    tone = np.clip(tone, 0.0, 1.0)
    meta = {
        "kind": "text",
        "screen": None if screen is None else screen.__dict__,
        "note": "sharp text, 1 px hairlines, diagonal hairlines, tone patches",
    }
    if screen is not None:
        tone = dot_screen(tone.shape, screen, tone)
    return tone, meta


def make_fabric_fixture(
    shape: tuple[int, int] = (900, 1200),
    *,
    screen: ScreenSpec | None = ScreenSpec(6.4, 45.0),
    seed: int = 11,
) -> tuple[np.ndarray, dict]:
    """Real-looking woven fabric: two orthogonal yarn frequencies + noise.

    The two yarn frequencies are deliberately different from the halftone
    screen frequency and angle.  This is the confounder test: the algorithm
    must not silently erase the textile repeat as if it were a screen.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warp_period = 9.0
    weft_period = 12.5
    warp_angle = 8.0
    weft_angle = 98.0
    tw = np.deg2rad(warp_angle)
    tf = np.deg2rad(weft_angle)
    warp = np.sin(2 * np.pi * (xx * np.cos(tw) + yy * np.sin(tw)) / warp_period)
    weft = np.sin(2 * np.pi * (xx * np.cos(tf) + yy * np.sin(tf)) / weft_period)
    # Yarn irregularity: slow amplitude modulation and random slubs.
    slow = 0.65 + 0.35 * np.sin(2 * np.pi * xx / 317.0 + 0.4)
    slub = rng.normal(0.0, 0.05, (h, w)).astype(np.float32)
    slub = np.clip(slub, -0.15, 0.15)
    weave = 0.78 + 0.075 * warp * slow + 0.065 * weft + slub
    # A printed photograph on fabric gives some large-scale shading.
    weave *= 0.92 + 0.08 * np.sin(2 * np.pi * (xx + 0.3 * yy) / 430.0)
    base = np.clip(weave, 0.0, 1.0).astype(np.float32)
    meta = {
        "kind": "fabric",
        "screen": None if screen is None else screen.__dict__,
        "real_periods_px": [warp_period, weft_period],
        "real_angles_deg": [warp_angle, weft_angle],
        "note": "woven fabric with real 9.0/12.5 px repeats; not a screen",
    }
    if screen is not None:
        base = dot_screen(base.shape, screen, base)
    return base, meta


def make_photo_fixture(
    shape: tuple[int, int] = (900, 1200),
    *,
    screen: ScreenSpec | None = ScreenSpec(8.0, 7.0),
    seed: int = 5,
) -> tuple[np.ndarray, dict]:
    """Continuous-tone photo-like content with no halftone screen.

    Built from smooth gradients, hard architectural edges, and fine natural
    texture.  Used to test false positives on screen-free originals.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / w
    yn = yy / h
    sky = 0.62 + 0.18 * yn
    ground = 0.38 + 0.10 * np.cos(2 * np.pi * xn * 1.5)
    base = np.where(yn < 0.52, sky, ground)
    # Hard horizon and building edges.
    base[yn > 0.52] -= 0.05
    for x0, x1, top, val in [
        (0.10, 0.26, 0.30, 0.25),
        (0.30, 0.46, 0.22, 0.18),
        (0.52, 0.72, 0.36, 0.30),
        (0.77, 0.92, 0.27, 0.22),
    ]:
        m = (xn > x0) & (xn < x1) & (yn > top) & (yn < 0.53)
        base[m] = val
    # Window grid (repeating texture) and fine natural noise.
    win = ((np.mod(xx - 20, 46) < 18) & (np.mod(yy - 30, 38) < 16)
           & (yn > 0.32) & (yn < 0.50) & (xn > 0.10) & (xn < 0.92))
    base[win] += 0.13
    texture = rng.normal(0, 0.018, (h, w)).astype(np.float32)
    base = np.clip(base.astype(np.float32) + texture, 0.0, 1.0)
    meta = {
        "kind": "photo",
        "screen": None if screen is not None else None,
        "note": "continuous tone; no periodic screen was applied",
    }
    # `screen` is accepted so the same experiment harness can request a
    # screened version, but the default fixture is screen-free.
    if screen is not None:
        base = dot_screen(base.shape, screen, base)
        meta["screen"] = screen.__dict__
    return base, meta


def make_palette_fixture(
    shape: tuple[int, int] = (760, 1000),
    *,
    angles: tuple[float, float, float] = (15.0, 75.0, 0.0),
    period_px: float = 7.5,
    screened: bool = True,
) -> tuple[np.ndarray, dict]:
    """Color halftone with three rotated separations and hard color edges.

    ``screened=False`` returns the same artwork without any screen, for use as a
    clean ground-truth reference when measuring content loss.
    """
    h, w = shape
    img = Image.new("RGB", (w, h), (238, 232, 220))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 360, 360], fill=(196, 32, 40))
    d.rectangle([390, 60, 690, 360], fill=(28, 74, 158))
    d.ellipse([740, 110, 930, 300], fill=(20, 130, 86))
    d.polygon([(100, 650), (330, 410), (560, 650)], fill=(240, 196, 40))
    d.rectangle([600, 430, 930, 680], fill=(90, 90, 96))
    d.text((90, 600), "Color alignment / registration", font=_font(34), fill=(20, 20, 20))
    base = np.asarray(img, dtype=np.float32) / 255.0
    out = base.copy()
    meta = {
        "kind": "palette",
        "screens": [{"channel": c, "period_px": period_px * (1 + 0.03 * c),
                     "angle_deg": a} for c, a in enumerate(angles)],
        "note": "three independently rotated screens; hard color edges",
    }
    if not screened:
        meta["screens"] = []
        meta["note"] = "clean color artwork, no screen (ground-truth reference)"
        return np.clip(base, 0.0, 1.0), meta
    # Separate screens per channel, as in real CMYK-ish halftones.
    for c, ang in enumerate(angles):
        ch = out[..., c]
        spec = ScreenSpec(period_px * (1.0 + 0.03 * c), ang, amplitude=0.10)
        out[..., c] = dot_screen(ch.shape, spec, ch)
    return np.clip(out, 0.0, 1.0), meta


def make_known_screen_fixture(
    shape: tuple[int, int] = (1024, 1280),
    *,
    period_px: float = 6.0,
    angle_deg: float = 33.0,
    seed: int = 3,
) -> tuple[np.ndarray, dict]:
    """Flat midtone carrying only a known screen plus slight grain.

    This isolates period/angle measurement from image content.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 0.58, dtype=np.float32)
    base += rng.normal(0, 0.003, base.shape).astype(np.float32)
    spec = ScreenSpec(period_px, angle_deg, amplitude=0.18)
    out = dot_screen(base.shape, spec, np.clip(base, 0, 1))
    meta = {"kind": "known_screen", "screen": spec.__dict__,
            "note": "flat tone + known screen; used for period/angle scoring"}
    return out, meta


def make_two_frequency_fixture(
    shape: tuple[int, int] = (900, 1200),
    *,
    split: int | None = None,
    left_screen: ScreenSpec = ScreenSpec(6.0, 33.0, 0.16),
    right_screen: ScreenSpec = ScreenSpec(9.5, 12.0, 0.17),
    seed: int = 21,
) -> tuple[np.ndarray, dict]:
    """One page with two different screens and text crossing the junction.

    This is the region-coordination test page: the left part carries one
    screen and the right part another, and the same words run continuously
    across the boundary.  A per-block paste produces a visible step there; a
    coordinated, smoothly weighted composite must not.

    The clean variant returns the identical artwork without any screen, which
    is the honest reference for text-detail retention.
    """
    h, w = shape
    split = int(w * 0.55) if split is None else int(split)
    meta = _two_frequency_art(shape, split, seed)
    clean = meta.pop("_image")
    left = dot_screen(clean.shape, left_screen, clean)
    right = dot_screen(clean.shape, right_screen, clean)
    page = left.copy()
    page[:, split:] = right[:, split:]
    meta["kind"] = "two_frequency"
    meta["split_x"] = int(split)
    meta["screens"] = [
        {"region": "left", "rect": [0, 0, split, h], **_spec_dict(left_screen)},
        {"region": "right", "rect": [split, 0, w - split, h],
         **_spec_dict(right_screen)},
    ]
    meta["note"] = "two different screens; text and rules cross the junction"
    return page.astype(np.float32), meta


def make_two_frequency_clean(
    shape: tuple[int, int] = (900, 1200),
    *,
    split: int | None = None,
    seed: int = 21,
) -> tuple[np.ndarray, dict]:
    """Same artwork as :func:`make_two_frequency_fixture`, without screens."""
    h, w = shape
    split = int(w * 0.55) if split is None else int(split)
    meta = _two_frequency_art(shape, split, seed)
    return meta.pop("_image"), meta


def _two_frequency_art(shape: tuple[int, int], split: int,
                       seed: int) -> dict:
    h, w = shape
    img = Image.new("L", (w, h), 240)
    d = ImageDraw.Draw(img)
    body = _font(30)
    small = _font(17)
    d.text((48, 40), "Two screen frequencies, one continuous line",
           font=_font(46), fill=20)
    for i in range(7):
        y = 130 + i * 52
        d.text((48, y), "The quick brown fox jumps over the lazy dog "
                        "while the press runs at full speed.", font=body, fill=22)
    baseline = 130 + 7 * 52 + 12
    d.line([(30, baseline), (w - 30, baseline)], fill=8, width=2)
    d.line([(30, baseline + 12), (w - 30, baseline + 12)], fill=8, width=1)
    d.text((48, baseline + 28),
           "Hairline grid and 9 pt text must stay resolved on both sides "
           "of the screen change.", font=small, fill=26)
    for i in range(14):
        x = 48 + i * 72
        d.line([(x, baseline + 62), (x + 52, baseline + 62)], fill=10, width=1)
    d.line([(split, 90), (split, baseline + 90)], fill=14, width=1)
    rng = np.random.default_rng(seed)
    tone = np.asarray(img, dtype=np.float32) / 255.0
    tone = np.clip(tone + rng.normal(0.0, 0.004, tone.shape).astype(np.float32),
                   0.0, 1.0)
    return {
        "kind": "two_frequency_clean",
        "screens": [],
        "note": "two-screen test artwork without a screen",
        "_image": tone,
    }


def _spec_dict(spec: ScreenSpec) -> dict:
    return {"period_px": spec.period_px, "angle_deg": spec.angle_deg,
            "amplitude": spec.amplitude}


def to_png_bytes(arr: np.ndarray) -> bytes:
    if arr.dtype != np.uint8:
        arr8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        arr8 = arr
    im = Image.fromarray(arr8)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def from_png_bytes(data: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(data)).convert("RGB" if Image.open(
        io.BytesIO(data)).mode in ("RGB", "RGBA") else "L")
    return np.asarray(im)
