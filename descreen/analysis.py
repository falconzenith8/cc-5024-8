"""Spectral screen detection with explicit evidence and manual overrides.

Core idea
---------
A printed halftone is a *2-D dot lattice*.  Its spectrum therefore contains a
coherent set of peaks at integer combinations ``m*a + n*b`` of two basis
vectors, with the fundamental strongest and harmonics weaker.  A woven fabric
or a repeated architectural grid is typically a sum of near-sinusoidal repeats:
it produces one or a few isolated peaks without that coherent lattice comb.

We measure that coherence explicitly and report it.  Classification is
evidence-based, never a silent "blur everything high frequency" decision:

* ``halftone_screen``   - coherent lattice + stationary coverage + narrow peak.
* ``ambiguous_periodic``- periodic energy that could be a real texture.
* ``real_texture``      - isolated/ridge-like energy with no lattice comb.

Only ``halftone_screen`` peaks are auto-selected.  Everything else is left for
the operator to confirm with the visible spectrum and pixel view.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .synth import to_png_bytes

DEFAULT_MIN_PERIOD = 2.5
DEFAULT_MAX_PERIOD = 32.0
ANALYSIS_WINDOW = "hann"
ANALYSIS_MAX_SIDE = 1280
PROBE_TOL_BINS = 2

# Integer lattice points used for coherence.  Weight falls off with |m|+|n|.
_LATTICE_COMBOS = [
    (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
    (1, 1, 0.85), (1, -1, 0.85), (-1, 1, 0.85), (-1, -1, 0.85),
    (2, 0, 0.70), (0, 2, 0.70), (-2, 0, 0.70), (0, -2, 0.70),
    (2, 1, 0.55), (2, -1, 0.55), (-2, 1, 0.55), (-2, -1, 0.55),
    (1, 2, 0.55), (1, -2, 0.55), (-1, 2, 0.55), (-1, -2, 0.55),
]


def fold_angle(deg: float) -> float:
    """Normalize an unoriented axis angle to [0, 180)."""
    a = deg % 180.0
    return a


def angle_diff(a: float, b: float) -> float:
    """Signed shortest difference of two unoriented axis angles, in [-90, 90).

    Comparing folded angles directly is wrong: 105.00 vs 104.98 becomes
    179.98 rather than -0.02, so near-identical screen axes would never merge.
    """
    d = (a - b) % 180.0
    if d > 90.0:
        d -= 180.0
    return d


def fold_frequency(f: float) -> float:
    return (f + 0.5) % 1.0 - 0.5


def luma(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    coeff = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return (rgb[..., :3].astype(np.float32) @ coeff).clip(0.0, 1.0)


def channel_array(rgb: np.ndarray, channel: str) -> np.ndarray:
    if channel == "luma":
        return luma(rgb)
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    return rgb[..., {"r": 0, "g": 1, "b": 2}[channel]].astype(np.float32)


def _hann2d(shape: tuple[int, int]) -> np.ndarray:
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1])).astype(np.float32)


def _parabolic(vm: float, v0: float, vp: float) -> float:
    denom = vm - 2.0 * v0 + vp
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (vm - vp) / denom, -0.75, 0.75))


@dataclass
class LatticePoint:
    m: int
    n: int
    fx: float
    fy: float
    period_px: float
    angle_deg: float
    rel_magnitude: float
    present: bool
    residual: float
    offset_bins: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Peak:
    """One screen candidate, represented by a lattice basis vector."""

    fx: float
    fy: float
    period_px: float
    angle_deg: float
    magnitude: float
    snr: float
    prominence: float
    isolation: float
    coverage: float
    coherence: float
    comb_present: int
    comb_expected: int
    collinear2_rel: float = 0.0
    collinear2_offset: float = 0.0
    collinear3_rel: float = 0.0
    collinear3_offset: float = 0.0
    squareness: float = 0.0
    perpendicularity: float = 0.0
    channel: str = "luma"
    channels: list[str] = field(default_factory=list)
    basis_b: dict | None = None
    lattice: list[dict] = field(default_factory=list)
    isotropy: float = 1.0
    lattice_orthogonality: float = 1.0
    classification: str = "uncertain"
    confidence: float = 0.0
    selected: bool = False
    reasons: list[str] = field(default_factory=list)
    harm_of_fx: float | None = None
    harm_of_fy: float | None = None
    display_angle_deg: float | None = None
    basis_display_angle_deg: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _structure_isotropy(logmag: np.ndarray, x: int, y: int, radius: int = 5) -> float:
    """Local isotropy of the log-spectrum around a bin.

    A printed dot lattice gives a compact, roughly symmetric peak.  Straight
    text rules and building edges give a spectral *ridge*: gradients are
    dominated by one direction, so the structure tensor is highly anisotropic.
    Returns 1 for isotropic, 0 for a pure ridge.
    """
    h, w = logmag.shape
    ys = np.arange(y - radius, y + radius + 1) % h
    xs = np.arange(x - radius, x + radius + 1) % w
    patch = logmag[np.ix_(ys, xs)].astype(np.float64)
    gy, gx = np.gradient(patch)
    jxx = float((gx * gx).mean())
    jyy = float((gy * gy).mean())
    jxy = float((gx * gy).mean())
    tr = jxx + jyy
    if tr <= 1e-12:
        return 1.0
    disc = math.sqrt(max(0.0, (jxx - jyy) ** 2 + 4 * jxy * jxy))
    lam1 = 0.5 * (tr + disc)
    lam2 = 0.5 * (tr - disc)
    anis = (lam1 - lam2) / max(tr, 1e-12)
    return float(np.clip(1.0 - anis, 0.0, 1.0))


def _local_background(logmag: np.ndarray) -> np.ndarray:
    # Separable percentile/mean approximation. A full 9x9 median on a 1.3 MP
    # ROI costs over a second and adds no classification value beyond a robust
    # local floor, so use a 3x3 minimum followed by a box mean.
    floor = ndimage.minimum_filter(logmag, size=3, mode="nearest")
    return ndimage.uniform_filter(floor, size=9, mode="nearest")


def _annulus_values(logmag: np.ndarray, shape: tuple[int, int],
                    fx: float, fy: float, tol_bins: float = 2.0,
                    min_angle_deg: float = 14.0) -> np.ndarray:
    h, w = shape
    FY, FX = np.mgrid[0:h, 0:w]
    # Center-shifted distance in bins.
    cy, cx = h / 2.0, w / 2.0
    # FFT bin coordinates relative to the candidate on the unshifted grid.
    d_fx = np.abs((np.arange(w) - (fx % 1.0) * w + w / 2) % w - w / 2)
    d_fy = np.abs((np.arange(h) - (fy % 1.0) * h + h / 2) % h - h / 2)
    r = np.hypot(d_fx[None, :], d_fy[:, None])
    ang = np.arctan2(d_fy[:, None] * np.ones_like(d_fx)[None, :],
                     np.ones_like(d_fy)[:, None] * d_fx[None, :])
    peak_ang = math.atan2(fy, fx)
    d_ang = np.abs((ang - peak_ang + np.pi) % (2 * np.pi) - np.pi)
    ring = (r > tol_bins) & (r <= tol_bins * 2.5) & (d_ang > math.radians(min_angle_deg))
    return logmag[ring]


def _probe(logmag: np.ndarray, shape: tuple[int, int],
           fx: float, fy: float, tol: int = PROBE_TOL_BINS) -> tuple[float, bool]:
    """Max log magnitude in a small wrapped neighborhood, and whether the
    frequency is inside the representable band (not folded onto DC)."""
    h, w = shape
    fx = fold_frequency(fx)
    fy = fold_frequency(fy)
    xb = int(round(fx * w)) % w
    yb = int(round(fy * h)) % h
    xs = (np.arange(xb - tol, xb + tol + 1)) % w
    ys = (np.arange(yb - tol, yb + tol + 1)) % h
    patch = logmag[np.ix_(ys, xs)]
    return float(patch.max()), True


def _tile_log_spectra(arr: np.ndarray, tiles: int, tile_side: int) -> tuple[list[np.ndarray], list[tuple[int, int]], int]:
    h, w = arr.shape
    th = min(tile_side, h)
    tw = min(tile_side, w)
    spectra: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    if th < 64 or tw < 64:
        return spectra, origins, 0
    ys = np.linspace(0, h - th, tiles).astype(int)
    xs = np.linspace(0, w - tw, tiles).astype(int)
    for ty in ys:
        for tx in xs:
            tile = arr[ty:ty + th, tx:tx + tw]
            tile = tile - tile.mean()
            spec = np.log1p(np.abs(np.fft.fft2(tile * _hann2d(tile.shape))))
            spectra.append(spec)
            origins.append((th, tw))
    return spectra, origins, len(spectra)


def _coverage_from_tiles(spectra: list[np.ndarray], fx: float, fy: float) -> float:
    if not spectra:
        return 0.0
    ok = 0
    for spec in spectra:
        h, w = spec.shape
        xb = int(round(fold_frequency(fx) * w)) % w
        yb = int(round(fold_frequency(fy) * h)) % h
        xs = np.arange(xb - 2, xb + 3) % w
        ys = np.arange(yb - 2, yb + 3) % h
        patch = spec[np.ix_(ys, xs)]
        peak = float(patch.max())
        bg = float(np.median(spec))
        # A screen peak is a narrow local maximum well above that tile's own
        # floor. Using a fraction of the tile maximum fails on flat patches,
        # where background structure is negligible and the peak is the max.
        if peak - bg > 0.55:
            ok += 1
    return ok / len(spectra)


def _find_candidates(logmag: np.ndarray, bg: np.ndarray,
                     *, min_period: float, max_period: float,
                     limit: int = 60) -> list[dict]:
    h, w = logmag.shape
    residual = logmag - bg
    localmax = ndimage.maximum_filter(logmag, size=5, mode="wrap")
    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    FX, FY = np.meshgrid(fx, fy)
    radius = np.hypot(FX, FY)
    band = (radius >= 1.0 / max_period) & (radius <= 1.0 / min_period)
    mask = band & (logmag == localmax) & (residual > 0.45)
    ys, xs = np.nonzero(mask)
    keep = (ys < h // 2) | ((ys == 0) & (xs > 0)) | ((ys == h // 2) & (xs > 0))
    ys, xs = ys[keep], xs[keep]
    if ys.size == 0:
        return []
    order = np.argsort(logmag[ys, xs])[::-1]
    out: list[dict] = []
    used: list[tuple[float, float]] = []
    for idx in order[: limit * 6]:
        y, x = int(ys[idx]), int(xs[idx])
        px, py = float(fx[x]), float(fy[y])
        # Deduplicate the same basin, including wrapped edges.
        if any(math.hypot(px - ux, py - uy) < 1.6 / min(h, w) for ux, uy in used):
            continue
        dx = _parabolic(logmag[y, (x - 1) % w], logmag[y, x], logmag[y, (x + 1) % w])
        dy = _parabolic(logmag[(y - 1) % h, x], logmag[y, x], logmag[(y + 1) % h, x])
        sx = px + dx / w
        sy = py + dy / h
        if sx < 0:
            sx, sy = -sx, -sy
        r = math.hypot(sx, sy)
        if r < 1.0 / max_period or r > 1.0 / min_period:
            continue
        used.append((px, py))
        out.append({
            "fx": float(sx), "fy": float(sy),
            "period_px": float(1.0 / r),
            "angle_deg": float(fold_angle(math.degrees(math.atan2(sy, sx)))),
            "logmag": float(logmag[y, x]),
            "residual": float(residual[y, x]),
            "magnitude": float(np.expm1(logmag[y, x])),
        })
        if len(out) >= limit:
            break
    return out


def _probe_rel(logmag: np.ndarray, bg: np.ndarray, shape: tuple[int, int],
               fx: float, fy: float, tol: int = PROBE_TOL_BINS,
               refine: bool = False) -> tuple[float, float, bool, float]:
    """Return (max logmag, residual vs bg, present, offset_bins).

    With ``refine`` the distance from the *exact* lattice point to the chosen
    peak bin is returned.  Printed-dot harmonics land within a fraction of a
    bin of the integer combination; amplitude-modulation sidebands of a real
    fabric repeat do not.  This is the main confounder discriminator.
    """
    h, w = logmag.shape
    fx_f = fold_frequency(fx)
    fy_f = fold_frequency(fy)
    xb = fx_f * w
    yb = fy_f * h
    xi = int(round(xb)) % w
    yi = int(round(yb)) % h
    xs = np.arange(xi - tol, xi + tol + 1) % w
    ys = np.arange(yi - tol, yi + tol + 1) % h
    patch_log = logmag[np.ix_(ys, xs)]
    patch_bg = bg[np.ix_(ys, xs)]
    i = np.unravel_index(np.argmax(patch_log), patch_log.shape)
    peak_log = float(patch_log[i])
    peak_bg = float(patch_bg[i])
    offset = 0.0
    if refine:
        # Sub-bin distance from the exact integer combination to the winning
        # bin, with circular wrap so negative frequencies stay small.
        dx = (i[1] - tol) + (xi - xb)
        dy = (i[0] - tol) + (yi - yb)
        dx = (dx + w / 2.0) % w - w / 2.0
        dy = (dy + h / 2.0) % h - h / 2.0
        offset = float(math.hypot(dx, dy))
    return peak_log, float(peak_log - peak_bg), True, offset


def _coherence(logmag: np.ndarray, bg: np.ndarray, shape: tuple[int, int],
              a: dict, b: dict | None, noise: float | None = None,
              *, offset_tol: float = 0.95
              ) -> tuple[float, list[LatticePoint], dict | None, int, int]:
    """Probe integer combinations of two basis vectors.

    A lattice line counts as present only when it is (i) prominent relative to
    the fundamental, (ii) above the global noise floor, and (iii) within
    ``offset_tol`` bins of the exact integer combination.  The last condition
    rejects the amplitude-modulation sidebands of a woven/printed real repeat,
    which sit at a slightly different vector than the integer multiple.
    Weighted score favors low-order combinations.
    """
    h, w = shape
    fund_log, fund_res, _, _ = _probe_rel(logmag, bg, shape, a["fx"], a["fy"])
    fund_res = max(fund_res, 1e-6)
    if noise is None:
        sample = logmag[::8, ::8]
        noise = float(np.median(np.abs(sample - np.median(sample))) + 1e-6)
    noise = max(float(noise), 1e-6)
    points: list[LatticePoint] = []
    present_weight = 0.0
    total_weight = 0.0
    present_count = 0
    if b is not None:
        combos = _LATTICE_COMBOS
    else:
        combos = [(2, 0, 1.0), (3, 0, 0.8), (4, 0, 0.6)]
    for m, n, weight in combos:
        fx = m * a["fx"] + (n * b["fx"] if b is not None else 0.0)
        fy = m * a["fy"] + (n * b["fy"] if b is not None else 0.0)
        r = math.hypot(fold_frequency(fx), fold_frequency(fy))
        if r < 1.0 / DEFAULT_MAX_PERIOD or r > 0.5:
            continue
        logv, res, _, offset = _probe_rel(logmag, bg, shape, fx, fy, refine=True)
        rel = max(0.0, res) / fund_res
        present = (rel > 0.26) and (res > 1.5 * max(noise, 1.0)) and (offset <= offset_tol)
        period = 1.0 / max(r, 1e-9)
        angle = fold_angle(math.degrees(math.atan2(fy, fx)))
        points.append(LatticePoint(m, n, float(fx), float(fy), float(period),
                                   float(angle), float(rel), bool(present),
                                   float(res), float(offset)))
        total_weight += weight
        if present:
            present_weight += weight
            present_count += 1
    total_weight += 1.0
    present_weight += 1.0
    present_count += 1
    coherence = present_weight / total_weight if total_weight > 0 else 0.0
    return coherence, points, b, present_count, int(round(total_weight))


def _classify(p: Peak) -> None:
    reasons: list[str] = []
    score = 0.0
    c = p.coherence

    # 1. Collinear harmonic comb: the signature of a non-sinusoidal dot
    #    lattice.  Offsets must be sub-bin; fabric sidebands fail this.
    harm_hits = 0
    if p.collinear2_rel >= 0.20 and p.collinear2_offset <= 0.85:
        harm_hits += 1
    if p.collinear3_rel >= 0.17 and p.collinear3_offset <= 0.85:
        harm_hits += 1
    if harm_hits == 2:
        score += 0.30
        reasons.append("strong collinear harmonic comb (2f and 3f on the dot pitch)")
    elif harm_hits == 1:
        score += 0.16
        reasons.append("one collinear harmonic on the dot pitch")
    else:
        score -= 0.16
        reasons.append("no collinear dot harmonics; periodic but not a dot screen")

    # 2. Lattice coherence (2-D set of integer combinations).
    if c >= 0.72:
        score += 0.20
        reasons.append(f"coherent lattice ({p.comb_present}/{p.comb_expected} lines)")
    elif c >= 0.52:
        score += 0.10
        reasons.append(f"partial lattice coherence ({c:.2f})")
    elif c < 0.42:
        score -= 0.10
        reasons.append(f"incoherent lattice ({c:.2f})")

    # 3. Square lattice geometry (two near-equal, near-orthogonal axes).
    if p.squareness >= 0.72 and p.perpendicularity >= 0.70:
        score += 0.12
        reasons.append(f"square two-axis lattice (ratio {p.squareness:.2f}, "
                       f"perpendicularity {p.perpendicularity:.2f})")
    elif p.basis_b is None:
        score -= 0.10
        reasons.append("single-axis only; unusual for a rotated dot screen")

    # 4. Stationarity.
    if p.coverage >= 0.85:
        score += 0.10
        reasons.append(f"stationary across {p.coverage*100:.0f}% of tiles")
    elif p.coverage >= 0.6:
        score += 0.02
    else:
        score -= 0.14
        reasons.append(f"spatially intermittent ({p.coverage*100:.0f}% coverage)")

    # 5. Compact, isotropic peak rather than a ridge from straight edges.
    if p.isolation >= 3.0:
        score += 0.08
        reasons.append(f"isolated peak (annulus ratio {p.isolation:.1f})")
    elif p.isolation < 1.7:
        score -= 0.08
        reasons.append(f"ridge-like energy (annulus ratio {p.isolation:.1f})")
    if p.isotropy >= 0.45:
        score += 0.04
    else:
        score -= 0.12
        reasons.append(f"anisotropic peak (isotropy {p.isotropy:.2f}); "
                       f"typical of straight edges")

    # 6. Ruling range typical of scanned print.
    if 3.0 <= p.period_px <= 18.0:
        score += 0.06
    elif p.period_px > 24.0:
        score -= 0.06
        reasons.append(f"period {p.period_px:.1f} px is unusually coarse for a screen")

    score = float(np.clip(score, 0.0, 0.98))
    is_screen = (harm_hits >= 1 and c >= 0.50 and p.coverage >= 0.75
                 and p.isotropy >= 0.40 and p.basis_b is not None
                 and p.squareness >= 0.60 and p.perpendicularity >= 0.55
                 and score >= 0.55)
    if is_screen:
        cls = "halftone_screen"
    elif (p.coverage >= 0.60 and p.basis_b is not None and
          (harm_hits >= 1 or c >= 0.55) and score >= 0.28):
        cls = "ambiguous_periodic"
    else:
        cls = "real_texture"
    p.classification = cls
    p.confidence = score
    p.reasons = reasons


def _lattice_distance(c: dict, ax: float, ay: float, bx: float, by: float,
                      shape: tuple[int, int]) -> float:
    h, w = shape
    best = 1e9
    for m in (-2, -1, 0, 1, 2):
        for n in (-2, -1, 0, 1, 2):
            if m == 0 and n == 0:
                continue
            dx = (fold_frequency(c["fx"] - (m * ax + n * bx)) * w + w / 2) % w - w / 2
            dy = (fold_frequency(c["fy"] - (m * ay + n * by)) * h + h / 2) % h - h / 2
            d = math.hypot(dx, dy)
            if d < best:
                best = d
    return best


def _reduce_basis(a: tuple[float, float], b: tuple[float, float]
                  ) -> tuple[tuple[float, float], tuple[float, float]]:
    """Gauss (Lagrange) reduction in the reciprocal plane.

    Given any two independent lattice vectors, returns the shortest pair that
    spans the same lattice.  This is what turns a measured family of harmonics
    and combination lines back into the primitive dot pitch.
    """
    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1]

    def n2(u):
        return dot(u, u)

    while True:
        if n2(b) < n2(a):
            a, b = b, a
        mu = round(dot(a, b) / max(n2(a), 1e-18))
        if mu == 0:
            break
        b = (b[0] - mu * a[0], b[1] - mu * a[1])
    return a, b


def _match_lattice(cands: list[dict], ax: float, ay: float,
                   bx: float, by: float, shape: tuple[int, int],
                   tol: float = 2.5) -> list[dict]:
    return [c for c in cands if _lattice_distance(c, ax, ay, bx, by, shape) <= tol]


_LATTICE_COMBINATION_ORDERS = [
    (m, n) for m in range(-2, 3) for n in range(-2, 3)
    if abs(m) + abs(n) >= 2
]


def _lattice_explanation(p: "Peak", q: "Peak",
                         shape: tuple[int, int]) -> tuple[int, int, float] | None:
    """If p is a nonzero small-integer combination of q's primitive cell, return it.

    A square dot screen puts energy not only on its two axes but on the diagonal
    (1,1) and (2,1) lattice lines.  At 7.95 px/0deg those appear at 5.62 px/45deg
    and 3.56 px/26.5deg, which an independent-peak search would otherwise report
    as separate screens.  They belong to q's lattice, so the UI must not present
    them as additional screens.  The test is sub-bin tight: real, unrelated
    screens are not at exact integer combinations of another screen's cell.
    """
    if q.basis_b is None:
        return None
    if p.period_px >= 0.95 * q.period_px:
        # Combination lines are always shorter-period than the fundamental.
        return None
    h, w = shape
    vx, vy = q.basis_b["fx"], q.basis_b["fy"]
    best: tuple[int, int, float] | None = None
    for m, n in _LATTICE_COMBINATION_ORDERS:
        px = m * q.fx + n * vx
        py = m * q.fy + n * vy
        dx = (fold_frequency(p.fx - px) * w + w / 2) % w - w / 2
        dy = (fold_frequency(p.fy - py) * h + h / 2) % h - h / 2
        d = math.hypot(dx, dy)
        if best is None or d < best[2]:
            best = (m, n, d)
    if best is not None and best[2] <= 1.0:
        return best
    return None


def canonicalize_axes(peaks: list["Peak"]) -> None:
    """Attach a display angle in [0, 90) to every square-dot screen.

    A square lattice is invariant under a 90 deg rotation, so {theta, theta+90}
    is the same physical screen. The measured primary axis may land on either
    member (a 0 deg screen can be measured at 90.004 deg), which would compare
    badly to a known 0 deg screen. ``display_angle_deg`` is the primary axis
    folded to [0, 90); the physical ``angle_deg``/``fx``/``fy`` are left alone
    because they define the actual notch direction.
    """
    for p in peaks:
        p.display_angle_deg = float(p.angle_deg) % 90.0
        if p.basis_b:
            p.basis_display_angle_deg = float(p.basis_b["angle_deg"]) % 90.0


def screen_angle_diff(a: float, b: float) -> float:
    """Shortest angular distance between two square-screen angles mod 90 deg."""
    d = (a - b) % 90.0
    return min(d, 90.0 - d)


def reconcile_lattice_peaks(peaks: list["Peak"], shape: tuple[int, int]
                            ) -> tuple[list["Peak"], list[dict]]:
    """Collapse lattice-combination lines onto the screen group that generates them.

    A line is absorbed when it sits within one bin of an exact integer
    combination of another peak's two primitive axes, so distinct screens at
    other angles survive. The shortest-period-first rule is not used; the
    longest-period candidate is evaluated first *whether or not it was selected*,
    because a diagonal line can score higher than the fundamental that produced
    it. If the parent is rejected by the operator, its children come back as
    independent candidates rather than being silently dropped.
    """
    # Longest period first: a lattice combination line is always *shorter* than
    # the fundamental that generates it.
    ordered = sorted(peaks, key=lambda x: (x.period_px, x.selected, x.confidence),
                     reverse=True)
    kept: list[Peak] = []
    explained: list[dict] = []
    for p in ordered:
        owner = None
        combo = None
        for q in kept:
            hit = _lattice_explanation(p, q, shape)
            if hit is not None:
                owner, combo = q, hit
                break
        if owner is None:
            kept.append(p)
            continue
        p.harm_of_fx = owner.fx
        p.harm_of_fy = owner.fy
        d = p.to_dict()
        d["explained_by_period_px"] = owner.period_px
        d["explained_by_angle_deg"] = owner.angle_deg
        d["explained_by_selected"] = owner.selected
        d["lattice_order"] = [combo[0], combo[1]]
        d["distance_bins"] = combo[2]
        # Lattice evidence belongs to the physical screen, so the parent
        # inherits the support of its own lines even when they scored higher.
        if owner.channels and p.channels:
            for ch in p.channels:
                if ch not in owner.channels and ch != "assist":
                    owner.channels.append(ch)
        if (p.selected and not owner.selected
                and owner.classification != "real_texture"):
            # A diagonal line at an exact (m,n) position of this cell is direct
            # evidence of the same physical screen. If the fundamental alone was
            # below the classifier's threshold (weak in this channel, or reduced
            # by the ROI), the lattice line supplies the missing confirmation.
            # A peak already judged real_texture is not promoted: that decision
            # rests on missing collinear harmonics, which a diagonal does not
            # supply.
            owner.selected = True
            if owner.classification != "halftone_screen":
                owner.classification = "halftone_screen"
            owner.confidence = max(owner.confidence, p.confidence)
            owner.coverage = max(owner.coverage, p.coverage)
            owner.reasons = list(dict.fromkeys(owner.reasons + [
                f"confirmed by its ({combo[0]},{combo[1]}) lattice line at "
                f"{p.period_px:.2f} px / {p.angle_deg:.1f}°"]))
        explained.append(d)
    kept.sort(key=lambda x: (x.selected, x.confidence, x.magnitude), reverse=True)
    return kept, explained


def detect_screens(channel: np.ndarray, *, min_period: float = DEFAULT_MIN_PERIOD,
                   max_period: float = DEFAULT_MAX_PERIOD,
                   channel_name: str = "luma",
                   tile_cache: list[np.ndarray] | None = None) -> tuple[list[Peak], dict]:
    arr = np.asarray(channel, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("detect_screens expects a 2-D channel")
    arr = np.clip(arr, 0.0, 1.0)
    h, w = arr.shape
    if h < 64 or w < 64:
        raise ValueError("ROI must be at least 64x64 pixels")
    win = _hann2d((h, w))
    spec = np.fft.fft2((arr - float(arr.mean())) * win)
    logmag = np.log1p(np.abs(spec))
    bg = _local_background(logmag)
    sample = logmag[::8, ::8]
    noise = float(np.median(np.abs(sample - np.median(sample))) + 1e-6)

    cands = _find_candidates(logmag, bg, min_period=min_period, max_period=max_period)
    if not cands:
        return [], {"shape": [h, w], "reason": "no narrowband candidates in band"}

    # Fundamental-first search: a halftone's primitive pitch is the *longest*
    # period in its lattice family; harmonics and combination lines are always
    # shorter. Sorting by period descending therefore seeds the search with a
    # primitive axis and prevents combination lines from becoming "screens".
    for c in cands:
        _, res, _, _ = _probe_rel(logmag, bg, (h, w), c["fx"], c["fy"])
        c["probe_res"] = float(max(res, 1e-6))
    # Keep the strongest candidates first, so a genuine screen is not crowded
    # out by dozens of weak low-frequency noise blobs. Family ordering is
    # handled inside each seed iteration (fundamental = longest period within
    # the matched family), not by a global period sort.
    cands.sort(key=lambda c: c["probe_res"], reverse=True)
    max_res = max(c["probe_res"] for c in cands)
    strong_cut = max(0.18 * max_res, 1.2 * noise)
    strong = [c for c in cands if c["probe_res"] >= strong_cut][:40]
    if len(strong) < 4:
        strong = cands[:24]
    cands = strong

    remaining = list(cands)
    screens: list[Peak] = []

    while remaining:
        a = remaining[0]
        ax, ay = a["fx"], a["fy"]
        pa_period = a["period_px"]

        # Find the second primitive axis. Restrict the partner to a comparable
        # pitch: a rotated square screen has |a| ~ |b|; allowing wildly shorter
        # partners invites combination lines.
        best = None
        for b in remaining[1:]:
            ratio = b["period_px"] / max(pa_period, 1e-9)
            # A square dot lattice has equal pitch on both axes. Keep the
            # partner search tight around 1.0 so a harmonic cannot serve as the
            # second basis vector.
            if not (0.75 <= ratio <= 1.33):
                continue
            ang = abs(angle_diff(b["angle_deg"], a["angle_deg"]))
            if ang < 65.0 or ang > 115.0:
                continue
            coh, pts, _, pc, pe = _coherence(logmag, bg, (h, w), a, b, noise)
            if coh < 0.40:
                continue
            members = _match_lattice(cands, ax, ay, b["fx"], b["fy"], (h, w), tol=2.0)
            if len(members) < 5:
                continue
            sq = min(1.0, pa_period / max(b["period_px"], 1e-9)) if pa_period <= b["period_px"] else                 b["period_px"] / max(pa_period, 1e-9)
            # Reward: lattice coherence, how much of the strong spectrum is
            # explained, squareness, and a right-angle pair.
            energy = sum(m["probe_res"] for m in members) / max(max_res, 1e-9)
            score = (1.6 * coh + 0.8 * sq + max(0.0, 1.0 - abs(ang - 90.0) / 45.0)
                     + 0.5 * min(3.0, energy))
            if best is None or score > best[0]:
                best = (score, b, coh, pts, pc, pe, members)
        if best is None:
            remaining.pop(0)
            continue

        _, b, _, _, _, _, members = best
        # Gauss-reduce to the primitive cell, then keep axes ordered.
        (ux, uy), (vx, vy) = _reduce_basis((ax, ay), (b["fx"], b["fy"]))
        if math.hypot(ux, uy) > math.hypot(vx, vy):
            (ux, uy), (vx, vy) = (vx, vy), (ux, uy)
        ru, rv = math.hypot(ux, uy), math.hypot(vx, vy)
        if rv < 1e-9:
            remaining.pop(0)
            continue
        # Reject a "primitive" cell that is implausibly coarse (noise pair) or
        # that has collapsed to one axis.
        if 1.0 / ru > max_period * 1.25 or rv < 0.35 * ru:
            remaining.pop(0)
            continue

        def snap(fx: float, fy: float) -> tuple[float, float, dict]:
            peak = min(cands, key=lambda c: (fold_frequency(c["fx"] - fx) * w) ** 2
                       + (fold_frequency(c["fy"] - fy) * h) ** 2)
            d = math.hypot(fold_frequency(peak["fx"] - fx) * w,
                           fold_frequency(peak["fy"] - fy) * h)
            return (peak["fx"], peak["fy"], peak) if d <= 1.2 else (fx, fy, peak)

        ux, uy, pa = snap(ux, uy)
        vx, vy, pb = snap(vx, vy)
        ru, rv = math.hypot(ux, uy), math.hypot(vx, vy)
        if rv < 0.35 * ru:
            remaining.pop(0)
            continue
        period_a = 1.0 / max(ru, 1e-9)
        period_b = 1.0 / max(rv, 1e-9)
        ang_a = fold_angle(math.degrees(math.atan2(uy, ux)))
        ang_b = fold_angle(math.degrees(math.atan2(vy, vx)))

        # Collinear harmonic evidence on the primitive axis (the dot-lattice
        # signature). Use both 2f and 3f; the offset test rejects AM sidebands.
        _, fres, _, _ = _probe_rel(logmag, bg, (h, w), ux, uy)
        fres = max(fres, 1e-6)
        c2_coll = c3_coll = 0.0
        o2 = o3 = 9.9
        for k in (2, 3):
            _, rk, _, ok = _probe_rel(logmag, bg, (h, w), k * ux, k * uy, refine=True)
            rel = rk / fres if ok <= 0.85 else 0.0
            if k == 2:
                c2_coll, o2 = rel, ok
            else:
                c3_coll, o3 = rel, ok

        coh, pts, _, pc, pe = _coherence(logmag, bg, (h, w),
                                         {"fx": ux, "fy": uy},
                                         {"fx": vx, "fy": vy}, noise)

        ring_vals = _annulus_values(logmag, (h, w), ux, uy, tol_bins=2.0)
        if ring_vals.size:
            bg_local = float(np.percentile(ring_vals, 50))
            hi_local = float(np.percentile(ring_vals, 95))
            snr = float(np.exp(fres - bg_local))
            isolation = float(np.exp(fres - hi_local))
        else:
            snr, isolation = 1.0, 1.0
        cov = max(_coverage_from_tiles(tile_cache or [], ux, uy),
                  _coverage_from_tiles(tile_cache or [], vx, vy))
        xb = int(round(fold_frequency(ux) * w)) % w
        yb = int(round(fold_frequency(uy) * h)) % h
        iso = _structure_isotropy(logmag, xb, yb, radius=6)
        squareness = min(ru, rv) / max(ru, rv, 1e-9)
        d = abs(angle_diff(ang_b, ang_a))
        perpendicularity = max(0.0, 1.0 - abs(d - 90.0) / 45.0)

        p = Peak(
            fx=float(ux), fy=float(uy),
            period_px=float(period_a), angle_deg=float(ang_a),
            magnitude=float(pa["magnitude"]), snr=snr,
            prominence=float(pa["residual"]), isolation=isolation,
            coverage=cov, coherence=float(coh),
            comb_present=pc, comb_expected=pe, channel=channel_name,
            channels=[channel_name],
            collinear2_rel=float(c2_coll), collinear2_offset=float(o2),
            collinear3_rel=float(c3_coll), collinear3_offset=float(o3),
            squareness=float(squareness), perpendicularity=float(perpendicularity),
            basis_b={"fx": float(vx), "fy": float(vy), "period_px": float(period_b),
                     "angle_deg": float(ang_b), "magnitude": float(pb["magnitude"])},
            lattice=[q.to_dict() for q in pts],
            isotropy=float(iso),
            lattice_orthogonality=float(rv / max(ru, 1e-9)),
        )
        _classify(p)
        p.selected = p.classification == "halftone_screen"
        screens.append(p)

        # Remove all candidates explained by this primitive lattice.
        new_remaining = [c for c in remaining
                         if _lattice_distance(c, ux, uy, vx, vy, (h, w)) > 2.0]
        if len(new_remaining) >= len(remaining):
            new_remaining = remaining[1:]
        remaining = new_remaining
        if not remaining:
            break

    screens.sort(key=lambda x: (x.selected, x.confidence, x.magnitude), reverse=True)
    diag = {
        "shape": [h, w],
        "window": ANALYSIS_WINDOW,
        "band_period_px": [min_period, max_period],
        "noise_floor_median": float(np.median(logmag)),
        "logmag": logmag,
        "bg": bg,
    }
    return screens, diag


def merge_channel_peaks(per_channel: dict[str, list[Peak]],
                        *, rel_tolerance: float = 0.035) -> list[Peak]:
    merged: list[Peak] = []
    for ch, peaks in per_channel.items():
        for p in peaks:
            found = None
            for q in merged:
                if (abs(p.period_px - q.period_px) / max(q.period_px, 1e-6) <= rel_tolerance
                        and abs(angle_diff(p.angle_deg, q.angle_deg)) <= 4.0):
                    found = q
                    break
            if found is None:
                q = Peak(**p.to_dict())
                q.channels = [ch]
                q.channel = ch
                merged.append(q)
            else:
                if ch not in found.channels:
                    found.channels.append(ch)
                # Keep the observation with the stronger lattice evidence.
                if (p.coherence, p.magnitude) > (found.coherence, found.magnitude):
                    channels = found.channels
                    keep = Peak(**p.to_dict())
                    keep.channels = channels
                    keep.channel = p.channel
                    found.__dict__.update(keep.__dict__)
                found.selected = found.selected or p.selected
                found.confidence = max(found.confidence, p.confidence)
                found.classification = (
                    "halftone_screen" if found.selected else
                    ("ambiguous_periodic" if found.confidence >= 0.36 else "real_texture"))
    # If a screen axis was seen in any channel, treat it as one physical screen.
    # Collapse duplicate observations of one physical screen geometry. This is
    # intentional: the same pitch/angle measured in two channels or two patches
    # is one screen, and the UI must not present it as several.
    deduped: list[Peak] = []
    for p in sorted(merged, key=lambda x: (x.selected, x.confidence, x.magnitude),
                    reverse=True):
        match = None
        for q in deduped:
            if (abs(p.period_px - q.period_px) / max(q.period_px, 1e-9) <= 0.045
                    and abs(angle_diff(p.angle_deg, q.angle_deg)) <= 5.0):
                match = q
                break
        # A square dot screen reports both axes (90 deg apart). Merge those as
        # one physical screen group, keeping the second axis for the notch.
        if match is None:
            for q in deduped:
                if (abs(p.period_px - q.period_px) / max(q.period_px, 1e-9) <= 0.06
                        and 70.0 <= abs(angle_diff(p.angle_deg, q.angle_deg)) <= 110.0):
                    match = q
                    break
        if match is None:
            deduped.append(p)
        else:
            for ch in p.channels:
                if ch not in match.channels:
                    match.channels.append(ch)
            if match.basis_b is None and abs(angle_diff(p.angle_deg, match.angle_deg)) > 60.0:
                match.basis_b = {"fx": p.fx, "fy": p.fy, "period_px": p.period_px,
                                 "angle_deg": p.angle_deg, "magnitude": p.magnitude}
            match.lattice = match.lattice + [q for q in p.lattice if q not in match.lattice]
            match.selected = match.selected or p.selected
            match.confidence = max(match.confidence, p.confidence)
            match.coverage = max(match.coverage, p.coverage)
            match.reasons = list(dict.fromkeys(match.reasons + p.reasons))
    deduped.sort(key=lambda x: (x.selected, x.confidence, x.magnitude), reverse=True)
    return deduped


def _flat_tile_candidates(rgb: np.ndarray, *, tile: int = 256,
                          max_tiles: int = 8) -> list[tuple[int, int, int, int]]:
    """Pick low-gradient tiles for a content-light spectral measurement.

    Large flat color patches expose a screen clearly. On a full page, strong
    edges raise the local floor and hide the narrow peak, so a flat-tile pass
    is a legitimate diagnostic aid: any screen it finds is still verified by the
    same classifier gates and reported as measured on a flat patch.
    """
    y = luma(rgb)
    h, w = y.shape
    tile = min(tile, h, w)
    if tile < 128:
        return []
    gy, gx = np.gradient(y)
    grad = np.hypot(gx, gy)
    step = max(64, tile // 2)
    scored = []
    for ty in range(0, h - tile + 1, step):
        for tx in range(0, w - tile + 1, step):
            block = grad[ty:ty + tile, tx:tx + tile]
            # Score by mean gradient. A percentile gate is deliberately not
            # used: a single patch border inside an otherwise flat block would
            # eliminate every usable tile on a page of commercial art.
            scored.append((float(np.mean(block)), tx, ty))
    if not scored:
        return []
    scored.sort()
    cut = max(scored[0][0] * 2.0 + 0.001, 0.004)
    return [(tx, ty, tile, tile) for g, tx, ty in scored[:max_tiles] if g <= cut]


def analyze_image(rgb: np.ndarray, roi: tuple[int, int, int, int] | None,
                  *, channel: str = "luma", deep_channels: bool = False,
                  flat_tile_assist: bool = True) -> dict:
    h, w = rgb.shape[:2]
    if roi is None:
        x0, y0, rw, rh = 0, 0, w, h
    else:
        x0, y0, rw, rh = [int(v) for v in roi]
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        rw = max(1, min(rw, w - x0))
        rh = max(1, min(rh, h - y0))
    crop = rgb[y0:y0 + rh, x0:x0 + rw]
    analysis_scale = 1.0
    if max(crop.shape[:2]) > ANALYSIS_MAX_SIDE:
        factor = ANALYSIS_MAX_SIDE / max(crop.shape[:2])
        new_w = max(64, int(round(crop.shape[1] * factor)))
        new_h = max(64, int(round(crop.shape[0] * factor)))
        if crop.ndim == 2:
            im = Image.fromarray((crop * 255).astype(np.uint8), mode="L")
            crop = np.asarray(im.resize((new_w, new_h), Image.Resampling.LANCZOS),
                              dtype=np.float32) / 255.0
        else:
            chans = []
            for c in range(3):
                im = Image.fromarray((crop[..., c] * 255).astype(np.uint8), mode="L")
                chans.append(np.asarray(im.resize((new_w, new_h), Image.Resampling.LANCZOS),
                                        dtype=np.float32) / 255.0)
            crop = np.stack(chans, axis=-1)
        analysis_scale = new_w / rw

    channels = ["luma"]
    if rgb.ndim == 3 and deep_channels:
        channels += ["r", "g", "b"]

    # Tile spectra are computed once per channel and reused for every peak.
    tile_cache_by_channel: dict[str, list[np.ndarray]] = {}
    per: dict[str, list[Peak]] = {}
    diags: dict[str, dict] = {}
    for ch in channels:
        arr = channel_array(crop, ch)
        spectra, _, _ = _tile_log_spectra(arr, tiles=4, tile_side=320)
        tile_cache_by_channel[ch] = spectra
        peaks, diag = detect_screens(arr, channel_name=ch, tile_cache=spectra)
        for p in peaks:
            if analysis_scale != 1.0:
                p.period_px /= analysis_scale
                p.fx *= analysis_scale
                p.fy *= analysis_scale
                for q in p.lattice:
                    q["period_px"] /= analysis_scale
                    q["fx"] *= analysis_scale
                    q["fy"] *= analysis_scale
        per[ch] = peaks
        diags[ch] = {k: v for k, v in diag.items() if k not in ("logmag", "bg")}

    merged = merge_channel_peaks(per)
    flat_assist_used = False
    flat_rois: list[list[int]] = []
    if not any(p.selected for p in merged) and flat_tile_assist:
        # Auxiliary diagnostic pass on flat regions. Peaks found here are
        # reported with the same classifier and remain subject to its gates;
        # they are explicitly labeled so the operator knows the measurement ROI
        # was a flat patch rather than the whole selection.
        flat_rois = _flat_tile_candidates(crop, tile=min(320, min(crop.shape[:2])))
        assist_peaks: list[Peak] = []
        for (fx0, fy0, fw, fh) in flat_rois[:6]:
            sub = crop[fy0:fy0 + fh, fx0:fx0 + fw]
            for ch in (["luma"] if rgb.ndim == 2 else ["luma", "r", "g", "b"]):
                arr = channel_array(sub, ch)
                try:
                    peaks, _ = detect_screens(arr, channel_name=ch)
                except ValueError:
                    continue
                for p in peaks:
                    if analysis_scale != 1.0:
                        p.period_px /= analysis_scale
                        p.fx *= analysis_scale
                        p.fy *= analysis_scale
                    p.reasons.append(
                        f"measured on a flat {fw}x{fh} patch at ({fx0},{fy0}) in the ROI")
                    assist_peaks.append(p)
        if assist_peaks:
            flat_assist_used = True
            # A candidate that recurs across independent flat patches is
            # stationary print; count how many patches support it. Coverage was
            # measured inside one patch's own tiles, which is not meaningful for
            # a 256 px patch, so derive it from patch recurrence instead.
            for p in assist_peaks:
                support = sum(
                    1 for q in assist_peaks
                    if abs(q.period_px - p.period_px) / max(p.period_px, 1e-9) < 0.04
                    and abs(angle_diff(q.angle_deg, p.angle_deg)) < 4.0)
                if support >= 2:
                    p.coverage = 1.0
                    p.reasons.append(f"recurred in {support} flat patches")
            extra = merge_channel_peaks({ch: [p for p in assist_peaks if p.channel == ch]
                                         for ch in {p.channel for p in assist_peaks}})
            merged = merge_channel_peaks({
                "main": merged, "assist": extra,
            })
            for p in merged:
                if "recurred in" in " ".join(p.reasons):
                    _classify(p)
                    p.selected = p.classification == "halftone_screen"
            merged.sort(key=lambda x: (x.selected, x.confidence, x.magnitude),
                        reverse=True)

    merged, lattice_explained = reconcile_lattice_peaks(merged, crop.shape[:2])
    canonicalize_axes(merged)

    warnings: list[str] = []
    if flat_assist_used:
        warnings.append(
            "Whole-ROI spectrum did not yield a screen; a flat-region diagnostic pass "
            "found candidate peaks. Confirm they are the same screen in the spectrum view "
            "before applying.")
    for p in merged:
        if p.classification == "halftone_screen" and p.confidence < 0.8:
            warnings.append(
                f"Screen at {p.period_px:.2f} px / {p.angle_deg:.1f}° has moderate "
                f"confidence ({p.confidence:.2f}); inspect the spectrum before applying.")
        elif p.classification == "ambiguous_periodic":
            warnings.append(
                f"Periodic energy at {p.period_px:.2f} px / {p.angle_deg:.1f}° could be a "
                f"real texture (lattice score {p.coherence:.2f}); left unselected unless you "
                f"confirm it.")
        elif p.classification == "real_texture":
            warnings.append(
                f"Peak at {p.period_px:.2f} px / {p.angle_deg:.1f}° looks like real content "
                f"(no coherent lattice); not selected.")
    if not merged:
        warnings.append(
            "No narrowband screen found. Do not apply a global blur; enlarge the ROI or "
            "add a known frequency manually.")

    return {
        "roi": [x0, y0, rw, rh],
        "analyzed_shape": list(crop.shape[:2]),
        "analysis_scale": analysis_scale,
        "channels": channels,
        "selected_channel": channel if channel in channels else "luma",
        "peaks": [p.to_dict() for p in merged],
        "lattice_explained": lattice_explained,
        "per_channel": {k: [p.to_dict() for p in v] for k, v in per.items()},
        "diagnostics": diags,
        "warnings": warnings,
        "auto_selected": sum(1 for p in merged if p.selected),
        "flat_tile_assist_used": flat_assist_used,
        "flat_tiles": flat_rois,
        "manual_review_required": any(
            p.classification != "halftone_screen" and p.magnitude > 0 for p in merged),
    }


def _colorize(logmag: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(logmag, [2, 99.7])
    v = np.clip((logmag - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    stops = np.array([
        [0.00, 0.00, 0.02], [0.20, 0.02, 0.35], [0.50, 0.06, 0.55],
        [0.78, 0.30, 0.42], [0.96, 0.55, 0.14], [1.00, 0.96, 0.76],
    ], dtype=np.float32)
    x = np.linspace(0, 1, len(stops))
    out = np.empty(v.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, x, stops[:, c])
    return out


def spectrum_png(channel: np.ndarray, peaks: list[dict] | list[Peak],
                 *, output_side: int = 640, label: str = "",
                 show_grid: bool = True) -> bytes:
    arr = np.asarray(channel, dtype=np.float32)
    arr = arr - arr.mean()
    h, w = arr.shape
    mag = np.abs(np.fft.fftshift(np.fft.fft2(arr * _hann2d((h, w)))))
    logmag = np.log1p(mag)
    img = (_colorize(logmag) * 255).astype(np.uint8)
    im = Image.fromarray(img, mode="RGB").resize((output_side, output_side),
                                                Image.Resampling.BILINEAR)
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    cx = cy = output_side / 2.0

    def to_px(fx: float, fy: float) -> tuple[float, float]:
        return cx + fx * output_side, cy + fy * output_side

    if show_grid:
        for f in (0.05, 0.10, 0.20, 0.30, 0.40):
            r = f * output_side
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(72, 78, 92), width=1)
            d.text((cx + r + 2, cy + 2), f"{f:.2f}", fill=(180, 190, 205), font=small)
        d.line([cx, 0, cx, output_side], fill=(65, 70, 82), width=1)
        d.line([0, cy, output_side, cy], fill=(65, 70, 82), width=1)

    for p in peaks:
        pd = p if isinstance(p, dict) else p.to_dict()
        if pd.get("harm_of_fx") is not None:
            continue
        color = (80, 235, 140) if pd.get("selected") else (
            (255, 190, 70) if pd.get("classification") == "ambiguous_periodic"
            else (255, 100, 100))
        for s in (1, -1):
            px, py = to_px(pd["fx"] * s, pd["fy"] * s)
            if not (0 <= px < output_side and 0 <= py < output_side):
                continue
            r = 7
            d.ellipse([px - r, py - r, px + r, py + r], outline=color, width=2)
            if s == 1:
                d.text((px + 9, py - 7), f"{pd['period_px']:.1f}px",
                       fill=color, font=small)
        for q in pd.get("lattice", []):
            if not q.get("present"):
                continue
            for s in (1, -1):
                px, py = to_px(q["fx"] * s, q["fy"] * s)
                if 0 <= px < output_side and 0 <= py < output_side:
                    d.ellipse([px - 2, py - 2, px + 2, py + 2],
                              fill=(90, 200, 255))
    if label:
        d.rectangle([0, 0, output_side, 24], fill=(15, 18, 24))
        d.text((8, 4), label, fill=(230, 235, 242), font=font)
    d.text((8, output_side - 20), "DC center | rings 0.05 cyc/px",
           fill=(210, 216, 226), font=small)
    return to_png_bytes(np.asarray(im, dtype=np.float32) / 255.0)


def crop_overview_png(rgb: np.ndarray, roi: tuple[int, int, int, int],
                      *, max_side: int = 720) -> bytes:
    x0, y0, w, h = roi
    crop = rgb[y0:y0 + h, x0:x0 + w]
    im = Image.fromarray(np.clip(crop * 255 + 0.5, 0, 255).astype(np.uint8))
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * f)), max(1, int(im.height * f))),
                       Image.Resampling.NEAREST)
    return to_png_bytes(np.asarray(im, dtype=np.float32) / 255.0)
