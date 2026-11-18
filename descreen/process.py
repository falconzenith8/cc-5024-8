"""Screen suppression and resampling.

Two interchangeable, explainable implementations are provided:

* ``frequency_notch`` - zero a small pair of conjugate regions in a real FFT.
  The mask is built geometrically so the caller can print/export its exact
  center, radius, and apodization.  A reflect pad plus a feathered edge ramp
  prevents the circular convolution from creating a wrap-around seam.
* ``spatial_fir`` - the same rejection expressed as a finite, linear-phase FIR
  kernel (Gaussian envelope * cosine modulation).  This is slower but trivially
  local, with explicit support and no FFT wrap at all.

Both are applied with the *same* mask/kernel to every color channel, so
chromatic registration is preserved and no new color fringes are introduced.
Resampling uses exact area averaging (the correct anti-alias filter for
shrinking), again with identical weights per channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft


@dataclass
class NotchSpec:
    period_px: float
    angle_deg: float
    radius_cyc_per_px: float = 0.0
    apodization: float = 0.35  # fraction of radius used as raised-cosine ramp
    harmonic_order: int = 1
    center_override: tuple[float, float] | None = None  # exact (fx,fy) in cyc/px
    label: str = ""
    analysis_shape: tuple[int, int] | None = None  # for the resolution floor

    def resolved_radius(self) -> float:
        """Stopband radius in cyc/px.

        The detection was made with a Hann window, whose main lobe is 4 bins
        wide. On a small ROI one bin is a large fraction of a cycle per pixel,
        so a fixed ``0.07 / period`` would be narrower than the measured peak's
        own lobe and would leave most of the peak standing. The radius therefore
        has a floor of two FFT bins along the shorter analysis axis.

        The windows are not applied to the image, so widening to the lobe width
        does not smear anything by itself; it just guarantees the peak is
        actually inside the stopband. The upper clamp keeps a very short ROI from
        turning the notch into a low-pass.
        """
        if self.radius_cyc_per_px > 0:
            return float(self.radius_cyc_per_px)
        base = 0.07 / max(self.period_px, 1e-6)
        radius = base * (1.0 + 0.35 * (self.harmonic_order - 1))
        if self.analysis_shape:
            h, w = self.analysis_shape
            shortest = max(1, min(int(h), int(w)))
            bin_floor = 2.0 / shortest
            radius = max(radius, min(bin_floor, 0.35 / max(self.period_px, 1e-6)))
        return radius

    def center(self) -> tuple[float, float]:
        if self.center_override is not None:
            return self.center_override
        th = np.deg2rad(self.angle_deg)
        k = self.harmonic_order
        return (np.cos(th) * k / self.period_px,
                np.sin(th) * k / self.period_px)


def _fold(f: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(f) + 0.5) % 1.0 - 0.5


def build_notch_specs(
    peaks: list[dict],
    *,
    harmonic_orders: tuple[int, ...] = (1, 2),
    radius_cyc_per_px: float | None = None,
    apodization: float = 0.35,
    min_period_px: float = 2.5,
    max_period_px: float = 32.0,
    include_second_axis: bool = True,
    include_lattice_lines: bool = True,
    lattice_rel_threshold: float = 0.45,
    analysis_shape: tuple[int, int] | None = None,
) -> tuple[list[NotchSpec], list[dict]]:
    """Turn accepted peaks into notch specs, folding harmonics at Nyquist.

    A detected square-dot screen carries two primitive axes 90 deg apart; both
    must be notched, otherwise the second axis rebuilds the dot pattern. For a
    square lattice the second axis has the same pitch, so its harmonics are
    generated the same way from ``basis_b``.

    Returns specs plus audit notes (e.g. harmonics skipped because they alias
    onto DC or fall outside the analysis band).
    """
    specs: list[NotchSpec] = []
    notes: list[dict] = []

    def add_axis(period: float, angle: float, axes_label: str) -> None:
        for k in harmonic_orders:
            effective_period = period / k
            if effective_period < min_period_px:
                notes.append({"period_px": period, "angle_deg": angle,
                              "harmonic_order": k, "axis": axes_label,
                              "skipped": "above analysis band"})
                continue
            if effective_period > max_period_px:
                notes.append({"period_px": period, "angle_deg": angle,
                              "harmonic_order": k, "axis": axes_label,
                              "skipped": "below analysis band"})
                continue
            th = np.deg2rad(angle)
            fx = np.cos(th) * k / period
            fy = np.sin(th) * k / period
            folded = (abs(float(_fold(fx))) < 1e-6 and abs(float(_fold(fy))) < 1e-6)
            if folded:
                notes.append({"period_px": period, "angle_deg": angle,
                              "harmonic_order": k, "axis": axes_label,
                              "skipped": "folds onto DC"})
                continue
            r = radius_cyc_per_px if radius_cyc_per_px is not None else 0.0
            specs.append(NotchSpec(
                period_px=period,
                angle_deg=angle,
                radius_cyc_per_px=r,
                apodization=apodization,
                harmonic_order=k,
                label=f"{axes_label}{k}",
                analysis_shape=analysis_shape,
            ))
            notes.append({"period_px": period, "angle_deg": angle,
                          "harmonic_order": k, "axis": axes_label,
                          "center_fx": float(_fold(fx)),
                          "center_fy": float(_fold(fy)), "applied": True})

    for p in peaks:
        add_axis(float(p["period_px"]), float(p["angle_deg"]), "a")
        basis_b = p.get("basis_b")
        if include_second_axis and basis_b:
            # Use the measured second axis when it is genuinely orthogonal;
            # otherwise the square-lattice assumption does not hold and a
            # single-axis notch is the honest choice.
            da = float(np.rad2deg(np.arctan2(basis_b["fy"], basis_b["fx"])))
            diff = float((da - float(p["angle_deg"])) % 180.0)
            diff = abs(diff - 180.0 if diff > 90.0 else diff)
            if 70.0 <= diff <= 110.0 and basis_b.get("period_px"):
                add_axis(float(basis_b["period_px"]),
                         float(basis_b["angle_deg"]), "b")
        if include_lattice_lines:
            # Diagonal (1,1) and (2,1) lattice lines are part of the same dot
            # pattern and carry real energy; suppressing only the two axes
            # leaves visible cross-hatching. They are measured points, so notch
            # their actual centers rather than assuming ideal geometry.
            seen: set[tuple[float, float]] = set()
            for q in p.get("lattice", []):
                if q.get("m", 0) == 0 or q.get("n", 0) == 0:
                    continue  # primary axes already covered
                if abs(int(q.get("m", 0))) + abs(int(q.get("n", 0))) <= 1:
                    continue
                if not q.get("present", False):
                    continue
                if float(q.get("rel_magnitude", 0.0)) < lattice_rel_threshold:
                    continue
                qp = float(q.get("period_px", 0.0))
                if qp < min_period_px or qp > max_period_px:
                    continue
                key = (round(q["fx"] * 1e4), round(q["fy"] * 1e4))
                if key in seen:
                    continue
                seen.add(key)
                specs.append(NotchSpec(
                    period_px=qp,
                    angle_deg=float(q["angle_deg"]),
                    radius_cyc_per_px=(radius_cyc_per_px
                                       if radius_cyc_per_px is not None else 0.0),
                    apodization=apodization,
                    harmonic_order=1,
                    center_override=(float(q["fx"]), float(q["fy"])),
                    label=f"lattice({q['m']},{q['n']})",
                    analysis_shape=analysis_shape,
                ))
                notes.append({"period_px": qp, "angle_deg": float(q["angle_deg"]),
                              "harmonic_order": 1, "axis": "lattice",
                              "lattice_mn": [q["m"], q["n"]],
                              "center_fx": float(_fold(q["fx"])),
                              "center_fy": float(_fold(q["fy"])),
                              "rel_magnitude": float(q["rel_magnitude"]),
                              "applied": True})

    # The same physical line can be reported by more than one merged screen
    # observation (per-channel and flat-patch passes), and a line at center c is
    # the same physical line as one at -c. The mask covers the conjugate pair,
    # so notch each distinct center once, canonically folded. Use a distance
    # tolerance rather than rounding: two estimates of one line can differ by a
    # fraction of a bin and still be the same line, and exact-key dedupe would
    # keep both (which corrupts the additive FIR kernel; see _fir_kernel).
    unique: list[NotchSpec] = []
    for s in specs:
        cfx, cfy = (float(_fold(v)) for v in s.center())
        if cfx < -1e-12 or (abs(cfx) <= 1e-12 and cfy < 0.0):
            cfx, cfy = -cfx, -cfy
        r = s.resolved_radius()
        tol = max(0.25 * r, 1e-5)
        for i, u in enumerate(unique):
            ufx, ufy = (float(_fold(v)) for v in u.center())
            if ufx < -1e-12 or (abs(ufx) <= 1e-12 and ufy < 0.0):
                ufx, ufy = -ufx, -ufy
            if np.hypot(cfx - ufx, cfy - ufy) <= tol:
                # Keep the wider notch; it covers the narrower one.
                if r > u.resolved_radius():
                    unique[i] = s
                break
        else:
            unique.append(s)
    return unique, notes


def _notch_mask(shape: tuple[int, int], specs: list[NotchSpec],
                *, pad: int) -> tuple[np.ndarray, list[dict]]:
    """Build a multiplicative notch mask on a real-FFT grid (H x W//2+1)."""
    h, w = shape
    fy = np.fft.fftfreq(h).astype(np.float32)
    fx = np.fft.rfftfreq(w).astype(np.float32)
    FX, FY = np.meshgrid(fx, fy)
    mask = np.ones(FX.shape, dtype=np.float32)
    audit: list[dict] = []
    for spec in specs:
        r = spec.resolved_radius()
        cfx, cfy = spec.center()
        cfx_f = float(_fold(cfx))
        cfy_f = float(_fold(cfy))
        d = np.hypot(FX - cfx_f, FY - cfy_f)
        d = np.minimum(d, np.hypot(FX + cfx_f, FY + cfy_f))
        inner = r * (1.0 - spec.apodization)
        outer = r
        m = np.ones_like(mask)
        m[d <= inner] = 0.0
        ramp = (d > inner) & (d < outer)
        if np.any(ramp):
            t = (d[ramp] - inner) / max(outer - inner, 1e-12)
            m[ramp] = 0.5 - 0.5 * np.cos(np.pi * t)
        mask *= m
        audit.append({
            "period_px": spec.period_px,
            "angle_deg": spec.angle_deg,
            "harmonic_order": spec.harmonic_order,
            "center_fx": cfx_f,
            "center_fy": cfy_f,
            "center_x_bin": float(((cfx_f % 1.0) * w)),
            "center_y_bin": float(((cfy_f % 1.0) * h)),
            "radius_cyc_per_px": float(r),
            "radius_bins_x": float(r * w),
            "radius_bins_y": float(r * h),
            "apodization": spec.apodization,
            "stopband_cyc_per_px": [float(inner), float(outer)],
            "pad_px": pad,
        })
    return mask, audit


def _pad_reflect(arr: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return arr
    if arr.ndim == 2:
        return np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
    return np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


def _edge_feather(shape: tuple[int, int], feather: int) -> np.ndarray:
    """Smooth 1 at center -> 0 at the outer border, for seam-free blending."""
    h, w = shape
    if feather <= 0:
        return np.ones(shape, dtype=np.float32)
    yy = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    xx = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
    t = np.minimum(yy[:, None], xx[None, :]) / float(feather)
    t = np.clip(t, 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def _default_pad(specs: list[NotchSpec]) -> int:
    max_period = max(s.period_px for s in specs)
    return int(np.clip(np.ceil(max(24.0, 3.0 * max_period)), 24, 256))


def frequency_notch(
    image: np.ndarray,
    specs: list[NotchSpec],
    *,
    pad: int | None = None,
    feather: int | None = None,
    strength: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Apply zero-phase frequency notches to all channels identically."""
    arr = np.asarray(image, dtype=np.float32)
    if not specs:
        return arr.copy(), {"method": "frequency_notch", "notches": [],
                            "note": "no notches requested"}
    h, w = arr.shape[:2]
    pad = _default_pad(specs) if pad is None else int(pad)
    feather = max(8, pad // 2) if feather is None else int(feather)
    pad = int(min(max(pad, 0), max(8, max(h, w) // 2)))
    feather = int(min(feather, max(h, w) // 4))
    padded_shape = (h + 2 * pad, w + 2 * pad)
    mask, audit = _notch_mask(padded_shape, specs, pad=pad)
    gain = 1.0 - strength * (1.0 - mask)

    def filt_plane(plane: np.ndarray) -> np.ndarray:
        src = _pad_reflect(plane, pad)
        spec = sfft.rfft2(src)
        filt = sfft.irfft2(spec * gain, s=src.shape).astype(np.float32)
        f = _edge_feather((h, w), feather)
        return plane + (filt[pad:pad + h, pad:pad + w] - plane) * f

    if arr.ndim == 2:
        out = filt_plane(arr)
    else:
        out = np.stack([filt_plane(arr[..., c]) for c in range(3)], axis=-1)
    out = np.clip(out, 0.0, 1.0)
    audit_dict = {
        "method": "frequency_notch",
        "channels_processed": "all (identical mask)" if arr.ndim == 3 else "luminance",
        "pad_px": pad,
        "padding_mode": "reflect",
        "edge_feather_px": feather,
        "strength": float(strength),
        "notches": audit,
        "blend": "reflect-pad FFT -> crop -> cosine feather to original at border",
        "notes": [
            "Reflect padding supplies continuity; the mask is never applied to the "
            "unpadded array, so there is no circular wrap seam.",
            f"The outer {feather} px ramp toward the original is an explicit, "
            "exportable compromise for boundary continuity.",
        ],
    }
    return out, audit_dict


def _fir_kernel(specs: list[NotchSpec], *, taps: int | None = None
                ) -> tuple[np.ndarray, dict]:
    """Finite linear-phase notch kernel: delta - 2 * sum_i envelope_i * cos_i.

    Each notch is a cosine-modulated Gaussian envelope. Its Fourier transform is
    a Gaussian of standard deviation ``sigma_f`` at +/- the notch center, so the
    stopband width is set explicitly in cycles/px, not by the tap count. The
    factor 2 is required: the modulation splits the envelope into two halves of
    amplitude 1/2 at each conjugate center, so delta - 1*envelope would leave
    50% gain (a 6 dB dip, not a notch). Support is exactly `taps` pixels.

    A Gaussian envelope is used rather than a hard disk because a truncated
    disk rings (Gibbs) and its ringing is itself a periodic artifact.
    """
    if taps is None:
        r_min = min(s.resolved_radius() for s in specs)
        sigma_px = 0.1874 / max(r_min, 1e-6)
        taps = int(np.clip(2 * int(np.ceil(4.0 * sigma_px)) + 1, 31, 601))
    taps = int(taps) | 1
    cy = cx = taps // 2
    yy, xx = np.mgrid[0:taps, 0:taps].astype(np.float64)
    gy = yy - cy
    gx = xx - cx
    r_min = min(s.resolved_radius() for s in specs)
    sigma_px = 0.1874 / max(r_min, 1e-6)
    env = np.exp(-(gx * gx + gy * gy) / (2.0 * sigma_px * sigma_px))
    # Normalize the envelope to unit DC response so delta - 2*env*cos cancels
    # exactly at each conjugate center (the factor 2 and this normalization
    # together give a true null, not a 50% dip).
    env /= max(env.sum(), 1e-12)
    # The kernel is a *sum* of cosines, so two near-coincident centers (a screen
    # axis reported twice, or an axis and a measured lattice line within a
    # fraction of a bin) add two almost-opposite-phase cosines and cancel each
    # other's null, restoring gain ~1 at a frequency that was supposed to be
    # removed. The multiplicative frequency mask does not suffer this. Collapse
    # centers closer than a small fraction of the narrowest radius before
    # accumulating; the kernel is linear-phase in each center, so the merged
    # center is the amplitude-weighted mean.
    centers: list[tuple[float, float]] = []
    min_r = min(s.resolved_radius() for s in specs)
    merge_tol = max(0.02 * min_r, 1e-5)
    for sp in specs:
        cfx, cfy = (float(_fold(v)) for v in sp.center())
        # A notch at c and one at -c are the same physical line: cos is even.
        # Adding both cosines doubles the kernel term and turns the intended
        # null into gain -1, so fold to one half-plane before merging.
        if cfx < -1e-12 or (abs(cfx) <= 1e-12 and cfy < 0.0):
            cfx, cfy = -cfx, -cfy
        for i, (px, py) in enumerate(centers):
            if np.hypot(cfx - px, cfy - py) <= merge_tol:
                centers[i] = ((px + cfx) / 2.0, (py + cfy) / 2.0)
                break
        else:
            centers.append((cfx, cfy))
    kernel = np.zeros((taps, taps), dtype=np.float64)
    for cfx, cfy in centers:
        kernel += env * np.cos(2 * np.pi * (cfx * gx + cfy * gy))
    delta = np.zeros((taps, taps), dtype=np.float64)
    delta[cy, cx] = 1.0
    kernel = delta - 2.0 * kernel
    kernel /= kernel.sum()

    n = 1024
    resp = np.fft.fftshift(np.abs(np.fft.fft2(kernel, s=(n, n))))
    freqs = np.fft.fftshift(np.fft.fftfreq(n))
    FX, FY = np.meshgrid(freqs, freqs)

    def gain_at(fx: float, fy: float) -> float:
        ix = int(round((float(_fold(fx)) + 0.5) * n)) % n
        iy = int(round((float(_fold(fy)) + 0.5) * n)) % n
        return float(resp[iy, ix])

    per_notch = []
    worst = 0.0
    for sp in specs:
        cfx, cfy = sp.center()
        cfx, cfy = float(_fold(cfx)), float(_fold(cfy))
        r = sp.resolved_radius()
        d = np.minimum(np.hypot(FX - cfx, FY - cfy), np.hypot(FX + cfx, FY + cfy))
        # The requested radius is the half-gain radius of the Gaussian notch.
        # Report the gain inside that disk as realized (a Gaussian cannot null
        # a whole disk; it nulls the center and rolls off).
        band = d <= 0.35 * r
        band_max = float(resp[band].max()) if np.any(band) else 0.0
        worst = max(worst, band_max)
        # 50%-rejection radius measured along the notch axis.
        th = np.arctan2(cfy, cfx)
        rad50 = r
        for step in np.linspace(0.0, 4.0 * r, 161):
            if gain_at(cfx + step * np.cos(th),
                       cfy + step * np.sin(th)) > 0.5:
                rad50 = step
                break
        per_notch.append({
            "period_px": sp.period_px,
            "angle_deg": sp.angle_deg,
            "harmonic_order": sp.harmonic_order,
            "label": sp.label,
            "center_gain": gain_at(cfx, cfy),
            "stopband_max_gain": band_max,
            "half_gain_radius_cyc_per_px": float(rad50),
            "requested_radius_cyc_per_px": float(r),
        })
    return kernel, {
        "taps": taps,
        "envelope_sigma_px": float(sigma_px),
        "max_stopband_gain": worst,
        "dc_gain": float(resp[n // 2, n // 2]),
        "notches": per_notch,
        "note": "Gain is measured from the FFT of the actual finite kernel, so "
                "the reported rejection is the realized one, not the design target.",
    }


def spatial_fir(
    image: np.ndarray,
    specs: list[NotchSpec],
    *,
    taps: int | None = None,
    strength: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Explicit-support FIR notch (local, no FFT wrap)."""
    arr = np.asarray(image, dtype=np.float32)
    if not specs:
        return arr.copy(), {"method": "spatial_fir", "notches": []}
    kernel, kval = _fir_kernel(specs, taps=taps)
    taps = int(kval["taps"])
    h, w = arr.shape[:2]
    # Pad by exactly the kernel half-support: less would clip the finite
    # convolution at the border; more only adds cost.
    pad = taps // 2
    # The feather is an explicit boundary compromise, but it must stay a small
    # fraction of the image. Tying it to the kernel size alone (taps//2) makes
    # it cover most of a small ROI and leaves the centre effectively unfiltered.
    feather = int(min(max(8, pad // 2), max(8, min(h, w) // 8)))
    _, audit = _notch_mask((h + 2 * pad, w + 2 * pad), specs, pad=pad)

    def filt_plane(plane: np.ndarray) -> np.ndarray:
        src = _pad_reflect(plane, pad)
        # The kernel has exact finite support. Zero-padded FFT convolution is
        # used only as an accelerator for that finite-support convolution: it
        # is mathematically identical to the direct sum, with no circular wrap,
        # because the transform length covers src + kernel - 1.
        from scipy.signal import fftconvolve
        filtered = fftconvolve(src, kernel, mode="same")
        base = src + strength * (filtered - src)
        f = _edge_feather((h, w), feather)
        out = plane + (base[pad:pad + h, pad:pad + w] - plane) * f
        return out

    if arr.ndim == 2:
        out = filt_plane(arr)
    else:
        out = np.stack([filt_plane(arr[..., c]) for c in range(3)], axis=-1)
    out = np.clip(out, 0.0, 1.0)
    return out, {
        "method": "spatial_fir",
        "kernel_taps": taps,
        "support_px": taps,
        "pad_px": pad,
        "padding_mode": "reflect + mirror convolution",
        "edge_feather_px": feather,
        "channels_processed": "all (identical kernel)" if arr.ndim == 3 else "luminance",
        "strength": float(strength),
        "notches": audit,
        "realized_filter": kval,
        "blend": "reflect pad, symmetric convolution, cosine feather at border",
    }


def coordinated_filter(
    image: np.ndarray,
    weights: dict,
    groups: dict[str, dict],
    region_group: dict[str, str],
    *,
    base_group: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Composite whole-page filtered candidates with smooth region weights.

    Each distinct parameter group is filtered over the *entire* image first.
    That is what keeps pixels just across a region boundary honest: the filter
    support always reads real neighbouring pixels, never a separately filtered
    block.  The candidates are then blended by normalized weights.

    ``groups`` maps group id -> ``{"method", "specs", "strength", "pad",
    "feather", "taps"}``.  ``region_group`` maps region id -> group id.
    ``base_group`` optionally filters the uncovered base layer; ``None`` leaves
    it as the original (identity) image.
    """
    arr = np.asarray(image, dtype=np.float32)
    h, w = arr.shape[:2]
    if weights["base"].shape != (h, w):
        raise ValueError("weights shape does not match the image")

    candidates: dict[str, np.ndarray] = {}
    audits: dict[str, dict] = {}
    for gid, g in groups.items():
        specs = g.get("specs") or []
        method = g.get("method", "frequency_notch")
        strength = float(g.get("strength", 1.0))
        if not specs or strength <= 0.0:
            candidates[gid] = arr.copy()
            audits[gid] = {"method": method, "notches": [],
                           "note": "identity candidate: no notch or zero strength"}
            continue
        if method == "spatial_fir":
            cand, audit = spatial_fir(arr, specs, taps=g.get("taps"),
                                      strength=strength)
        elif method == "frequency_notch":
            cand, audit = frequency_notch(arr, specs, pad=g.get("pad"),
                                          feather=g.get("feather"),
                                          strength=strength)
        else:
            raise ValueError(f"unknown method {method!r}")
        candidates[gid] = cand
        audits[gid] = audit

    if base_group is not None:
        base_candidate = candidates[base_group]
    else:
        base_candidate = arr

    out = base_candidate * weights["base"][..., None] if arr.ndim == 3 else (
        base_candidate * weights["base"])
    per_region_weight: dict[str, float] = {}
    for rid, wgt in weights["per_region"].items():
        gid = region_group.get(rid)
        if gid is None:
            raise ValueError(f"region {rid!r} has no filter group")
        cand = candidates[gid]
        out = out + (cand * wgt[..., None] if arr.ndim == 3 else cand * wgt)
        per_region_weight[rid] = float(wgt.mean())
    out = np.clip(out.astype(np.float32), 0.0, 1.0)

    group_rows = []
    for gid, g in groups.items():
        used = [rid for rid, rg in region_group.items() if rg == gid]
        group_rows.append({
            "group": gid,
            "method": g.get("method", "frequency_notch"),
            "strength": float(g.get("strength", 1.0)),
            "regions": used,
            "notch_count": len(g.get("specs") or []),
        })
    return out, {
        "mode": "region_coordinated",
        "groups": group_rows,
        "candidate_passes": len(candidates),
        "base_group": base_group,
        "weight_sum_range": [weights["sum_min"], weights["sum_max"]],
        "base_fraction": weights["base_fraction"],
        "mean_region_weight": per_region_weight,
        "group_audits": audits,
        "note": "Every group was filtered over the whole image; the composite is "
                "a normalized weighted blend. No region was filtered in isolation.",
    }


def _area_weights(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact area-overlap spans for one axis.

    Returns (start_index, end_index, weights) for each output cell. Boundaries
    are fractional, so a shrink by any ratio is correctly low-passed.
    """
    edges = np.linspace(0.0, n_in, n_out + 1, dtype=np.float64)
    starts = np.floor(edges[:-1]).astype(np.int64)
    ends = np.ceil(edges[1:]).astype(np.int64)
    counts = ends - starts
    max_count = int(counts.max())
    idx = starts[:, None] + np.arange(max_count)[None, :]
    valid = (idx < ends[:, None]) & (idx < n_in)
    idx_clipped = np.clip(idx, 0, n_in - 1)
    lo = np.maximum(idx.astype(np.float64), edges[:-1, None])
    hi = np.minimum(idx.astype(np.float64) + 1.0, edges[1:, None])
    w = np.where(valid, np.clip(hi - lo, 0.0, None), 0.0)
    w /= w.sum(axis=1, keepdims=True)
    return idx_clipped, valid, w


def _resize_axis(arr: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    n_in = arr.shape[axis]
    idx, valid, w = _area_weights(n_in, n_out)
    moved = np.moveaxis(arr, axis, 0)
    gathered = moved[idx]  # (n_out, span, ...)
    w_shape = (n_out, idx.shape[1]) + (1,) * (moved.ndim - 1)
    out = (gathered * w.reshape(w_shape)).sum(axis=1)
    return np.moveaxis(out, 0, axis)


def area_resize(image: np.ndarray, scale: float) -> tuple[np.ndarray, dict]:
    """Anti-aliased area-average downscale by a linear factor.

    Area averaging is the correct low-pass for shrinking. It is applied along
    each axis with identical per-channel weights, preserving registration.
    Upscaling is refused rather than silently aliasing.
    """
    arr = np.asarray(image, dtype=np.float32)
    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale > 1.0 + 1e-9:
        raise ValueError("area_resize only downsamples; upscaling is out of scope")
    if abs(scale - 1.0) < 1e-9:
        return arr.copy(), {"scale": 1.0, "output_shape": list(arr.shape[:2])}
    h, w = arr.shape[:2]
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    tmp = _resize_axis(arr, nh, axis=0)
    out = _resize_axis(tmp, nw, axis=1)
    out = np.clip(out.astype(np.float32), 0.0, 1.0)
    return out, {
        "method": "area_average",
        "requested_scale": float(scale),
        "output_shape": [nh, nw],
        "prefilter": "exact fractional area box (per channel identical)",
        "note": "Area averaging is applied after screen suppression; it also "
                "prevents any leftover high frequency from aliasing into new moire.",
    }


def resample_plan(
    scale: float,
    *,
    order: str = "suppress_then_resize",
) -> dict:
    return {
        "scale": float(scale),
        "order": order,
        "reason": "Suppression first removes the scanned screen; area resampling "
                  "then prevents residual high frequency from aliasing into a new "
                  "lower-frequency moire." if order == "suppress_then_resize" else
                  "Resampling first moves the screen frequency; suppression must then "
                  "target the moved peak. This is supported only with re-analysis.",
    }
