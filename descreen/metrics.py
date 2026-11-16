"""Objective measurements on real runs.

Two families of measurements are reported, and both are required for judging a
result.  "Less texture" alone is not success:

1. Screen peak suppression: the same narrowband peaks that were detected before
   processing, re-measured after.  A real screen removal must drop them.
2. Content preservation: edge sharpness, fine-line contrast, and local detail
   energy measured on the *content* regions, compared with the untreated input.
   A global blur passes (1) but fails these.
3. Moire risk after resampling: compare the energy of the resampled output with
   the source block-averaged to the same size.  Output must not carry extra
   low-frequency beats that the reference does not.

All metrics are computed from actual pipeline outputs, never from a pre-fixed
"after" image.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def _hann2d(shape: tuple[int, int]) -> np.ndarray:
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1])).astype(np.float32)


def _channel_luma(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    coeff = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return (img[..., :3].astype(np.float32) @ coeff).clip(0.0, 1.0)


def _dtft_amplitude(arr: np.ndarray, fx: float, fy: float,
                    win: np.ndarray | None = None) -> float:
    """Hann-windowed DTFT magnitude at exactly (fx, fy), in cycles/pixel.

    This is a matched filter for the screen line: unlike taking the maximum of
    the nearest FFT bins, it evaluates the continuous spectrum at the detected
    frequency, so it neither misses the peak (fractional-bin split) nor reads
    energy from a neighbouring feature just outside the notch.
    """
    h, w = arr.shape
    win = _hann2d((h, w)) if win is None else win
    a = (arr.astype(np.float32) - float(arr.mean())) * win
    ey = np.exp(-2j * np.pi * fy * np.arange(h))
    ex = np.exp(-2j * np.pi * fx * np.arange(w))
    return float(abs(ey @ a @ ex) / max(float(win.sum()), 1e-12))


def _local_floor(arr: np.ndarray, fx: float, fy: float, win: np.ndarray,
                 *, exclude_deg: float = 20.0) -> float:
    """Median DTFT amplitude on a ring at the same radius as the peak.

    Frequencies within +-``exclude_deg`` of the screen axis (both directions)
    are skipped so the peak's own lobe cannot masquerade as floor. The result
    is the local broadband/leakage background the peak stands above.
    """
    r = math.hypot(fx, fy)
    if r < 1e-6:
        return 0.0
    th0 = math.degrees(math.atan2(fy, fx))
    vals = []
    for dth in range(0, 180, 15):
        th = math.radians(th0 + dth)
        delta = (dth + 180.0) % 180.0
        if delta < exclude_deg or delta > 180.0 - exclude_deg:
            continue
        vals.append(_dtft_amplitude(arr, r * math.cos(th), r * math.sin(th), win))
    if not vals:
        return 0.0
    return float(np.median(vals))


def peak_amplitude(img: np.ndarray, fx: float, fy: float) -> float:
    """Matched-filter amplitude of the screen line at exactly (fx, fy)."""
    arr = _channel_luma(img)
    return _dtft_amplitude(arr, fx, fy, _hann2d(arr.shape))


def _lobe_max_db(img: np.ndarray, fx: float, fy: float,
                 *, radius_bins: float = 2.0) -> float:
    """Diagnostic only: max log-magnitude within +-2 bins, above the median."""
    arr = _channel_luma(img)
    h, w = arr.shape
    spec = np.fft.fft2((arr - float(arr.mean())) * _hann2d((h, w)))
    logmag = np.log1p(np.abs(spec))
    xi = int(round(fx * w)) % w
    yi = int(round(fy * h)) % h
    xs = np.arange(xi - 2, xi + 3) % w
    ys = np.arange(yi - 2, yi + 3) % h
    return float(logmag[np.ix_(ys, xs)].max() - float(np.median(logmag)))


def peak_suppression(before: np.ndarray, after: np.ndarray,
                     peaks: list[dict]) -> dict:
    """Change in matched-filter amplitude of each reported screen line.

    ``reduction_db`` is the ratio of the DTFT amplitude at the exact screen
    frequency before and after. That is the direct, unambiguous measure of the
    periodic peak changing. ``lobe_max_*`` is reported alongside as a coarse
    check that no lobe was merely shifted sideways; it is not the headline
    number because on a small ROI its +-2-bin window includes broadband content
    that exists with or without a screen.
    """
    rows = []
    for p in peaks:
        fx, fy = float(p["fx"]), float(p["fy"])
        win_b = _hann2d(_channel_luma(before).shape)
        win_a = _hann2d(_channel_luma(after).shape)
        lb = _channel_luma(before)
        la = _channel_luma(after)
        amp_b = _dtft_amplitude(lb, fx, fy, win_b)
        amp_a = _dtft_amplitude(la, fx, fy, win_a)
        floor_b = _local_floor(lb, fx, fy, win_b)
        floor_a = _local_floor(la, fx, fy, win_a)
        per = {
            "period_px": p["period_px"],
            "angle_deg": p["angle_deg"],
            "amplitude_before": amp_b,
            "amplitude_after": amp_a,
            "local_floor_before": floor_b,
            "local_floor_after": floor_a,
            "snr_before_db": float(20.0 * math.log10(
                max(amp_b, 1e-12) / max(floor_b, 1e-12))),
            "snr_after_db": float(20.0 * math.log10(
                max(amp_a, 1e-12) / max(floor_a, 1e-12))),
            "lobe_max_before": _lobe_max_db(before, fx, fy),
            "lobe_max_after": _lobe_max_db(after, fx, fy),
        }
        per["reduction_db"] = float(20.0 * math.log10(
            max(amp_b, 1e-12) / max(amp_a, 1e-12)))
        per["reduction_ratio"] = float(1.0 - amp_a / max(amp_b, 1e-12))
        # Also measure the conjugate peak so a unilateral mask is not reported
        # as success.
        per["amplitude_after_conjugate"] = _dtft_amplitude(la, -fx, -fy, win_a)
        rows.append(per)
    if not rows:
        return {"peaks": [], "note": "no peaks were selected"}
    mean_db = float(np.mean([r["reduction_db"] for r in rows]))
    min_db = float(min(r["reduction_db"] for r in rows))
    return {
        "peaks": rows,
        "mean_reduction_db": mean_db,
        "min_reduction_db": min_db,
        "all_reduced": min_db > 3.0,
        "note": "Reduction is the before/after ratio of a Hann-windowed DTFT "
                "evaluated at the exact detected screen frequency (matched "
                "filter), not the max of nearby FFT bins. 3 dB is the minimum "
                "meaningful drop; 12 dB indicates a clean notch. The conjugate "
                "amplitude is reported so a one-sided mask cannot pass.",
    }


def gradient_sharpness(img: np.ndarray) -> float:
    """Mean gradient magnitude of the top-decile edges (edge-based sharpness)."""
    y = _channel_luma(img)
    gx = ndimage.sobel(y, axis=1)
    gy = ndimage.sobel(y, axis=0)
    g = np.hypot(gx, gy)
    hi = np.percentile(g, 90)
    strong = g[g >= hi]
    return float(strong.mean() if strong.size else 0.0)


def edge_contrast(img: np.ndarray) -> float:
    """RMS contrast in the highest-gradient 5% of pixels (edge acutance)."""
    y = _channel_luma(img)
    gx = ndimage.sobel(y, axis=1)
    gy = ndimage.sobel(y, axis=0)
    g = np.hypot(gx, gy)
    thr = np.percentile(g, 95)
    mask = g >= thr
    if mask.sum() < 16:
        return 0.0
    # Local contrast measured as |pixel - local mean| on edges.
    blur = ndimage.gaussian_filter(y, 2.0)
    return float(np.abs(y - blur)[mask].mean())


def hairline_contrast(img: np.ndarray) -> float:
    """Peak-to-peak amplitude of high-frequency line content via DoG.

    A hairline produces a strong difference-of-Gaussians response; a low-pass
    that blurs it away reduces this metric.
    """
    y = _channel_luma(img)
    fine = y - ndimage.gaussian_filter(y, 1.2)
    return float(np.percentile(np.abs(fine), 99.0))


def detail_energy(img: np.ndarray, *, low_sigma: float = 1.0,
                  high_sigma: float = 3.0) -> float:
    """Band-pass detail energy outside the screen band (texture retention)."""
    y = _channel_luma(img)
    band = ndimage.gaussian_filter(y, low_sigma) - ndimage.gaussian_filter(y, high_sigma)
    return float(np.std(band))


def content_preservation(before: np.ndarray, after: np.ndarray) -> dict:
    b = {
        "edge_sharpness": gradient_sharpness(before),
        "edge_contrast": edge_contrast(before),
        "hairline_contrast": hairline_contrast(before),
        "detail_energy": detail_energy(before),
    }
    a = {
        "edge_sharpness": gradient_sharpness(after),
        "edge_contrast": edge_contrast(after),
        "hairline_contrast": hairline_contrast(after),
        "detail_energy": detail_energy(after),
    }
    delta = {}
    for k in b:
        base = max(abs(b[k]), 1e-9)
        delta[k] = {
            "before": float(b[k]), "after": float(a[k]),
            "retained_fraction": float(a[k] / base),
        }
    worst = min(v["retained_fraction"] for v in delta.values())
    return {
        "metrics": delta,
        "worst_retained_fraction": float(worst),
        "note": "Retained fraction near 1.0 means content is preserved. Values "
                "below ~0.85 on edge/hairline metrics indicate over-smoothing.",
    }


def moire_after_resample(input_img: np.ndarray, output_img: np.ndarray,
                         scale: float) -> dict:
    """Compare low-frequency beat energy with an exact area reference.

    The reference is the *input* block-averaged with the same area weights and
    no screen suppression. Any extra low-frequency energy in the actual output
    is a candidate for newly introduced moire.
    """
    from .process import area_resize

    ref, _ = area_resize(input_img, scale)
    out = np.asarray(output_img, dtype=np.float32)
    h = min(ref.shape[0], out.shape[0])
    w = min(ref.shape[1], out.shape[1])
    ref = ref[:h, :w]
    out = out[:h, :w]
    ref_y = _channel_luma(ref)
    out_y = _channel_luma(out)

    # Low-frequency beat band: below the original screen frequency, where
    # aliases and beat patterns appear.
    def beat_energy(y: np.ndarray) -> float:
        low = ndimage.gaussian_filter(y, 6.0)
        return float(np.std(y - low))

    e_ref = beat_energy(ref_y)
    e_out = beat_energy(out_y)
    # Ratio of band energy; values >1 mean the output has extra structure.
    ratio = float(e_out / max(e_ref, 1e-9))
    return {
        "reference_energy": e_ref,
        "output_energy": e_out,
        "ratio": ratio,
        "added_moire_risk": ratio > 1.12,
        "note": "Reference is area-averaged input with no screen removal. If the "
                "output carries more low-frequency beat energy than this reference, "
                "new moire is likely.",
    }


def content_vs_reference(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """Compare a processed image against the *clean* unscreened ground truth.

    Comparing a descreened output with its screened input penalizes the tool for
    removing the dots: the dots contribute most of the input's high-frequency
    energy, so a correct removal looks like "detail loss". The honest reference
    is the same scene without a screen. Then a drop in edge acutance or hairline
    contrast is real damage, not removal of the artifact.

    ``reference`` must be the same size as ``candidate``.
    """
    a = _channel_luma(candidate)
    b = _channel_luma(reference)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    ref_gx = ndimage.sobel(b, axis=1)
    ref_gy = ndimage.sobel(b, axis=0)
    ref_g = np.hypot(ref_gx, ref_gy)
    # Strongest edges: the lines and letters that matter. A percentile alone is
    # unsafe on artwork dominated by flat fills (95% of pixels can be nearly
    # zero-gradient, which would make the "strong edge" set the whole image).
    # Require both a high percentile and a fraction of the peak gradient.
    thr = max(float(np.percentile(ref_g, 95)), 0.12 * float(ref_g.max()))
    mask = ref_g >= thr
    if mask.sum() < 16:
        mask = ref_g >= 0.5 * float(ref_g.max())

    cand_gx = ndimage.sobel(a, axis=1)
    cand_gy = ndimage.sobel(a, axis=0)
    cand_g = np.hypot(cand_gx, cand_gy)

    edge_true = float(ref_g[mask].mean())
    edge_got = float(cand_g[mask].mean())

    # Local-contrast metric on the same edge set: a notch that smears an edge
    # reduces |pixel - local mean| there.
    blur_b = ndimage.gaussian_filter(b, 2.0)
    blur_a = ndimage.gaussian_filter(a, 2.0)
    acut_true = float(np.abs(b - blur_b)[mask].mean())
    acut_got = float(np.abs(a - blur_a)[mask].mean())

    diff = a - b
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    psnr = float(20.0 * math.log10(1.0 / max(rmse, 1e-9)))

    return {
        "edge_slope_retention": edge_got / max(edge_true, 1e-9),
        "edge_acutance_retention": acut_got / max(acut_true, 1e-9),
        "rmse_vs_clean": rmse,
        "psnr_vs_clean_db": psnr,
        "edge_pixels_measured": int(mask.sum()),
        "note": "Retention here is output-vs-clean-truth, so removing the screen "
                "does not count as loss. Values near 1.0 mean the edges survived.",
    }


def color_registration(rgb: np.ndarray) -> dict:
    """Channel alignment via phase correlation against the luma guide.

    Returns the integer-ish shift of R and B relative to G.  A channel-identical
    processing path must not increase misregistration.
    """
    if rgb.ndim != 3:
        return {"note": "single-channel image"}
    g = rgb[..., 1].astype(np.float32)
    out = {}
    for name, idx in (("r", 0), ("b", 2)):
        ch = rgb[..., idx].astype(np.float32)
        # Phase correlation against green: a channel-identical processing path
        # must not increase misregistration.
        f = np.fft.fft2(g - g.mean())
        h = np.fft.fft2(ch - ch.mean())
        cross = f * np.conj(h)
        cross /= np.maximum(np.abs(cross), 1e-12)
        cc = np.fft.ifft2(cross).real
        peak = np.unravel_index(np.argmax(cc), cc.shape)
        dy = peak[0] if peak[0] <= rgb.shape[0] // 2 else peak[0] - rgb.shape[0]
        dx = peak[1] if peak[1] <= rgb.shape[1] // 2 else peak[1] - rgb.shape[1]
        out[name] = {"dy": int(dy), "dx": int(dx)}
    return out


def _border_wrap_step(y: np.ndarray) -> float:
    """p95 of the step across the wrap boundary: col 0 vs col W-1, row 0 vs row H-1."""
    return float(np.percentile(np.concatenate([
        np.abs(y[:, 0] - y[:, -1]), np.abs(y[0, :] - y[-1, :])]), 95))


def seam_check(img: np.ndarray, *, reference: np.ndarray | None = None) -> dict:
    """Did processing introduce a new discontinuity across the wrap boundary?

    A circular FFT seam is a *change* in the step between opposite borders. That
    step is compared with the same step in the unprocessed input. Comparing the
    border step with interior steps would be wrong: removing a screen genuinely
    lowers interior high-frequency steps, which would make a correct result look
    like a seam. A ratio near 1.0 means the border pixels were left as they were,
    i.e. no wrap seam was introduced.
    """
    y = _channel_luma(img)
    h, w = y.shape
    if h < 4 or w < 4:
        return {"note": "image too small"}
    border = _border_wrap_step(y)
    inner = float(np.percentile(np.concatenate([
        np.abs(np.diff(y, axis=1)).ravel(),
        np.abs(np.diff(y, axis=0)).ravel()]), 95))
    out = {
        "border_wrap_step_p95": border,
        "interior_step_p95": inner,
        "border_over_interior": border / max(inner, 1e-9),
    }
    if reference is not None:
        r = _channel_luma(reference)
        if r.shape == y.shape:
            ref_border = _border_wrap_step(r)
            change = border / max(ref_border, 1e-9)
            out["input_border_wrap_step_p95"] = ref_border
            out["border_step_change"] = float(change)
            out["wrap_seam_suspected"] = change > 1.25
            out["note"] = (
                "border_step_change compares the wrap-boundary step with the same "
                "step in the input. ~1.0 means the border was restored and no "
                "circular seam was introduced. 'border_over_interior' is reported "
                "for context only; removing a screen changes interior steps.")
            return out
    out["wrap_seam_suspected"] = False
    out["note"] = "No reference supplied; only descriptive border statistics."
    return out


def spectrum_peak_change(before: np.ndarray, after: np.ndarray,
                         *, band: tuple[float, float] = (1 / 32, 1 / 2.5)) -> dict:
    """Broadband high-frequency energy change, for an at-a-glance comparison."""
    def band_energy(img):
        arr = _channel_luma(img)
        h, w = arr.shape
        spec = np.abs(np.fft.fft2((arr - arr.mean()) * _hann2d((h, w)))) ** 2
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        r = np.hypot(fx, fy)
        m = (r >= band[0]) & (r <= band[1])
        return float(np.sum(spec[m]) / max(spec.sum(), 1e-12))
    b = band_energy(before)
    a = band_energy(after)
    return {"before_fraction": b, "after_fraction": a,
            "memory": "band energy fraction, not a peak-specific measure"}
