"""End-to-end descreen pipeline: analyze -> notch -> resample -> measure.

Every number reported here is computed from the actual input and the actual
output of this run. No pre-repaired "after" image is involved anywhere.
"""

from __future__ import annotations

import json
import time

import numpy as np

from .analysis import analyze_image, channel_array, spectrum_png, crop_overview_png
from .metrics import (
    color_registration,
    content_preservation,
    moire_after_resample,
    peak_suppression,
    seam_check,
)
from .process import (
    area_resize,
    build_notch_specs,
    frequency_notch,
    resample_plan,
    spatial_fir,
)
from .synth import to_png_bytes


def _peak_dicts(peaks: list[dict], *, selected_only: bool) -> list[dict]:
    out = []
    for p in peaks:
        if selected_only and not p.get("selected"):
            continue
        out.append(p)
    return out


def run_pipeline(
    rgb: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
    scale: float = 1.0,
    method: str = "frequency_notch",
    deep_channels: bool = False,
    flat_tile_assist: bool = True,
    manual_peaks: list[dict] | None = None,
    reject_peaks: list[dict] | None = None,
    harmonic_orders: tuple[int, ...] = (1, 2),
    include_lattice_lines: bool = True,
    radius_cyc_per_px: float | None = None,
    apodization: float = 0.35,
    strength: float = 1.0,
    pad: int | None = None,
    feather: int | None = None,
    taps: int | None = None,
    force_peaks: bool = False,
    register_scale: float | None = None,
) -> dict:
    """Analyze a region, suppress its screens, resample, and measure.

    ``manual_peaks`` adds operator-specified screens (e.g. a known 133 lpi
    screen that the automatic pass left ambiguous). ``reject_peaks`` removes
    automatically selected peaks, for the case where a real texture was wrongly
    selected. Both are matched by period/angle, not by index, so they survive
    re-analysis.
    ``force_peaks`` processes even when nothing scored as a high-confidence
    screen; the caller is responsible for having made that decision explicitly.
    """
    t0 = time.time()
    x0, y0, rw, rh = roi if roi else (0, 0, rgb.shape[1], rgb.shape[0])
    crop = rgb[y0:y0 + rh, x0:x0 + rw]

    analysis = analyze_image(crop, None, deep_channels=deep_channels,
                             flat_tile_assist=flat_tile_assist)
    peaks = analysis["peaks"]

    rejected_keys: list[tuple[float, float]] = []
    if reject_peaks:
        for rp in reject_peaks:
            rejected_keys.append((float(rp["period_px"]), float(rp["angle_deg"])))
            for p in peaks:
                if _same_screen(p, rp):
                    p["selected"] = False
                    p["classification"] = "manually_rejected"
                    p["reasons"] = list(p.get("reasons", [])) + [
                        "deselected by the operator as real texture"]
    if manual_peaks:
        for mp in manual_peaks:
            peaks.append(_manual_peak(mp))

    accepted = _peak_dicts(peaks, selected_only=True)
    if not accepted and force_peaks:
        # Explicit operator override: use the strongest periodic candidates,
        # clearly relabelled so the report never claims automatic confidence.
        cands = sorted(peaks, key=lambda p: p.get("magnitude", 0.0), reverse=True)
        accepted = []
        for p in cands[:1]:
            q = dict(p)
            q["selected"] = True
            q["classification"] = "manual_override"
            q["confidence"] = 0.0
            q["reasons"] = list(q.get("reasons", [])) + [
                "applied by explicit operator override"]
            accepted.append(q)

    specs, notch_notes = build_notch_specs(
        accepted,
        harmonic_orders=harmonic_orders,
        radius_cyc_per_px=radius_cyc_per_px,
        apodization=apodization,
        include_lattice_lines=include_lattice_lines,
        analysis_shape=crop.shape[:2],
    )

    if specs:
        if method == "spatial_fir":
            suppressed, proc_audit = spatial_fir(
                crop, specs, taps=taps, strength=strength)
        elif method == "frequency_notch":
            suppressed, proc_audit = frequency_notch(
                crop, specs, pad=pad, feather=feather, strength=strength)
        else:
            raise ValueError(f"unknown method {method!r}")
    else:
        suppressed = crop.astype(np.float32).copy()
        proc_audit = {"method": method, "notches": [],
                      "note": "no peaks accepted; image passed through unchanged"}

    out, resize_audit = area_resize(suppressed, scale)
    plan = resample_plan(scale)

    rs = scale if register_scale is None else register_scale
    metrics = {
        "peak_suppression": peak_suppression(crop, suppressed, accepted) if accepted else {
            "peaks": [], "note": "no peaks were accepted"},
        "content_preservation": content_preservation(crop, suppressed),
        "moire_vs_area_reference": (
            moire_after_resample(crop, out, scale) if scale < 1.0 else {
                "note": "no resampling requested; moire reference not applicable"}),
        "seam": seam_check(suppressed, reference=crop),
        "color_registration": {
            "input": color_registration(crop),
            "output": color_registration(out),
            "suppressed_output": color_registration(suppressed),
        },
        "resample_scale": float(scale),
        "note": "Edge/hairline/detail metrics are measured on the suppressed "
                "image before resampling, so the resampling low-pass is not "
                "misreported as descreen smoothing.",
    }

    analysis_peaks_png = []
    accepted_ids = {id(p) for p in accepted}

    overview = crop_overview_png(crop, (0, 0, crop.shape[1], crop.shape[0]))
    luma = channel_array(crop, analysis.get("selected_channel", "luma"))
    spectrum = spectrum_png(luma, analysis["peaks"])

    result = {
        "roi": [int(v) for v in analysis["roi"]],
        "absolute_roi": [int(x0), int(y0), int(rw), int(rh)],
        "analysis_scale": analysis["analysis_scale"],
        "scale": float(scale),
        "method": method,
        "harmonics": list(harmonic_orders),
        "include_lattice_lines": bool(include_lattice_lines),
        "radius_cyc_per_px": radius_cyc_per_px,
        "apodization": float(apodization),
        "strength": float(strength),
        "taps": taps,
        "pad": pad,
        "feather": feather,
        "peaks": analysis["peaks"],
        "accepted_peaks": accepted,
        "rejected_peaks": [{"period_px": q[0], "angle_deg": q[1]}
                           for q in rejected_keys],
        "manual_peaks": manual_peaks or [],
        "lattice_explained": analysis["lattice_explained"],
        "notch_specs": [s.__dict__ | {"center_fx": float(s.center()[0]),
                                      "center_fy": float(s.center()[1]),
                                      "resolved_radius_cyc_per_px": float(s.resolved_radius())}
                        for s in specs],
        "notch_notes": notch_notes,
        "processing_audit": proc_audit,
        "resample_audit": resize_audit,
        "resample_plan": plan,
        "metrics": metrics,
        "warnings": analysis["warnings"] + _pipeline_warnings(accepted, specs, metrics),
        "flat_tile_assist_used": analysis["flat_tile_assist_used"],
        "flat_tiles": analysis["flat_tiles"],
        "manual_review_required": analysis["manual_review_required"],
        "elapsed_s": float(time.time() - t0),
        "inputs": {
            "input_shape": list(crop.shape[:2]),
            "suppressed_shape": list(suppressed.shape[:2]),
            "output_shape": list(out.shape[:2]),
        },
    }
    result["_images"] = {
        "input": crop,
        "suppressed": suppressed,
        "output": out,
        "spectrum_png": spectrum,
        "overview_png": overview,
    }
    return result


def _same_screen(a: dict, b: dict, *, period_tol: float = 0.05,
                 angle_tol: float = 5.0) -> bool:
    from .analysis import angle_diff
    pa, pb = float(a["period_px"]), float(b["period_px"])
    if abs(pa - pb) / max(pb, 1e-9) > period_tol:
        return False
    return abs(angle_diff(float(a["angle_deg"]), float(b["angle_deg"]))) <= angle_tol


def _manual_peak(mp: dict) -> dict:
    period = float(mp["period_px"])
    angle = float(mp["angle_deg"])
    th = np.deg2rad(angle)
    return {
        "fx": float(np.cos(th) / period),
        "fy": float(np.sin(th) / period),
        "period_px": period,
        "angle_deg": angle,
        "magnitude": 0.0,
        "snr": 0.0,
        "prominence": 0.0,
        "isolation": 0.0,
        "coverage": 1.0,
        "coherence": 0.0,
        "comb_present": False,
        "comb_expected": False,
        "collinear2_rel": 0.0,
        "collinear2_offset": 0.0,
        "collinear3_rel": 0.0,
        "collinear3_offset": 0.0,
        "squareness": 0.0,
        "perpendicularity": 0.0,
        "channel": "manual",
        "channels": ["manual"],
        "basis_b": None,
        "lattice": [],
        "isotropy": 1.0,
        "lattice_orthogonality": 1.0,
        "classification": "manual",
        "confidence": 0.0,
        "selected": True,
        "reasons": ["entered by the operator"],
    }


def _pipeline_warnings(accepted: list[dict], specs, metrics: dict) -> list[str]:
    out: list[str] = []
    if accepted and not specs:
        out.append("Peaks were accepted but no notch could be built; check the "
                   "period band (2.5-32 px) and harmonic orders.")
    if any(p.get("classification") in ("manual", "manual_override") for p in accepted):
        out.append("At least one screen was applied manually; automatic detection "
                   "did not confirm it.")
    ps = metrics.get("peak_suppression", {})
    if ps.get("peaks") and not ps.get("all_reduced", True):
        out.append("Not every reported peak dropped by more than 3 dB. Inspect the "
                   "spectrum and the suppression image before exporting.")
    cp = metrics.get("content_preservation", {}).get("worst_retained_fraction")
    if cp is not None and cp < 0.75:
        out.append(f"Detail retention fell to {cp:.2f}; the notch may be too wide "
                   "for this image. Reduce the radius or the harmonic set.")
    if metrics.get("moire_vs_area_reference", {}).get("added_moire_risk"):
        out.append("Output carries more low-frequency beat energy than the plain "
                   "area-averaged reference; inspect for new moire.")
    if metrics.get("seam", {}).get("wrap_seam_suspected"):
        out.append("Border/interior step ratio is high; increase padding or the "
                   "edge feather.")
    reg = metrics.get("color_registration", {})
    for side in ("output", "suppressed_output"):
        r = reg.get(side, {})
        if isinstance(r, dict) and any(
                isinstance(v, dict) and (v.get("dx") or v.get("dy"))
                for v in r.values()):
            out.append(f"Color channels moved relative to green in the {side}; "
                       "registration must stay at (0,0).")
            break
    return out


def result_pngs(result: dict) -> dict[str, bytes]:
    imgs = result["_images"]
    return {
        "input.png": to_png_bytes(imgs["input"]),
        "suppressed.png": to_png_bytes(imgs["suppressed"]),
        "output.png": to_png_bytes(imgs["output"]),
        "spectrum.png": imgs["spectrum_png"],
        "overview.png": imgs["overview_png"],
        "difference.png": to_png_bytes(_difference_image(
            imgs["input"], imgs["suppressed"])),
    }


def _difference_image(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    b = np.asarray(before, dtype=np.float32)
    a = np.asarray(after, dtype=np.float32)
    d = np.abs(a - b)
    if d.ndim == 3:
        d = d.max(axis=2)
    scale = np.percentile(d, 99.5)
    if scale <= 1e-6:
        return np.zeros(d.shape + (3,), dtype=np.float32)
    v = np.clip(d / scale, 0.0, 1.0)
    return np.stack([v * 0.15, v * 0.55, v], axis=-1)


def result_json(result: dict) -> str:
    payload = {k: v for k, v in result.items() if not k.startswith("_")}

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    return json.dumps(clean(payload), indent=2, ensure_ascii=False)
