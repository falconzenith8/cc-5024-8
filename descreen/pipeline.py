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
    boundary_continuity,
    color_registration,
    content_preservation,
    moire_after_resample,
    peak_suppression,
    seam_check,
)
from .process import (
    area_resize,
    build_notch_specs,
    coordinated_filter,
    frequency_notch,
    resample_plan,
    spatial_fir,
)
from .regions import (
    composite_weights,
    effective_strength,
    group_signature,
    output_regions,
    region_boundaries,
    weight_grid,
    weight_heatmap,
    weight_overlay,
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


def _region_peak(p: dict) -> dict:
    """Full peak record for a region peak, so metrics can use fx/fy."""
    q = _manual_peak({"period_px": p["period_px"], "angle_deg": p["angle_deg"]})
    q["classification"] = str(p.get("classification", "region"))
    q["confidence"] = float(p.get("confidence", 0.0) or 0.0)
    q["reasons"] = [str(p.get("source", "region parameter"))]
    if p.get("basis_b"):
        q["basis_b"] = p["basis_b"]
    if p.get("lattice"):
        q["lattice"] = p["lattice"]
    return q


def _region_group(gid: str, params: dict, peaks: list[dict],
                  shape: tuple[int, int]) -> tuple[dict, list[dict]]:
    specs, notes = build_notch_specs(
        peaks,
        harmonic_orders=tuple(params.get("harmonic_orders", [1, 2])),
        radius_cyc_per_px=params.get("radius_cyc_per_px"),
        apodization=float(params.get("apodization", 0.35)),
        include_lattice_lines=bool(params.get("include_lattice_lines", True)),
        analysis_shape=shape,
    )
    return {
        "id": gid,
        "method": params.get("method", "frequency_notch"),
        "strength": float(params.get("strength", 1.0)),
        "specs": specs,
        "pad": params.get("pad"),
        "feather": params.get("feather"),
        "taps": params.get("taps"),
        "peaks": peaks,
    }, notes


def run_region_pipeline(
    rgb: np.ndarray,
    regions: list[dict],
    *,
    roi: tuple[int, int, int, int] | None = None,
    scale: float = 1.0,
    base_strength: float = 0.0,
    base_params: dict | None = None,
    global_peaks: list[dict] | None = None,
) -> dict:
    """Region-coordinated suppression over the same crop as ``run_pipeline``.

    ``regions`` arrive in absolute source-page coordinates.  They are rebound to
    the ROI, so moving or resizing the ROI maps them correctly, and the weight
    maps are computed at ROI resolution.  Identical parameter groups are
    filtered once over the whole crop and blended by smooth weights; no block
    is processed independently.
    """
    t0 = time.time()
    x0, y0, rw, rh = roi if roi else (0, 0, rgb.shape[1], rgb.shape[0])
    crop = np.asarray(rgb[y0:y0 + rh, x0:x0 + rw], dtype=np.float32)
    shape = crop.shape[:2]

    rebound = []
    mapping_notes = []
    for r in regions:
        q = json.loads(json.dumps(r))
        gx, gy, gw, gh = [float(v) for v in q["rect"]]
        q["rect_page"] = [gx, gy, gw, gh]
        q["rect"] = [gx - x0, gy - y0, gw, gh]
        if (q["rect"][0] + q["rect"][2] <= 0 or q["rect"][1] + q["rect"][3] <= 0
                or q["rect"][0] >= rw or q["rect"][1] >= rh):
            q["clipped_to_roi"] = "outside"
            mapping_notes.append(
                f"{q['id']}: region lies outside the current ROI and contributes "
                f"no pixels")
        else:
            q["clipped_to_roi"] = "partial" if (
                q["rect"][0] < 0 or q["rect"][1] < 0
                or q["rect"][0] + q["rect"][2] > rw
                or q["rect"][1] + q["rect"][3] > rh) else "inside"
        rebound.append(q)

    weights = composite_weights(shape, rebound)
    groups: dict[str, dict] = {}
    group_notes: dict[str, list[dict]] = {}
    group_peaks: dict[str, list[dict]] = {}
    region_group: dict[str, str] = {}
    region_peak_dicts: dict[str, list[dict]] = {}
    region_issues: list[str] = []

    for r in rebound:
        rid = str(r["id"])
        if not r.get("enabled", True):
            region_issues.append(f"{rid}: disabled; its area is left to the base layer")
            continue
        peaks_in = r.get("peaks") or []
        peak_dicts = [_region_peak(p) for p in peaks_in]
        region_peak_dicts[rid] = peak_dicts
        params = r.get("params") or {}
        group, notes = _region_group("", params, peak_dicts, shape)
        gid = f"g_{group_signature(params, group['specs'])}"
        group["id"] = gid
        if gid not in groups:
            groups[gid] = group
            group_notes[gid] = notes
            group_peaks[gid] = peak_dicts
        region_group[rid] = gid
        if not peak_dicts:
            region_issues.append(
                f"{rid}: no peaks were applied; its area is left at the base "
                f"strength ({base_strength:.2f})")

    base_group = None
    if base_strength > 0:
        bp = dict(base_params or {})
        bp["strength"] = float(base_strength)
        base_peaks = [_region_peak(p) for p in (global_peaks or [])]
        if base_peaks:
            group, notes = _region_group("", bp, base_peaks, shape)
            base_group = f"gbase_{group_signature(bp, group['specs'])}"
            group["id"] = base_group
            if base_group not in groups:
                groups[base_group] = group
                group_notes[base_group] = notes
                group_peaks[base_group] = base_peaks
        else:
            region_issues.append(
                "base strength > 0 but no global screen peaks are available; "
                "the base layer stays unfiltered")

    suppressed, proc_audit = coordinated_filter(
        crop, weights, groups, region_group, base_group=base_group)

    eff = effective_strength(
        weights,
        {str(r["id"]): float((r.get("params") or {}).get("strength", 1.0))
         for r in rebound},
        base_strength if base_group is not None else 0.0)

    out, resize_audit = area_resize(suppressed, scale)

    region_metrics = []
    for r in rebound:
        rid = str(r["id"])
        if r.get("clipped_to_roi") == "outside":
            region_metrics.append({"id": rid, "status": "outside_roi"})
            continue
        rx, ry, rw2, rh2 = [int(round(float(v))) for v in r["rect"]]
        cx0, cy0 = max(0, rx), max(0, ry)
        cx1, cy1 = min(shape[1], rx + rw2), min(shape[0], ry + rh2)
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            region_metrics.append({"id": rid, "status": "too_small_to_measure"})
            continue
        before_r = crop[cy0:cy1, cx0:cx1]
        after_r = suppressed[cy0:cy1, cx0:cx1]
        wgt = weights["per_region"].get(rid)
        row = {
            "id": rid,
            "status": "measured",
            "rect_in_crop": [cx0, cy0, cx1 - cx0, cy1 - cy0],
            # Mean over the region's own rectangle: a page-wide mean would
            # report a small zone as "weakly weighted" even when it fully owns
            # its own pixels. The page-wide figure is kept separately.
            "mean_weight": float(wgt[cy0:cy1, cx0:cx1].mean())
            if wgt is not None else 0.0,
            "mean_weight_page": float(wgt.mean()) if wgt is not None else 0.0,
            "min_weight_in_rect": float(wgt[cy0:cy1, cx0:cx1].min())
            if wgt is not None else 0.0,
            "strength": float((r.get("params") or {}).get("strength", 1.0)),
            "method": (r.get("params") or {}).get("method", "frequency_notch"),
            "peaks_applied": [{"period_px": p["period_px"],
                               "angle_deg": p["angle_deg"]}
                              for p in region_peak_dicts.get(rid, [])],
            "content_preservation": content_preservation(before_r, after_r),
        }
        pk = region_peak_dicts.get(rid) or []
        if pk:
            row["peak_suppression"] = peak_suppression(before_r, after_r, pk)
            row["peak_suppression"].pop("peaks", None)
        region_metrics.append(row)

    boundaries = region_boundaries(rebound, shape)
    continuity = boundary_continuity(crop, suppressed, boundaries)

    wgrid = weight_grid(weights, eff)
    overlay, legend = weight_overlay(weights)
    heat = weight_heatmap(eff)

    warnings = list(region_issues)
    uncovered = float(weights["base_fraction"])
    if base_group is None and uncovered > 0.02:
        warnings.append(
            f"{uncovered * 100:.0f}% of the crop is outside every zone and the "
            f"base strength is 0, so that area is not filtered at all. Extend "
            f"the zones or raise the base strength if it also carries a screen.")
    if continuity.get("seam_suspected"):
        warnings.append(
            "A region boundary step exceeds the continuity threshold; widen the "
            "feather or reduce the strength difference between the two sides.")
    for rm in region_metrics:
        if rm.get("status") != "measured":
            continue
        cp = rm["content_preservation"]["worst_retained_fraction"]
        if cp < 0.75:
            warnings.append(
                f"{rm['id']}: detail retention {cp:.2f} is low; reduce that "
                f"region's strength or notch radius.")
        ps = rm.get("peak_suppression") or {}
        if ps.get("peaks") is not None and not ps.get("all_reduced", True):
            warnings.append(
                f"{rm['id']}: not every applied screen peak dropped > 3 dB.")
        if rm["mean_weight"] < 0.5:
            warnings.append(
                f"{rm['id']}: over its own rectangle its weight averages only "
                f"{rm['mean_weight']:.2f} (heavy overlap or a very narrow "
                f"rectangle); the effective parameters there are a blend, not "
                f"this region's alone.")

    result = {
        "mode": "region",
        "roi": [int(x0), int(y0), int(rw), int(rh)],
        "absolute_roi": [int(x0), int(y0), int(rw), int(rh)],
        "scale": float(scale),
        "base_strength": float(base_strength),
        "regions": rebound,
        "output_regions": output_regions(rebound, scale, offset=(x0, y0),
                                         source_shape=shape),
        "region_mapping": {
            "binding": "region geometry is stored in absolute source-page pixels",
            "roi_rebind": "rect_in_crop = rect_page - roi_origin",
            "output_mapping": "rect_out = (rect_page) * scale",
            "notes": mapping_notes + [
                "changing the ROI re-binds regions; their page coordinates "
                "do not silently drift"],
            "region_count": len(rebound),
        },
        "weights": {
            "grid": wgrid,
            "legend": legend,
            "sum_range": [weights["sum_min"], weights["sum_max"]],
            "base_fraction": weights["base_fraction"],
            "effective_strength_range": [float(eff.min()), float(eff.max())],
            "rule": "w_raw = smooth rect * opacity; k = 1/max(1, sum raw); "
                    "region = raw*k; base = 1 - sum(raw)*k",
        },
        "groups": [
            {"id": gid,
             "method": g["method"],
             "strength": g["strength"],
             "notch_count": len(g["specs"]),
             "notch_specs": [s.__dict__ | {
                 "center_fx": float(s.center()[0]),
                 "center_fy": float(s.center()[1]),
                 "resolved_radius_cyc_per_px": float(s.resolved_radius())}
                 for s in g["specs"]],
             "notes": group_notes.get(gid, [])}
            for gid, g in groups.items()
        ],
        "processing_audit": proc_audit,
        "resample_audit": resize_audit,
        "resample_plan": resample_plan(scale),
        "region_metrics": region_metrics,
        "boundary_continuity": continuity,
        "warnings": warnings,
        "elapsed_s": float(time.time() - t0),
        "inputs": {
            "input_shape": list(shape),
            "suppressed_shape": list(suppressed.shape[:2]),
            "output_shape": list(out.shape[:2]),
        },
    }
    result["_images"] = {
        "input": crop,
        "suppressed": suppressed,
        "output": out,
        "weight_heatmap": heat,
        "weight_overlay": overlay,
        "difference": _difference_image(crop, suppressed),
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
    out = {
        "input.png": to_png_bytes(imgs["input"]),
        "suppressed.png": to_png_bytes(imgs["suppressed"]),
        "output.png": to_png_bytes(imgs["output"]),
        "difference.png": to_png_bytes(imgs.get("difference") if imgs.get("difference")
                                       is not None else _difference_image(
                                           imgs["input"], imgs["suppressed"])),
    }
    if imgs.get("spectrum_png") is not None:
        out["spectrum.png"] = imgs["spectrum_png"]
    if imgs.get("overview_png") is not None:
        out["overview.png"] = imgs["overview_png"]
    if imgs.get("weight_heatmap") is not None:
        out["weights.png"] = to_png_bytes(imgs["weight_heatmap"])
    return out


def result_overlay_png(result: dict) -> bytes | None:
    img = result.get("_images", {}).get("weight_overlay")
    if img is None:
        return None
    return to_png_bytes(img)


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
    if isinstance(payload.get("weights"), dict):
        # The per-pixel weight grid is preview data. The exported parameters
        # keep the region geometry, feather and normalization rule, which are
        # exactly what is needed to reproduce the same transition.
        payload["weights"] = {k: v for k, v in payload["weights"].items()
                              if k != "grid"}
        payload["weights"]["grid_note"] = (
            "per-pixel weight grid omitted from the export; recompute it from "
            "regions, feather_px and the weight rule")

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
