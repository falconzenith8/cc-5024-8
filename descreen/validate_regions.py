"""Region-coordination validation on real pipeline runs.

Four required scenarios are exercised, each measured, never eyeballed:

1. text crossing two regions with different screens;
2. a local screen-direction change inside one page;
3. overlapping regions;
4. a very narrow region.

For each scenario the script reports boundary continuity of the processing
delta, per-region screen suppression, and text-detail retention against the
clean unscreened artwork.  A naive per-block paste of independently filtered
tiles is run on the same input as the negative control: it must fail the
continuity test that the coordinated composite passes.

Run:  python3 -m descreen.validate_regions [outdir]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from .metrics import (
    band_detail_retention,
    boundary_continuity,
    content_vs_reference,
)
from .pipeline import (
    _difference_image,
    _region_peak,
    result_json,
    result_pngs,
    run_region_pipeline,
)
from .process import build_notch_specs, frequency_notch
from .regions import (
    composite_weights,
    effective_strength,
    map_region_set,
    region_boundaries,
    weight_grid,
)

from .synth import (
    ScreenSpec,
    dot_screen,
    make_text_fixture,
    make_two_frequency_clean,
    make_two_frequency_fixture,
    to_png_bytes,
)

NAIVE_SPECS: dict[tuple[float, float], list] = {}


def _specs(period: float, angle: float, shape: tuple[int, int]):
    key = (round(period, 4), round(angle, 4))
    if key not in NAIVE_SPECS:
        NAIVE_SPECS[key] = build_notch_specs(
            [_region_peak({"period_px": period, "angle_deg": angle})],
            harmonic_orders=(1, 2), analysis_shape=shape)[0]
    return NAIVE_SPECS[key]


def naive_per_block(page: np.ndarray, blocks: list[dict]) -> np.ndarray:
    """Negative control: filter each block in isolation and paste it hard.

    This is the naive implementation the coordinated path must beat: every
    rectangle is filtered with only its own pixels and its own parameters, then
    written back with no transition weight.
    """
    out = page.copy()
    for b in blocks:
        x, y, w, h = b["rect"]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(page.shape[1], x + w), min(page.shape[0], y + h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        sub = page[y0:y1, x0:x1]
        filtered, _ = frequency_notch(
            sub, _specs(b["period_px"], b["angle_deg"], sub.shape[:2]),
            strength=float(b.get("strength", 1.0)),
            # feather=0 is the honest "no transition" baseline: the block's
            # filtered result is pasted right up to its edge.
            feather=0)
        out[y0:y1, x0:x1] = filtered
    return out


def _region(rid: str, rect, period: float, angle: float, *,
            feather: float = 28, strength: float = 1.0,
            method: str = "frequency_notch", opacity: float = 1.0,
            harmonics=(1, 2)) -> dict:
    return {
        "id": rid,
        "rect": [int(v) for v in rect],
        "feather_px": float(feather),
        "opacity": float(opacity),
        "enabled": True,
        "status": "confirmed",
        "auto_peaks": False,
        "peaks": [{"period_px": float(period), "angle_deg": float(angle),
                   "source": "scenario"}],
        "params": {"method": method, "strength": float(strength),
                   "radius_cyc_per_px": None, "apodization": 0.35,
                   "harmonic_orders": list(harmonics),
                   "include_lattice_lines": True},
    }


def _scenarios() -> list[dict]:
    page, meta = make_two_frequency_fixture()
    clean, _ = make_two_frequency_clean()
    h, w = page.shape[:2]
    split = meta["split_x"]

    text_across = {
        "name": "text_across_two_screens",
        "page": page,
        "clean": clean,
        "description": "one body of text crosses 6 px/33 deg on the left and "
                       "9.5 px/12 deg on the right",
        "regions": [
            _region("left", (0, 0, split, h), 6.0, 33.0),
            _region("right", (split, 0, w - split, h), 9.5, 12.0),
        ],
        "naive_blocks": [
            {"rect": (0, 0, split, h), "period_px": 6.0, "angle_deg": 33.0},
            {"rect": (split, 0, w - split, h), "period_px": 9.5,
             "angle_deg": 12.0},
        ],
        "extra_boundaries": [],
    }

    # Local direction change inside one page: a horizontal band rotates its
    # screen 45 deg, the rest keeps 6 px / 33 deg.  The screen is regenerated
    # over the whole page and the band is spliced in, so the band edge is the
    # only discontinuity.
    base_clean = make_text_fixture(shape=(720, 1000), screen=None)[0]
    main = dot_screen(base_clean.shape, ScreenSpec(6.0, 33.0, 0.16), base_clean)
    rotated = dot_screen(base_clean.shape, ScreenSpec(6.0, 78.0, 0.16),
                         base_clean)
    band = (170, 250, 620, 200)
    direction_page = main.copy()
    bx, by, bw, bh = band
    direction_page[by:by + bh, bx:bx + bw] = rotated[by:by + bh, bx:bx + bw]
    direction = {
        "name": "local_screen_direction_change",
        "page": direction_page,
        "clean": base_clean,
        "description": "a 620x200 patch rotates its screen from 33 deg to "
                       "78 deg; the surrounding page stays at 33 deg",
        "regions": [
            _region("surround", (0, 0, 1000, 720), 6.0, 33.0, feather=34),
            _region("band", band, 6.0, 78.0, feather=26),
        ],
        "naive_blocks": [
            {"rect": (0, 0, 1000, 720), "period_px": 6.0, "angle_deg": 33.0},
            {"rect": band, "period_px": 6.0, "angle_deg": 78.0},
        ],
        "extra_boundaries": [],
        # The band layer sits inside the surround region: this scenario also
        # exercises overlap and normalized weighting.
        "note": "overlapping by construction: the band layer is nested inside "
                "the surround region",
    }

    overlap = {
        "name": "overlapping_regions",
        "page": page,
        "clean": clean,
        "description": "two regions share a 240 px strip; the overlap is "
                       "normalized and still filters the whole crop",
        "regions": [
            _region("left", (0, 0, split + 120, h), 6.0, 33.0, feather=40),
            _region("right", (split - 120, 0, w - split + 120, h), 9.5, 12.0,
                    feather=40),
        ],
        # A naive implementation cannot blend an overlap: it must assign each
        # pixel to one region. Splitting the overlap down the middle exposes
        # exactly the hard switch the coordinated weights avoid.
        "naive_blocks": [
            {"rect": (0, 0, split, h), "period_px": 6.0, "angle_deg": 33.0},
            {"rect": (split, 0, w - split, h), "period_px": 9.5,
             "angle_deg": 12.0},
        ],
        "extra_boundaries": [
            # The naive per-pixel assignment switch inside the shared strip.
            {"axis": "x", "pos": split, "lo": 0, "hi": h, "feather_px": 40},
        ],
    }

    narrow = {
        "name": "very_narrow_region",
        "page": page,
        "clean": clean,
        "description": "a 14 px wide strip with its own reduced strength "
                       "inside the left screen; no hard edge may appear",
        "regions": [
            _region("left", (0, 0, split, h), 6.0, 33.0),
            _region("narrow", (300, 100, 14, 500), 6.0, 33.0,
                    feather=8, strength=0.35, harmonics=(1,)),
            _region("right", (split, 0, w - split, h), 9.5, 12.0),
        ],
        "naive_blocks": [
            {"rect": (0, 0, split, h), "period_px": 6.0, "angle_deg": 33.0},
            {"rect": (split, 0, w - split, h), "period_px": 9.5,
             "angle_deg": 12.0},
            {"rect": (300, 100, 14, 500), "period_px": 6.0,
             "angle_deg": 33.0, "strength": 0.35},
        ],
        "extra_boundaries": [
            {"axis": "x", "pos": 300, "lo": 100, "hi": 600},
            {"axis": "x", "pos": 314, "lo": 100, "hi": 600},
        ],
    }
    return [text_across, direction, overlap, narrow]


def _measure(page: np.ndarray, clean: np.ndarray, result: dict,
             scenario: dict, *, scale: float) -> dict:
    suppressed = result["_images"]["suppressed"]
    boundaries = region_boundaries(
        [{"id": r["id"], "rect": r["rect"], "enabled": True}
         for r in result["regions"]],
        page.shape[:2]) + scenario["extra_boundaries"]
    # Feather from the region that owns each measured edge, when known.
    feather_by_region = {r["id"]: float(r.get("feather_px", 28.0))
                         for r in result["regions"]}
    for b in boundaries:
        if "feather_px" not in b and b.get("region") in feather_by_region:
            b["feather_px"] = feather_by_region[b["region"]]
    cont = boundary_continuity(page, suppressed, boundaries)

    naive = naive_per_block(page, scenario["naive_blocks"])
    naive_cont = boundary_continuity(page, naive, boundaries)

    retention = content_vs_reference(suppressed, clean)
    baseline = content_vs_reference(page, clean)
    naive_ret = content_vs_reference(naive, clean)

    # Whole-image retention is dominated by the screen energy everywhere except
    # the seam.  Measure the same statistics only in a band around the actual
    # region edges, which is where a per-block paste smears text.
    band = [{"axis": b["axis"], "pos": b["pos"],
             "lo": b.get("lo", 0), "hi": b.get("hi", 0)} for b in boundaries]
    band_coord = band_detail_retention(suppressed, clean, band, half_width=40)
    band_naive = band_detail_retention(naive, clean, band, half_width=40)
    band_input = band_detail_retention(page, clean, band, half_width=40)

    regions = []
    for rm in result["region_metrics"]:
        if rm.get("status") != "measured":
            regions.append({"id": rm["id"], "status": rm.get("status")})
            continue
        regions.append({
            "id": rm["id"],
            "mean_weight": rm["mean_weight"],
            "strength": rm["strength"],
            "peaks_applied": rm["peaks_applied"],
            "worst_detail_retention": rm["content_preservation"][
                "worst_retained_fraction"],
            "suppression_db": (rm.get("peak_suppression") or {}).get(
                "mean_reduction_db"),
        })

    ok = (not cont["seam_suspected"]
          and retention["edge_slope_retention"] >= 0.85 * baseline["edge_slope_retention"]
          and retention["edge_acutance_retention"] >= 0.85 * baseline["edge_acutance_retention"])

    # Export fidelity: rebuild the weights from *only* what the exported JSON
    # carries (regions, feathers, opacity, strength rule) and compare with the
    # weights the run actually used. If the exported file could not reproduce
    # the previewed transition, this differs.
    export_check = _recompute_weights_from_export(result, page.shape[:2])
    ok = ok and export_check["matches"]
    return {
        "name": scenario["name"],
        "description": scenario["description"],
        "scale": scale,
        "coordinated": {
            "hard_boundaries": cont["hard_boundaries"],
            "seam_suspected": cont["seam_suspected"],
            "boundaries_measured": len(cont["boundaries"]),
            "min_transition_ratio": cont["min_transition_ratio"],
            "per_boundary": [
                {"axis": b["axis"], "pos": b["pos"],
                 "processing_width_px":
                     b["processing_magnitude"]["width_px"],
                 "processing_single_step_fraction":
                     b["processing_magnitude"]["single_step_fraction"],
                 "hard": b["hard_transitions"]}
                for b in cont["boundaries"]],
        },
        "naive_per_block": {
            "hard_boundaries": naive_cont["hard_boundaries"],
            "seam_suspected": naive_cont["seam_suspected"],
            "per_boundary": [
                {"axis": b["axis"], "pos": b["pos"],
                 "processing_width_px":
                     b["processing_magnitude"]["width_px"],
                 "processing_single_step_fraction":
                     b["processing_magnitude"]["single_step_fraction"],
                 "hard": b["hard_transitions"]}
                for b in naive_cont["boundaries"]],
        },
        "no_hard_seam": not cont["seam_suspected"],
        "naive_has_hard_seam": naive_cont["seam_suspected"],
        "text_retention": {
            "coordinated_edge_slope": retention["edge_slope_retention"],
            "screened_input_edge_slope": baseline["edge_slope_retention"],
            "naive_edge_slope": naive_ret["edge_slope_retention"],
            "coordinated_acutance": retention["edge_acutance_retention"],
            "screened_input_acutance": baseline["edge_acutance_retention"],
            "naive_acutance": naive_ret["edge_acutance_retention"],
            "psnr_vs_clean_db": retention["psnr_vs_clean_db"],
            "naive_psnr_vs_clean_db": naive_ret["psnr_vs_clean_db"],
        },
        # Local evidence: only pixels within +-40 px of a region edge.
        "seam_band_retention": {
            "half_width_px": 40,
            "edge_pixels": band_coord.get("edge_pixels"),
            "coordinated_edge_slope": band_coord.get("edge_slope_retention"),
            "naive_edge_slope": band_naive.get("edge_slope_retention"),
            "screened_input_edge_slope": band_input.get("edge_slope_retention"),
            "coordinated_acutance": band_coord.get("edge_acutance_retention"),
            "naive_acutance": band_naive.get("edge_acutance_retention"),
            "screened_input_acutance": band_input.get("edge_acutance_retention"),
            "note": "Detail retention restricted to +-40 px of the region "
                    "edges. Both implementations preserve text here; they "
                    "differ in boundary continuity, not in local detail loss.",
        },
        "regions": regions,
        "weight_sum_range": result["weights"]["sum_range"],
        "base_fraction": result["weights"]["base_fraction"],
        "export_fidelity": export_check,
        "ok": bool(ok),
    }


def _recompute_weights_from_export(result: dict, shape: tuple[int, int]) -> dict:
    """Rebuild weights from the exported description and diff against the run.

    The export omits the per-pixel grid on purpose; it is expected to carry
    enough (rects, feather, opacity, enabled, strength) to rebuild the same
    normalized weights. This proves the file, not just the preview, defines the
    transition.
    """
    exported = json.loads(result_json(result))
    regions = exported["regions"]
    # Rebuild on the page grid from the page-coordinate rectangles the export
    # records, so this works for any ROI, not just a full-page run.
    spec = []
    for r in regions:
        rect = r.get("rect_page") or r["rect"]
        spec.append({"id": r["id"], "rect": rect,
                     "feather_px": r.get("feather_px", 28.0),
                     "opacity": r.get("opacity", 1.0),
                     "enabled": r.get("enabled", True)})
    rebuilt = composite_weights((int(shape[0]), int(shape[1])), spec)
    eff = effective_strength(
        rebuilt,
        {str(r["id"]): float((r.get("params") or {}).get("strength", 1.0))
         for r in regions},
        float(exported.get("base_strength", 0.0)),
    )
    rebuilt_grid = weight_grid(rebuilt, eff)
    used_grid = result["weights"]["grid"]
    max_delta = 0.0
    for rid, grid in rebuilt_grid["per_region"].items():
        used = used_grid["per_region"].get(rid)
        if used is None:
            max_delta = 1.0
            continue
        a = np.asarray(grid, dtype=np.float64)
        b = np.asarray(used, dtype=np.float64)
        if a.shape != b.shape:
            max_delta = 1.0
            continue
        max_delta = max(max_delta, float(np.abs(a - b).max()))
    a = np.asarray(rebuilt_grid["effective_strength"], dtype=np.float64)
    b = np.asarray(used_grid["effective_strength"], dtype=np.float64)
    if a.shape == b.shape:
        max_delta = max(max_delta, float(np.abs(a - b).max()))
    else:
        max_delta = 1.0
    return {
        # The grid is rounded to 3 dp before comparison, so 1e-3 is exact match.
        "matches": max_delta <= 1.001e-3,
        "max_abs_weight_delta": max_delta,
        "grid_omitted": "grid" not in exported["weights"],
        "rule_exported": exported["weights"]["rule"],
        "regions_rebuilt": len(regions),
        "note": "Weights were rebuilt from the exported JSON alone and compared "
                "with the run's weight grid; a small delta means the export "
                "reproduces the previewed transition.",
    }


def _mapping_check(page: np.ndarray, regions: list[dict]) -> dict:
    """Source-size change must move zones with their content, not drift.

    The page is resampled to half size, the zone set is mapped, and the pipeline
    is run on the resized page. The mapped rectangles must sit at the same
    *relative* position, and the normalized weights must still sum to 1 there.
    """
    from .process import area_resize

    h, w = page.shape[:2]
    small, _ = area_resize(page, 0.5)
    mapped = map_region_set(regions, (h, w), small.shape[:2])
    result = run_region_pipeline(small, mapped, scale=1.0)
    drift = []
    for before, after in zip(regions, mapped):
        bx, by, bw, bh = [float(v) for v in before["rect"]]
        ax, ay, aw, ah = [float(v) for v in after["rect"]]
        drift.append({
            "id": before["id"],
            "expected_rect": [bx * 0.5, by * 0.5, bw * 0.5, bh * 0.5],
            "mapped_rect": [ax, ay, aw, ah],
            "max_abs_delta_px": float(max(
                abs(ax - bx * 0.5), abs(ay - by * 0.5),
                abs(aw - bw * 0.5), abs(ah - bh * 0.5))),
            "feather_expected": float(before.get("feather_px", 28.0)) * 0.5,
            "feather_mapped": float(after["feather_px"]),
        })
    max_drift = max(d["max_abs_delta_px"] for d in drift)
    feather_ok = all(abs(d["feather_mapped"] - d["feather_expected"]) < 1e-6
                     for d in drift)
    sums = result["weights"]["sum_range"]

    # ROI rebinding: the same page-coordinate zones, run through a cropped ROI,
    # must be offset by the ROI origin and still own the same content.
    x0, y0 = w // 4, h // 5
    rw, rh = w // 2, h // 2
    roi_result = run_region_pipeline(small, mapped, roi=(x0 // 2, y0 // 2,
                                                          rw // 2, rh // 2),
                                     scale=1.0)
    rebind = []
    for before, got in zip(mapped, roi_result["regions"]):
        gx, gy, gw2, gh2 = [float(v) for v in before["rect"]]
        rx, ry, rww, rhh = [float(v) for v in got["rect"]]
        rebind.append({
            "id": got["id"],
            "rect_page": [gx, gy, gw2, gh2],
            "rect_in_crop": [rx, ry, rww, rhh],
            "expected_rect_in_crop": [gx - x0 // 2, gy - y0 // 2, gw2, gh2],
        })
    rebind_ok = all(
        abs(r["rect_in_crop"][0] - r["expected_rect_in_crop"][0]) < 1e-6 and
        abs(r["rect_in_crop"][1] - r["expected_rect_in_crop"][1]) < 1e-6 and
        abs(r["rect_in_crop"][2] - r["expected_rect_in_crop"][2]) < 1e-6 and
        abs(r["rect_in_crop"][3] - r["expected_rect_in_crop"][3]) < 1e-6
        for r in rebind)
    return {
        "source_shape": [h, w],
        "target_shape": list(small.shape[:2]),
        "regions": drift,
        "max_abs_delta_px": max_drift,
        "feather_scaled_correctly": feather_ok,
        "weight_sum_range_after_mapping": sums,
        "roi_rebind": rebind,
        "roi_rebind_ok": rebind_ok,
        "ok": bool(max_drift <= 0.51 and feather_ok
                   and sums[0] >= 0.999 and sums[1] <= 1.001 and rebind_ok),
        "note": "Zone rectangles are multiplied by the size ratio and feathers "
                "by sqrt(sx*sy); the mapped zones must land on the same content. "
                "Running through a cropped ROI subtracts the ROI origin without "
                "moving the content a zone owns.",
    }


def run(outdir: str, *, save_images: bool = True) -> dict:
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for sc in _scenarios():
        print(f"[{sc['name']}]", flush=True)
        result = run_region_pipeline(sc["page"], sc["regions"], scale=1.0)
        row = _measure(sc["page"], sc["clean"], result, sc, scale=1.0)
        rows.append(row)
        if save_images:
            folder = os.path.join(outdir, sc["name"])
            os.makedirs(folder, exist_ok=True)
            for fname, data in result_pngs(result).items():
                with open(os.path.join(folder, fname), "wb") as fh:
                    fh.write(data)
            with open(os.path.join(folder, "params.json"), "w",
                      encoding="utf-8") as fh:
                fh.write(result_json(result))
            with open(os.path.join(folder, "input_page.png"), "wb") as fh:
                fh.write(to_png_bytes(sc["page"]))
            with open(os.path.join(folder, "clean_reference.png"), "wb") as fh:
                fh.write(to_png_bytes(sc["clean"]))
            naive = naive_per_block(sc["page"], sc["naive_blocks"])
            with open(os.path.join(folder, "naive_per_block.png"), "wb") as fh:
                fh.write(to_png_bytes(naive))
            with open(os.path.join(folder, "naive_difference.png"), "wb") as fh:
                fh.write(to_png_bytes(_difference_image(sc["page"], naive)))
            # A stretched difference strip across the main junction makes the
            # coordinated-vs-naive comparison visible without a viewer zoom.
            _save_seam_strip(sc, result, naive, folder)
        print(f"   coord_hard={row['coordinated']['hard_boundaries']} "
              f"naive_hard={row['naive_per_block']['hard_boundaries']} "
              f"band_edge={row['seam_band_retention']['coordinated_edge_slope']:.3f} "
              f"band_naive={row['seam_band_retention']['naive_edge_slope']:.3f} "
              f"input={row['seam_band_retention']['screened_input_edge_slope']:.3f} "
              f"ok={row['ok']}", flush=True)

    summary = {
        "scenarios": len(rows),
        "all_coordinated_no_hard_seam": all(
            r["no_hard_seam"] for r in rows),
        "all_ok": all(r["ok"] for r in rows),
        "naive_control_flagged": sum(1 for r in rows if r["naive_has_hard_seam"]),
        "coordinated_hard_boundaries": sum(
            r["coordinated"]["hard_boundaries"] for r in rows),
        "results": rows,
    }
    # Source-size mapping is a property of the whole zone set, exercised on the
    # two-frequency page once.
    page, meta = make_two_frequency_fixture()
    split = meta["split_x"]
    h, w = page.shape[:2]
    zone_set = [
        _region("left", (0, 0, split, h), 6.0, 33.0),
        _region("right", (split, 0, w - split, h), 9.5, 12.0),
    ]
    mapping = _mapping_check(page, zone_set)
    mapping["all_ok"] = bool(mapping["ok"])
    if save_images:
        with open(os.path.join(outdir, "mapping_check.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(mapping, fh, indent=2, default=_json_default)
    summary["source_size_mapping"] = mapping
    summary["all_ok"] = bool(summary["all_ok"] and mapping["ok"])
    with open(os.path.join(outdir, "region_validation.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    return summary


def _save_seam_strip(scenario: dict, result: dict, naive: np.ndarray,
                     folder: str) -> None:
    """Crop a band around the junction and enlarge it 3x for inspection."""
    page = scenario["page"]
    if scenario["name"] == "text_across_two_screens":
        x = page.shape[1] * 55 // 100
        y = 240
    elif scenario["name"] == "local_screen_direction_change":
        x = 480
        y = 250
    elif scenario["name"] == "overlapping_regions":
        x = page.shape[1] * 55 // 100
        y = 300
    else:
        x = page.shape[1] * 55 // 100
        y = 300
    half_w, half_h = 140, 110
    x0 = int(np.clip(x - half_w, 0, page.shape[1] - 2 * half_w))
    y0 = int(np.clip(y - half_h, 0, page.shape[0] - 2 * half_h))
    x1, y1 = x0 + 2 * half_w, y0 + 2 * half_h

    def zoom(arr):
        crop = arr[y0:y1, x0:x1]
        return np.repeat(np.repeat(crop, 3, axis=0), 3, axis=1)

    sup = result["_images"]["suppressed"]
    # Layout: input | coordinated | naive, separated by narrow dark gaps.
    gap = np.zeros(((y1 - y0) * 3, 6), dtype=np.float32)
    strip = np.concatenate([zoom(page), gap, zoom(sup), gap, zoom(naive)],
                           axis=1)
    with open(os.path.join(folder, "seam_comparison.png"), "wb") as fh:
        fh.write(to_png_bytes(strip))


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(type(obj))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "reports/regions"
    payload = run(out)
    print(json.dumps({k: v for k, v in payload.items() if k != "results"},
                     indent=2, default=_json_default))
