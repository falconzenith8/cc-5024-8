"""Ground-truth validation over the synthetic fixture set.

Each fixture declares its true screen geometry (or that it has none), so
detection can be scored exactly rather than eyeballed. The measurements cover
the two things a descreen tool must be judged on separately:

* periodic peak change: did the known screen frequency actually drop?
* content loss: did text edges, hairlines, and non-screen detail survive?

Fixtures are generated fresh in this process; every "after" image is the real
output of the pipeline, never a stored or pre-fixed image.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from .analysis import screen_angle_diff
from .metrics import content_vs_reference
from .pipeline import result_json, result_pngs, run_pipeline
from .synth import (
    make_fabric_fixture,
    make_known_screen_fixture,
    make_palette_fixture,
    make_photo_fixture,
    make_text_fixture,
)

CASES = [
    {
        "name": "known_6px_33deg",
        "maker": lambda: make_known_screen_fixture(period_px=6.0, angle_deg=33.0),
        "clean": None,
        "expect": {"screens": [{"period_px": 6.0, "angle_deg": 33.0}],
                   "should_select": True},
        "content": False,
    },
    {
        "name": "known_9p5px_12deg",
        "maker": lambda: make_known_screen_fixture(period_px=9.5, angle_deg=12.0),
        "clean": None,
        "expect": {"screens": [{"period_px": 9.5, "angle_deg": 12.0}],
                   "should_select": True},
        "content": False,
    },
    {
        "name": "known_12px_75deg",
        "maker": lambda: make_known_screen_fixture(period_px=12.0, angle_deg=75.0),
        "clean": None,
        "expect": {"screens": [{"period_px": 12.0, "angle_deg": 75.0}],
                   "should_select": True},
        "content": False,
    },
    {
        "name": "text_clean",
        "maker": lambda: make_text_fixture(screen=None),
        "clean": lambda: make_text_fixture(screen=None)[0],
        "expect": {"screens": [], "should_select": False},
        "content": True,
    },
    {
        "name": "text_screen_7p2px_15deg",
        "maker": lambda: make_text_fixture(),
        "clean": lambda: make_text_fixture(screen=None)[0],
        "expect": {"screens": [{"period_px": 7.2, "angle_deg": 15.0}],
                   "should_select": True},
        "content": True,
    },
    {
        "name": "fabric_clean",
        "maker": lambda: make_fabric_fixture(screen=None),
        "clean": lambda: make_fabric_fixture(screen=None)[0],
        "expect": {"screens": [], "should_select": False},
        "content": True,
    },
    {
        "name": "fabric_w_real_repeats_plus_screen",
        "maker": lambda: make_fabric_fixture(),
        "clean": lambda: make_fabric_fixture(screen=None)[0],
        "expect": {"screens": [{"period_px": 6.4, "angle_deg": 45.0}],
                   "should_select": True,
                   "must_not_select_periods": [9.0, 12.5]},
        "content": True,
    },
    {
        "name": "photo_clean",
        "maker": lambda: make_photo_fixture(screen=None),
        "clean": lambda: make_photo_fixture(screen=None)[0],
        "expect": {"screens": [], "should_select": False},
        "content": True,
    },
    {
        "name": "photo_screen_8px_7deg",
        "maker": lambda: make_photo_fixture(),
        "clean": lambda: make_photo_fixture(screen=None)[0],
        "expect": {"screens": [{"period_px": 8.0, "angle_deg": 7.0}],
                   "should_select": True},
        "content": True,
    },
    {
        "name": "palette_three_rotated",
        "maker": lambda: make_palette_fixture(),
        "clean": lambda: make_palette_fixture(screened=False)[0],
        "expect": {"screens": [
            {"period_px": 7.5, "angle_deg": 15.0},
            {"period_px": 7.725, "angle_deg": 75.0},
            {"period_px": 7.95, "angle_deg": 0.0},
        ], "should_select": True},
        "content": True,
        "deep": True,
    },
]


def _match_screens(detected: list[dict], expected: list[dict],
                   *, period_tol: float = 0.06,
                   angle_tol: float = 5.0) -> list[dict]:
    rows = []
    for e in expected:
        best = None
        for d in detected:
            if abs(d["period_px"] - e["period_px"]) / e["period_px"] <= period_tol:
                da = screen_angle_diff(d["angle_deg"], e["angle_deg"])
                if da <= angle_tol:
                    score = abs(d["period_px"] - e["period_px"]) / e["period_px"] + da / 100.0
                    if best is None or score < best[0]:
                        best = (score, d, da)
        if best is None:
            rows.append({"expected_period_px": e["period_px"],
                         "expected_angle_deg": e["angle_deg"],
                         "matched": False})
        else:
            d = best[1]
            rows.append({
                "expected_period_px": e["period_px"],
                "expected_angle_deg": e["angle_deg"],
                "matched": True,
                "detected_period_px": d["period_px"],
                "detected_angle_deg": d["angle_deg"],
                "period_error_pct": 100.0 * abs(d["period_px"] - e["period_px"]) / e["period_px"],
                "angle_error_deg": best[2],
                "confidence": d.get("confidence"),
            })
    return rows


def validate_case(case: dict) -> dict:
    img, meta = case["maker"]()
    t0 = time.time()
    pipeline = run_pipeline(
        img,
        scale=0.5,
        method="frequency_notch",
        deep_channels=bool(case.get("deep", False)),
    )
    analysis_peaks = pipeline["peaks"]
    selected = [p for p in analysis_peaks if p.get("selected")]
    expect = case["expect"]
    matches = _match_screens(analysis_peaks, expect["screens"])

    false_positives = []
    for p in selected:
        ok = any(
            abs(p["period_px"] - e["period_px"]) / e["period_px"] <= 0.06
            and screen_angle_diff(p["angle_deg"], e["angle_deg"]) <= 5.0
            for e in expect["screens"])
        if not ok:
            false_positives.append({
                "period_px": p["period_px"], "angle_deg": p["angle_deg"],
                "confidence": p["confidence"]})
    for bad in expect.get("must_not_select_periods", []):
        for p in selected:
            if abs(p["period_px"] - bad) / bad <= 0.06:
                false_positives.append({
                    "period_px": p["period_px"], "angle_deg": p["angle_deg"],
                    "confidence": p["confidence"],
                    "reason": f"matched real textile repeat {bad} px"})

    m = pipeline["metrics"]
    ps = m["peak_suppression"]
    cp = m["content_preservation"]

    # Cross-check the alternative suppression method on the same real input.
    # Both paths must remove the same detected peaks; a method that only works
    # in the unit tests is not a delivered algorithm.
    fir_check = None
    if selected:
        fir = run_pipeline(
            img, scale=0.5, method="spatial_fir",
            deep_channels=bool(case.get("deep", False)))
        fps = fir["metrics"]["peak_suppression"]
        fir_check = {
            "mean_reduction_db": fps.get("mean_reduction_db"),
            "min_reduction_db": fps.get("min_reduction_db"),
            "all_reduced": fps.get("all_reduced"),
        }

    clean_truth = None
    clean_baseline = None
    if case.get("clean") is not None:
        ref = case["clean"]()
        suppressed = pipeline["_images"]["suppressed"]
        if ref.shape == suppressed.shape:
            clean_truth = content_vs_reference(suppressed, ref)
            # The screened input is already degraded by the dot pattern itself.
            # Measuring it against the clean truth gives the starting point, so
            # the notch's own damage is clean_truth - baseline.
            clean_baseline = content_vs_reference(pipeline["_images"]["input"], ref)
            for key in ("edge_slope_retention", "edge_acutance_retention"):
                if clean_baseline.get(key):
                    clean_truth[f"{key}_vs_screened_input"] = (
                        clean_truth[key] / clean_baseline[key])

    row = {
        "name": case["name"],
        "meta": meta,
        "analysis_seconds": round(time.time() - t0, 3),
        "expected": expect,
        "detected_all": [
            {"period_px": p["period_px"], "angle_deg": p["angle_deg"],
             "selected": p["selected"], "classification": p["classification"],
             "confidence": p["confidence"]}
            for p in analysis_peaks],
        "selected": [{"period_px": p["period_px"], "angle_deg": p["angle_deg"],
                      "confidence": p["confidence"]} for p in selected],
        "screen_matches": matches,
        "all_expected_found": all(r["matched"] for r in matches),
        "false_positives": false_positives,
        "no_false_positive": len(false_positives) == 0,
        "selected_count": len(selected),
        "suppression": {
            "mean_reduction_db": ps.get("mean_reduction_db"),
            "min_reduction_db": ps.get("min_reduction_db"),
            "all_reduced": ps.get("all_reduced"),
            "per_peak": ps.get("peaks", []),
        },
        "suppression_spatial_fir": fir_check,
        "content": {
            k: {"retained_fraction": v["retained_fraction"],
                "before": v["before"], "after": v["after"]}
            for k, v in cp["metrics"].items()
        } if case.get("content", True) else None,
        "content_vs_clean_truth": clean_truth,
        "content_vs_clean_baseline": clean_baseline,
        "moire": m["moire_vs_area_reference"],
        "seam_ratio": m["seam"].get("border_step_change"),
        "seam": m["seam"],
        "lattice_explained": pipeline["lattice_explained"],
        "warnings": pipeline["warnings"],
        "output_shape": pipeline["inputs"]["output_shape"],
    }
    row["_pipeline"] = pipeline
    return row


def run_validation(outdir: str, *, save_images: bool = True) -> dict:
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for case in CASES:
        row = validate_case(case)
        pipeline = row.pop("_pipeline")
        if save_images:
            folder = os.path.join(outdir, case["name"])
            os.makedirs(folder, exist_ok=True)
            for fname, data in result_pngs(pipeline).items():
                with open(os.path.join(folder, fname), "wb") as fh:
                    fh.write(data)
            with open(os.path.join(folder, "params.json"), "w", encoding="utf-8") as fh:
                fh.write(result_json(pipeline))
        rows.append(row)
        print(f"{case['name']:34s} screens={row['selected_count']} "
              f"matched={sum(1 for r in row['screen_matches'] if r['matched'])}"
              f"/{len(row['screen_matches'])} fp={len(row['false_positives'])} "
              f"dB={row['suppression']['mean_reduction_db'] if row['suppression']['mean_reduction_db'] is not None else float('nan'):6.1f} "
              f"t={row['analysis_seconds']:.2f}s", flush=True)
    summary = _summarise(rows)
    payload = {"summary": summary, "cases": rows}
    with open(os.path.join(outdir, "validation.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return payload


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(type(obj))


def _summarise(rows: list[dict]) -> dict:
    screen_cases = [r for r in rows if r["expected"]["screens"]]
    clean_cases = [r for r in rows if not r["expected"]["screens"]]
    matched = sum(1 for r in screen_cases for m in r["screen_matches"] if m["matched"])
    total = sum(len(r["screen_matches"]) for r in screen_cases)
    fp_total = sum(len(r["false_positives"]) for r in rows)
    reduced = [r for r in screen_cases
               if r["suppression"]["all_reduced"]]
    fir_rows = [r for r in screen_cases
                if r.get("suppression_spatial_fir") is not None]
    fir_reduced = [r for r in fir_rows
                   if r["suppression_spatial_fir"]["all_reduced"]]
    content_rows = [r for r in rows if r["content"]]
    truth_rows = [r for r in rows if r.get("content_vs_clean_truth")]
    worst_edge = min(
        (r["content"]["edge_sharpness"]["retained_fraction"] for r in content_rows),
        default=None)
    worst_hair = min(
        (r["content"]["hairline_contrast"]["retained_fraction"] for r in content_rows),
        default=None)
    worst_truth_edge = min(
        (r["content_vs_clean_truth"]["edge_slope_retention"] for r in truth_rows),
        default=None)
    worst_truth_acut = min(
        (r["content_vs_clean_truth"]["edge_acutance_retention"] for r in truth_rows),
        default=None)
    worst_vs_screened_edge = min(
        (r["content_vs_clean_truth"]["edge_slope_retention_vs_screened_input"]
         for r in truth_rows
         if "edge_slope_retention_vs_screened_input" in r["content_vs_clean_truth"]),
        default=None)
    worst_vs_screened_acut = min(
        (r["content_vs_clean_truth"]["edge_acutance_retention_vs_screened_input"]
         for r in truth_rows
         if "edge_acutance_retention_vs_screened_input" in r["content_vs_clean_truth"]),
        default=None)
    return {
        "cases": len(rows),
        "known_screen_cases": len(screen_cases),
        "clean_cases": len(clean_cases),
        "screen_found": f"{matched}/{total}",
        "screen_detection_recall": matched / max(total, 1),
        "false_positive_screens": fp_total,
        "clean_cases_with_false_screen": sum(
            1 for r in clean_cases if r["selected_count"] > 0),
        "screen_cases_all_peaks_reduced": f"{len(reduced)}/{len(screen_cases)}",
        "screen_cases_all_peaks_reduced_spatial_fir": (
            f"{len(fir_reduced)}/{len(fir_rows)}" if fir_rows else None),
        "worst_edge_sharpness_retained_vs_input": worst_edge,
        "worst_hairline_contrast_retained_vs_input": worst_hair,
        "worst_edge_slope_retention_vs_clean": worst_truth_edge,
        "worst_edge_acutance_retention_vs_clean": worst_truth_acut,
        "worst_edge_slope_retention_vs_screened_input": worst_vs_screened_edge,
        "worst_edge_acutance_retention_vs_screened_input": worst_vs_screened_acut,
        "max_seam_ratio": max((r["seam_ratio"] for r in rows
                               if r["seam_ratio"] is not None), default=None),
        "moire_risk_flags": sum(
            1 for r in rows if r["moire"].get("added_moire_risk")),
    }


if __name__ == "__main__":
    out = os.environ.get("DESCREEN_REPORT_DIR", "reports/validation")
    payload = run_validation(out)
    print(json.dumps(payload["summary"], indent=2, default=_json_default))
