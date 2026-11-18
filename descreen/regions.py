"""Region-coordinated screen suppression.

Regions never filter their own block in isolation.  Every distinct parameter
group is applied to the *whole page*, and the filtered pages are combined with
smooth normalized spatial weights.  A filter whose support crosses a region
boundary therefore still reads the real neighbouring pixels, and the composite
has no per-block brightness or sharpness step.

Weights form a partition of unity::

    w_region(x, y) = raw_region(x, y) * k
    base(x, y)     = max(0, 1 - sum(raw) * k)
    k              = 1 / max(1, sum(raw))

so a lone region fades smoothly to the base image over ``feather`` pixels
instead of ending on a hard edge.  Overlapping regions are normalized; the
base layer receives whatever weight is left.

Automatic region suggestions are only candidates.  Local screen measurements
can be missing or wrong on real artwork, so every candidate carries a
confidence and ``requires_confirmation`` and is never applied silently.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .analysis import angle_diff

MIN_AUTO_ANALYSIS_SIDE = 64
DEFAULT_FEATHER_PX = 28
MAX_REGIONS = 24


def _smoothstep(t: np.ndarray | float) -> np.ndarray | float:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rect_weight(shape: tuple[int, int],
                rect: tuple[float, float, float, float],
                feather: float) -> np.ndarray:
    """Smooth rectangle weight in [0, 1].

    The ramp is centred on each rectangle edge and spans ``feather`` pixels.
    Sides lying on the image border do not ramp, because there is no
    neighbouring image area to transition into.  At a shared edge between two
    regions each side contributes 0.5, which is the intended blend point.
    """
    h, w = int(shape[0]), int(shape[1])
    x, y, rw, rh = [float(v) for v in rect]
    if rw <= 0 or rh <= 0:
        return np.zeros((h, w), dtype=np.float32)
    f = float(max(2.0, feather))
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]

    def axis_ramp(pos: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
        left = np.ones_like(pos)
        right = np.ones_like(pos)
        if lo > 1e-6:
            left = _smoothstep(0.5 + (pos - lo) / f)
        if hi < n - 1e-6:
            right = _smoothstep(0.5 + (hi - pos) / f)
        return (left * right).astype(np.float32)

    wx = axis_ramp(xs, x, x + rw, w)
    wy = axis_ramp(ys, y, y + rh, h)
    return (wy * wx).astype(np.float32)


def region_raw_weight(shape: tuple[int, int], region: dict) -> np.ndarray:
    if not region.get("enabled", True):
        return np.zeros(shape, dtype=np.float32)
    opacity = float(np.clip(region.get("opacity", 1.0), 0.0, 1.0))
    if opacity <= 0:
        return np.zeros(shape, dtype=np.float32)
    feather = float(region.get("feather_px", DEFAULT_FEATHER_PX))
    return rect_weight(shape, region["rect"], feather) * opacity


def composite_weights(shape: tuple[int, int],
                      regions: list[dict]) -> dict:
    """Build normalized per-region weights plus the base-layer weight.

    ``regions`` entries need ``id``, ``rect`` (crop-local), ``feather_px`` and
    optionally ``opacity`` / ``enabled``.  The returned maps always sum to 1.
    """
    ids: list[str] = []
    raws: list[np.ndarray] = []
    for r in regions:
        ids.append(str(r["id"]))
        raws.append(region_raw_weight(shape, r))
    if raws:
        stacked = np.stack(raws, axis=0)
        total = stacked.sum(axis=0)
    else:
        stacked = np.zeros((0,) + tuple(shape), dtype=np.float32)
        total = np.zeros(shape, dtype=np.float32)
    k = 1.0 / np.maximum(total, 1.0)
    per_region = {rid: (stacked[i] * k).astype(np.float32)
                  for i, rid in enumerate(ids)}
    base = np.clip(1.0 - total * k, 0.0, 1.0).astype(np.float32)
    s = base.copy()
    for wgt in per_region.values():
        s += wgt
    return {
        "ids": ids,
        "per_region": per_region,
        "base": base,
        "raw_total": total,
        "sum_min": float(s.min()),
        "sum_max": float(s.max()),
        "base_fraction": float(base.mean()),
    }


def effective_strength(weights: dict, region_strength: dict[str, float],
                       base_strength: float) -> np.ndarray:
    shape = weights["base"].shape
    eff = weights["base"] * float(base_strength)
    for rid, wgt in weights["per_region"].items():
        eff = eff + wgt * float(region_strength.get(rid, 0.0))
    return eff.astype(np.float32)


def group_signature(params: dict, specs: list) -> str:
    """Stable id for a filter group: identical parameters share one pass."""
    centers = []
    for s in specs:
        cfx, cfy = s.center()
        centers.append([round(float(cfx), 9), round(float(cfy), 9),
                        round(float(s.resolved_radius()), 9),
                        int(s.harmonic_order)])
    centers.sort()
    payload = {
        "method": params.get("method", "frequency_notch"),
        "strength": round(float(params.get("strength", 1.0)), 6),
        "apodization": round(float(params.get("apodization", 0.35)), 6),
        "radius": params.get("radius_cyc_per_px"),
        "harmonics": list(params.get("harmonic_orders", [1, 2])),
        "lattice": bool(params.get("include_lattice_lines", True)),
        "pad": params.get("pad"),
        "feather": params.get("feather"),
        "taps": params.get("taps"),
        "centers": centers,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def map_region_set(regions: list[dict],
                   src_shape: tuple[int, int],
                   dst_shape: tuple[int, int]) -> list[dict]:
    """Scale region rects and feathers from one source size to another.

    Region geometry is stored in source pixel coordinates, so a change of
    source size (or an output resample factor) is a pure coordinate mapping.
    """
    sw, sh = int(src_shape[1]), int(src_shape[0])
    dw, dh = int(dst_shape[1]), int(dst_shape[0])
    if sw <= 0 or sh <= 0:
        raise ValueError("source shape must be positive")
    sx = dw / float(sw)
    sy = dh / float(sh)
    sf = math.sqrt(sx * sy)
    out = []
    for r in regions:
        q = json.loads(json.dumps(r))
        x, y, w, h = q["rect"]
        q["rect"] = [x * sx, y * sy, w * sx, h * sy]
        q["feather_px"] = float(q.get("feather_px", DEFAULT_FEATHER_PX)) * sf
        q["mapped_from"] = {"source_shape": [sh, sw],
                            "target_shape": [dh, dw],
                            "scale": [sx, sy]}
        out.append(q)
    return out


def output_regions(regions: list[dict], scale: float,
                   offset: tuple[int, int] = (0, 0),
                   source_shape: tuple[int, int] | None = None) -> list[dict]:
    """Region geometry as it appears on the resampled output grid.

    ``rect`` is reported relative to the output image (i.e. the processed
    crop); ``rect_page`` adds the output-scale ROI offset.  Feathers scale with
    the same factor, so a re-run from the exported values reproduces the same
    transition width.
    """
    ox, oy = offset
    rows = []
    for r in regions:
        x, y, w, h = [float(v) for v in r["rect"]]
        rows.append({
            "id": r["id"],
            "rect": [x * scale, y * scale, w * scale, h * scale],
            "rect_page": [(ox + x) * scale, (oy + y) * scale,
                          w * scale, h * scale],
            "feather_px": float(r.get("feather_px", DEFAULT_FEATHER_PX)) * scale,
            "source_shape": list(source_shape) if source_shape else None,
            "scale": float(scale),
        })
    return rows


def _downsample(arr: np.ndarray, max_side: int = 96) -> np.ndarray:
    h, w = arr.shape[:2]
    step = max(1, int(math.ceil(max(h, w) / float(max_side))))
    if step == 1:
        return arr.astype(np.float32)
    hh = int(math.ceil(h / step))
    ww = int(math.ceil(w / step))
    out = np.zeros((hh, ww), dtype=np.float32)
    for i in range(hh):
        for j in range(ww):
            out[i, j] = float(arr[i * step:(i + 1) * step,
                                 j * step:(j + 1) * step].mean())
    return out


def _rounded_grid(arr: np.ndarray) -> list:
    return np.round(arr, 3).tolist()


def weight_grid(weights: dict, eff: np.ndarray,
                *, max_side: int = 72) -> dict:
    """Compact grid for the browser overlay and per-pixel readout."""
    per = {rid: _rounded_grid(_downsample(wgt, max_side))
           for rid, wgt in weights["per_region"].items()}
    return {
        "shape": list(_downsample(eff, max_side).shape),
        "per_region": per,
        "base": _rounded_grid(_downsample(weights["base"], max_side)),
        "effective_strength": _rounded_grid(_downsample(eff, max_side)),
    }


_WEIGHT_STOPS = np.array([
    [0.00, 10, 12, 16],
    [0.25, 22, 74, 120],
    [0.50, 32, 168, 186],
    [0.75, 222, 188, 62],
    [1.00, 255, 250, 222],
], dtype=np.float32)


def weight_heatmap(eff: np.ndarray) -> np.ndarray:
    stops = _WEIGHT_STOPS
    x = stops[:, 0]
    v = np.clip(eff, 0.0, 1.0)
    out = np.empty(v.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, x, stops[:, c + 1]) / 255.0
    return out


_REGION_COLORS = np.array([
    [255, 82, 82], [82, 160, 255], [90, 220, 140], [245, 190, 70],
    [200, 110, 255], [70, 225, 225], [255, 140, 210], [170, 220, 80],
], dtype=np.float32) / 255.0


def weight_overlay(weights: dict,
                   *, max_side: int = 256) -> tuple[np.ndarray, list[dict]]:
    """RGBA overlay: each region a distinct hue, opacity = its weight.

    This is what the page draws over the source so the operator sees the
    parameter weights themselves, not merely rectangles.
    """
    h, w = weights["base"].shape
    step = max(1, int(math.ceil(max(h, w) / float(max_side))))
    gh = int(math.ceil(h / step))
    gw = int(math.ceil(w / step))
    rgba = np.zeros((gh, gw, 4), dtype=np.float32)
    legend = []
    for i, rid in enumerate(weights["ids"]):
        wgt = weights["per_region"][rid]
        small = _downsample(wgt, max(gw, gh))
        if small.shape != (gh, gw):
            pad = np.zeros((gh, gw), dtype=np.float32)
            pad[:small.shape[0], :small.shape[1]] = small[:gh, :gw]
            small = pad
        color = _REGION_COLORS[i % len(_REGION_COLORS)]
        tint = color[None, None, :] * np.clip(small, 0.0, 1.0)[..., None]
        rgba[..., :3] += tint
        rgba[..., 3] = np.maximum(rgba[..., 3], small)
        legend.append({"id": rid, "color": [float(c) for c in color]})
    alpha = np.clip(rgba[..., 3], 0.0, 1.0)[..., None]
    rgb = np.clip(rgba[..., :3] / np.maximum(alpha, 1e-6), 0.0, 1.0)
    # Weight 1 must read as a strong tint without washing the source out; the
    # alpha is deliberately capped so the page image stays legible underneath.
    return np.concatenate([rgb, alpha * 0.45], axis=-1), legend


def _tile_observations(rgb: np.ndarray, *, tile: int = 320,
                       max_tiles: int = 12) -> tuple[list[dict], dict]:
    """Analyze overlapping tiles and record the whole screen set of each.

    A tile is represented by *all* screens it contains, not by a single best
    peak: a colour separation can carry three rotated screens in the same
    place, and they must end up in one region's parameter set rather than in
    three layers that would blend each other's suppression away.
    """
    from .analysis import analyze_image

    h, w = rgb.shape[:2]
    tile = int(min(tile, h, w))
    step = max(96, tile // 2)
    spots = []
    for y in range(0, max(1, h - tile + 1), step):
        for x in range(0, max(1, w - tile + 1), step):
            spots.append((x, y))
    corner = (max(0, w - tile), max(0, h - tile))
    if corner not in spots:
        spots.append(corner)
    if len(spots) > max_tiles:
        idx = np.linspace(0, len(spots) - 1, max_tiles).round().astype(int)
        spots = [spots[i] for i in dict.fromkeys(idx.tolist())]

    obs: list[dict] = []
    failures: list[dict] = []
    deep = rgb.ndim == 3
    for (x, y) in spots:
        sub = rgb[y:y + tile, x:x + tile]
        try:
            res = analyze_image(sub, None, deep_channels=deep,
                                flat_tile_assist=True)
        except Exception as exc:
            failures.append({"rect": [x, y, tile, tile],
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        chosen = [p for p in res["peaks"] if p.get("selected")]
        confident = True
        if not chosen:
            amb = [p for p in res["peaks"]
                   if p.get("classification") == "ambiguous_periodic"]
            chosen = sorted(amb, key=lambda p: p.get("confidence", 0.0),
                            reverse=True)[:2]
            confident = False
        if not chosen:
            continue
        sig = []
        for p in chosen:
            sig.append({
                "period_px": float(p["period_px"]),
                "angle_deg": float(p["angle_deg"]),
                "confidence": float(p.get("confidence", 0.0)),
                "classification": p.get("classification", "unknown"),
                "basis_b": p.get("basis_b"),
                "selected": bool(p.get("selected")),
            })
        obs.append({
            "rect": [x, y, tile, tile],
            "peaks": sig,
            "confident": confident,
            "flat_tile_assist_used": bool(res.get("flat_tile_assist_used")),
        })
    return obs, {"tiles_analyzed": len(spots), "failures": failures,
                 "deep_channels": deep}


def _same_signature(a: list[dict], b: list[dict],
                    *, period_tol: float = 0.05,
                    angle_tol: float = 5.0) -> bool:
    """Two tiles belong together when their screen sets match one-to-one."""
    if len(a) != len(b):
        return False
    used = [False] * len(b)
    for p in a:
        hit = None
        for i, q in enumerate(b):
            if used[i]:
                continue
            if (abs(p["period_px"] - q["period_px"])
                    / max(q["period_px"], 1e-9) <= period_tol
                    and abs(angle_diff(p["angle_deg"], q["angle_deg"])) <= angle_tol):
                hit = i
                break
        if hit is None:
            return False
        used[hit] = True
    return True


def suggest_regions(rgb: np.ndarray,
                    *, max_candidates: int = 6,
                    tile: int = 320) -> dict:
    """Propose rectangular regions from local screen measurements.

    Tiles with the same screen set are grouped into one region whose parameter
    set carries *all* of those screens.  Candidates are never applied
    silently: the page layout is not assumed to be reliably recognised, so each
    one is marked ``requires_confirmation`` and low-confidence, ambiguous-only
    groups are flagged.
    """
    h, w = rgb.shape[:2]
    obs, diag = _tile_observations(rgb, tile=tile)
    groups: list[dict] = []
    for o in obs:
        target = None
        for g in groups:
            if _same_signature(o["peaks"], g["signature"]):
                target = g
                break
        if target is None:
            groups.append({"signature": o["peaks"], "members": [o]})
        else:
            target["members"].append(o)
            # Merge each screen estimate across supporting tiles.
            for sig in target["signature"]:
                hits = [p for o2 in target["members"] for p in o2["peaks"]
                        if abs(p["period_px"] - sig["period_px"])
                        / max(sig["period_px"], 1e-9) <= 0.05
                        and abs(angle_diff(p["angle_deg"], sig["angle_deg"])) <= 5.0]
                n = len(hits)
                sig["period_px"] = float(np.mean([p["period_px"] for p in hits]))
                # Folded-axis mean without circular wrap-around; tiles already
                # matched within 5 deg, so a plain mean is stable.
                sig["angle_deg"] = float(np.mean([p["angle_deg"] for p in hits]))
                sig["confidence"] = float(np.mean([p["confidence"] for p in hits]))

    groups.sort(key=lambda g: (len(g["members"]),
                               max(p["confidence"] for p in g["signature"])),
                reverse=True)
    candidates = []
    uncovered_tiles = [o for o in obs if not o["confident"]]
    for i, g in enumerate(groups[:max_candidates]):
        xs0 = min(m["rect"][0] for m in g["members"])
        ys0 = min(m["rect"][1] for m in g["members"])
        xs1 = max(m["rect"][0] + m["rect"][2] for m in g["members"])
        ys1 = max(m["rect"][1] + m["rect"][3] for m in g["members"])
        if xs0 <= 0:
            xs0 = 0
        if ys0 <= 0:
            ys0 = 0
        if xs1 >= w:
            xs1 = w
        if ys1 >= h:
            ys1 = h
        conf = float(np.mean([p["confidence"] for p in g["signature"]]))
        any_confident = any(m["confident"] for m in g["members"])
        support = len(g["members"])
        candidates.append({
            "id": f"auto-{i + 1}",
            "kind": "screen_group",
            "status": "candidate",
            "requires_confirmation": True,
            "rect": [int(xs0), int(ys0), int(xs1 - xs0), int(ys1 - ys0)],
            "feather_px": DEFAULT_FEATHER_PX,
            "opacity": 1.0,
            "enabled": True,
            "confidence": conf,
            "support_tiles": support,
            "auto_peaks": True,
            "peaks": [{"period_px": p["period_px"],
                       "angle_deg": p["angle_deg"],
                       "confidence": p["confidence"],
                       "source": "tile_analysis"} for p in g["signature"]],
            "seed_peaks": [{"period_px": p["period_px"],
                            "angle_deg": p["angle_deg"]} for p in g["signature"]],
            "params": {
                "method": "frequency_notch",
                "strength": 1.0,
                "radius_cyc_per_px": None,
                "apodization": 0.35,
                "harmonic_orders": [1, 2],
                "include_lattice_lines": True,
            },
            "notes": [
                f"{len(g['signature'])} screen(s) measured in {support} tile(s): "
                + ", ".join(f"{p['period_px']:.2f} px / {p['angle_deg']:.1f} deg"
                            for p in g["signature"]),
                ("only ambiguous periodic evidence was found in part of this "
                 "area" if not any_confident else
                 "measurements were classified as halftone screens"),
                "candidate only; confirm geometry and parameters before applying",
            ],
        })
    covered = 0.0
    for cand in candidates:
        x, y, rw, rh = cand["rect"]
        covered += (rw * rh) / max(1.0, float(w * h))
    return {
        "candidates": candidates,
        "diagnostics": diag,
        "coverage_estimate": float(min(1.0, covered)),
        "tiles_without_screen": len(uncovered_tiles),
        "manual_review_required": True,
        "note": "Candidates are local periodic measurements, not a recognised "
                "page layout. Tiles without periodic evidence are intentionally "
                "left unassigned; add or edit regions manually.",
    }


def _region_rect_clipped(region: dict, shape: tuple[int, int]
                         ) -> tuple[int, int, int, int] | None:
    h, w = shape
    x, y, rw, rh = [int(round(float(v))) for v in region["rect"]]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + rw)
    y1 = min(h, y + rh)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1 - x0, y1 - y0


def region_auto_analysis(rgb: np.ndarray, region: dict,
                         *, deep_channels: bool = False,
                         flat_tile_assist: bool = True) -> dict:
    """Analyze one region for its own screens, with explicit failure states."""
    rect = _region_rect_clipped(region, rgb.shape[:2])
    if rect is None:
        return {"status": "outside_image", "peaks": [], "warnings": [
            "region does not intersect the image"]}
    x, y, rw, rh = rect
    if min(rw, rh) < MIN_AUTO_ANALYSIS_SIDE:
        return {
            "status": "too_small_for_auto_analysis",
            "peaks": [],
            "warnings": [
                f"region is {rw}x{rh} px; automatic screen analysis needs "
                f"at least {MIN_AUTO_ANALYSIS_SIDE} px on each side. Supply "
                f"period/angle manually or use the seed peaks."],
            "rect_used": [x, y, rw, rh],
        }
    from .analysis import analyze_image
    try:
        res = analyze_image(rgb, (x, y, rw, rh),
                            deep_channels=deep_channels,
                            flat_tile_assist=flat_tile_assist)
    except Exception as exc:  # reported to the page, never hidden
        return {"status": "error", "peaks": [],
                "error": f"{type(exc).__name__}: {exc}",
                "warnings": ["region analysis failed; parameters were not "
                             "applied to this region"],
                "rect_used": [x, y, rw, rh]}
    selected = [p for p in res["peaks"] if p.get("selected")]
    peaks = [{"period_px": float(p["period_px"]),
              "angle_deg": float(p["angle_deg"]),
              "confidence": float(p.get("confidence", 0.0)),
              "source": "region_analysis"} for p in selected]
    warnings = list(res.get("warnings", []))
    status = "ok" if peaks else "no_screen_detected"
    return {"status": status, "peaks": peaks, "peaks_full": selected,
            "warnings": warnings,
            "rect_used": [x, y, rw, rh],
            "auto_selected": res.get("auto_selected", 0),
            "flat_tile_assist_used": res.get("flat_tile_assist_used", False)}


def region_boundaries(regions: list[dict],
                      shape: tuple[int, int]) -> list[dict]:
    """Interior edges of enabled regions, for continuity measurement.

    Each edge is measured over the extent of its own region; an edge on the
    image border is omitted because there is no neighbouring side.  Identical
    edges reported by two adjacent regions are measured once.
    """
    h, w = int(shape[0]), int(shape[1])
    rows: list[dict] = []
    seen: set[tuple[str, int, int, int]] = set()
    for r in regions:
        if not r.get("enabled", True):
            continue
        x, y, rw, rh = [int(round(float(v))) for v in r["rect"]]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        key = ("x", int(x0), int(y0), int(y1))
        if 0 < x0 < w and key not in seen:
            seen.add(key)
            rows.append({"axis": "x", "pos": int(x0),
                         "lo": int(y0), "hi": int(y1),
                         "region": str(r["id"])})
        key = ("x", int(x1), int(y0), int(y1))
        if 0 < x1 < w and key not in seen:
            seen.add(key)
            rows.append({"axis": "x", "pos": int(x1),
                         "lo": int(y0), "hi": int(y1),
                         "region": str(r["id"])})
        key = ("y", int(y0), int(x0), int(x1))
        if 0 < y0 < h and key not in seen:
            seen.add(key)
            rows.append({"axis": "y", "pos": int(y0),
                         "lo": int(x0), "hi": int(x1),
                         "region": str(r["id"])})
        key = ("y", int(y1), int(x0), int(x1))
        if 0 < y1 < h and key not in seen:
            seen.add(key)
            rows.append({"axis": "y", "pos": int(y1),
                         "lo": int(x0), "hi": int(x1),
                         "region": str(r["id"])})
    return rows


def _dependents(regions: list[dict], roots: set[str]) -> set[str]:
    """Transitive dependents by explicit ``depends_on`` / ``inherits_from``."""
    wanted = set(roots)
    changed = True
    while changed:
        changed = False
        for r in regions:
            rid = str(r["id"])
            if rid in wanted:
                continue
            deps = set(str(v) for v in r.get("depends_on", []))
            if r.get("inherits_from"):
                deps.add(str(r["inherits_from"]))
            if deps & wanted:
                wanted.add(rid)
                changed = True
    return wanted


def recompute_regions(rgb: np.ndarray, regions: list[dict],
                      dirty_ids: list[str] | None = None,
                      *, deep_channels: bool = False,
                      flat_tile_assist: bool = True) -> dict:
    """Re-analyze only dirty regions and their declared dependents.

    A user edit to a rectangle invalidates that region's own screen
    measurement; regions that explicitly inherit from it are recomputed too.
    Unrelated regions keep their existing measurements, so an edit does not
    silently change the rest of the page.
    """
    out = [json.loads(json.dumps(r)) for r in regions]
    dirty = set(str(v) for v in (dirty_ids or [])) if dirty_ids is not None else {
        str(r["id"]) for r in out}
    targets = _dependents(out, dirty)
    recomputed: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    for r in out:
        rid = str(r["id"])
        if rid not in targets:
            r["recompute"] = {"recomputed": False, "reason": "not dirty"}
            continue
        if not r.get("auto_peaks", True):
            r["recompute"] = {"recomputed": False,
                              "reason": "manual peaks are preserved"}
            if r.get("status") in (None, "pending"):
                r["status"] = "manual"
            continue
        res = region_auto_analysis(rgb, r, deep_channels=deep_channels,
                                   flat_tile_assist=flat_tile_assist)
        # ``status`` records the operator-facing confirmation state; the
        # analysis outcome is kept separately so recomputing does not silently
        # promote an unconfirmed candidate to a confirmed zone.
        prior_status = r.get("status")
        r["analysis_status"] = res["status"]
        if prior_status not in ("candidate", "confirmed", "manual"):
            r["status"] = res["status"]
        r["analysis"] = {k: v for k, v in res.items() if k != "peaks"}
        if res.get("peaks"):
            r["peaks"] = res["peaks"]
            r["analysis_source"] = "region_analysis"
        elif r.get("seed_peaks") and res["status"] in (
                "too_small_for_auto_analysis", "no_screen_detected",
                "error", "outside_image"):
            # Explicit, visible fallback to the candidate's measured seed.
            r["peaks"] = [dict(p, source="candidate_seed")
                          for p in r["seed_peaks"]]
            r["analysis_source"] = "candidate_seed"
            warnings.append(
                f"{rid}: automatic analysis returned {res['status']}; using the "
                f"candidate seed peak. Verify it in the spectrum view.")
        else:
            r["peaks"] = []
            r["analysis_source"] = "none"
        if res["status"] == "error":
            errors.append(f"{rid}: {res.get('error', 'analysis failed')}")
        warnings.extend(f"{rid}: {msg}" for msg in res.get("warnings", []))
        r["recompute"] = {"recomputed": True,
                          "status": res["status"],
                          "peaks": len(r["peaks"])}
        recomputed.append(rid)
    return {
        "regions": out,
        "dirty_ids": sorted(dirty),
        "recomputed": recomputed,
        "errors": errors,
        "warnings": warnings,
        "note": "only dirty regions and their declared dependents were re-analyzed",
    }
