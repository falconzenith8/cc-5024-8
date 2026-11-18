# Descreen

Halftone screen analysis, suppression, and anti-aliased resampling for scanned
print. One page can carry text, photos, and several screen angles at once; the
zone pipeline applies different parameters per zone and composites them on a
single output grid.

## Start the interface

No build step and no third-party web framework. Requirements: Python 3.10+ with
`numpy`, `scipy`, `pillow`.

```bash
pip install numpy scipy pillow
python3 -m descreen.server --host 127.0.0.1 --port 8000
```

Open the URL it prints: <http://127.0.0.1:8000/>.

The page loads the two-frequency fixture automatically. Choose another fixture in
**1 · Source**, or upload a scanned page.

## Workflow

1. **Load** a fixture or upload an image.
2. **Analyze region** (panel 1) to get the page's screens. This whole-image pass
   still drives the original single-parameter pipeline in panel 4.
3. **Zones** (panel 3):
   - *Suggest zones from screens* measures overlapping tiles and proposes
     candidate rectangles. Candidates are proposals only — they are marked
     `candidate` and require confirmation; the extractor never assumes the page
     layout was recognised reliably.
   - Edit each zone's rectangle, feather, opacity, strength, method, and its
     screen list (`period@angle`, comma separated). Editing marks the zone dirty.
   - *Recompute changed zones* re-analyzes only the zones you edited plus those
     that declare `depends_on` / `inherits_from` them. A zone the detector cannot
     measure keeps its candidate seed peak and says so, rather than silently
     changing.
   - *Show weight overlay* draws the per-pixel parameter weights over the source
     — hues per zone, brightness by weight — not just rectangles. Hovering the
     image reports the effective strength and the constituent zone weights at
     that pixel.
4. **Run zone pipeline** (panel 4). Set *Base strength outside zones* if the whole
   page carries a screen; that base layer needs a whole-image analysis first. If
   base strength is 0 and part of the crop lies outside every zone, the result
   says how much is left unfiltered.
5. **Export** (panel 5): parameters `.json` (region geometry, feather, weight
   rule, coordinate binding; the heavy per-pixel grid is omitted but the rule to
   reproduce it is included), plus output/suppressed/weight PNGs.

If an operation fails, the server's error message is shown in the page banner and
no partial result replaces the last good one.

## How zones avoid seams

Every distinct parameter group is filtered over the **whole** crop, then the
results are blended with normalized smooth weights (a partition of unity):

```
w_raw  = smooth_rect * opacity
k      = 1 / max(1, sum(w_raw))        # overlap normalisation
w_zone = w_raw * k
w_base = 1 - sum(w_raw) * k
```

Rectangle ramps are smoothstep, centred on each edge, and do not ramp on sides
that sit on the image border. A filter neighbourhood crossing a boundary
therefore samples real pixels from both sides; nothing is processed or pasted
block by block. The exported parameters carry the same rectangles and feathers,
so the file reproduces the previewed transition, not just the boxes.

Region rectangles are stored in **absolute page pixels**. Changing the ROI
re-binds them (`rect_in_crop = rect_page - roi_origin`); resizing the source maps
them (`map_region_set`, feather scaled by `sqrt(sx*sy)`). They do not drift.

## Verify it

```bash
# Four scenarios: text across two screens, a local direction change,
# overlapping zones, a very narrow zone. Each is compared with a naive
# per-block paste used as the negative control.
python3 -m descreen.validate_regions reports/regions
```

Outputs `reports/regions/region_validation.json` plus per-scenario input, clean
reference, naive control, difference, `seam_comparison.png`, and weight map. The
coordinated composite reports zero hard boundaries in all four scenarios; the
naive control is flagged on the scenarios where independent blocks produce a
real step. `REPORT.md` in that folder is a readable summary.

The same run also proves two properties that are easy to lose:

- **Export fidelity** — the weights are rebuilt from the exported JSON alone
  (rectangles, feather, opacity, enabled) and compared with the run's weight
  grid. They match to the grid's rounding, so the file reproduces the previewed
  transition, not merely the boxes.
- **Coordinate binding** — resizing the source scales rectangles and feathers
  and lands the zones on the same content (zero drift measured), and running
  through a cropped ROI shifts them by the ROI origin without moving the content
  a zone owns.

Measurement notes:

- **Boundary continuity** looks at the *processing delta* (`after - before`)
  along each edge and flags a transition concentrated in one pixel. A smooth ramp
  spreads the change over the feather width; a per-block paste puts it in a step.
  A boundary already present in the input is exempted from the sharpness check,
  because the input itself contains a real splice there.
- **Detail retention** is measured against the clean, unscreened artwork, both
  whole-image and restricted to a band around the zone edges. In the delivered
  scenarios the coordinated and naive implementations both retain text (~99% of
  the input's edge slope in the seam band); the difference they exhibit is
  boundary continuity, not local detail loss. The metric reports this rather than
  claiming otherwise.

Browser-level checks (load, analyze, zone suggest/recompute/run, weight overlay
and readout, export, and an error banner for a rejected run) live in
`tests/ui_check.js` and require `playwright-core` plus a Chromium build.

## Layout

```
descreen/
  analysis.py         FFT peak detection, channel handling, spectrum rendering
  process.py          notch / FIR filtering, coordinated composite, area resample
  regions.py          weights, zone suggestion, dirty recompute, coordinate mapping
  pipeline.py         whole-image and zone pipelines, result serialisation
  metrics.py          peak suppression, content retention, boundary continuity
  validate_regions.py four-scenario verification with a naive negative control
  synth.py            deterministic ground-truth fixtures
  server.py           stdlib HTTP server and JSON API
  static/             the single-page UI (vanilla JS, no build step)
```
