# Region descreen samples (generated)

Regenerate with `python3 -m descreen.validate_regions reports/regions`.

Each scenario compares the coordinated composite with a naive per-block
paste of independently filtered tiles (the negative control).

## text_across_two_screens

one body of text crosses 6 px/33 deg on the left and 9.5 px/12 deg on the right

- coordinated hard boundaries: **0** of 1 (seam suspected: False)
- naive per-block hard boundaries: **1** (seam suspected: True)
- min transition width / feather: 0.39
- weight sum range: 1.000–1.000, base-only fraction 0.000
- seam-band edge-slope retention: coordinated 0.654, naive 0.653, screened input 0.661
- seam-band acutance retention: coordinated 0.660, naive 0.659
- export rebuild from JSON matches the run: True (max delta 0.000000, heavy grid omitted: True)
- per-zone:
  - left: weight 1.00, detail 0.985, suppression 73.7 dB
  - right: weight 0.99, detail 0.980, suppression 57.1 dB

## local_screen_direction_change

a 620x200 patch rotates its screen from 33 deg to 78 deg; the surrounding page stays at 33 deg

- coordinated hard boundaries: **0** of 4 (seam suspected: False)
- naive per-block hard boundaries: **0** (seam suspected: False)
- min transition width / feather: 0.77
- weight sum range: 1.000–1.000, base-only fraction 0.000
- seam-band edge-slope retention: coordinated 0.676, naive 0.676, screened input 0.677
- seam-band acutance retention: coordinated 0.677, naive 0.676
- export rebuild from JSON matches the run: True (max delta 0.000000, heavy grid omitted: True)
- per-zone:
  - surround: weight 0.91, detail 0.985, suppression 27.7 dB
  - band: weight 0.49, detail 0.979, suppression 6.0 dB

## overlapping_regions

two regions share a 240 px strip; the overlap is normalized and still filters the whole crop

- coordinated hard boundaries: **0** of 3 (seam suspected: False)
- naive per-block hard boundaries: **1** (seam suspected: True)
- min transition width / feather: 0.23
- weight sum range: 1.000–1.000, base-only fraction 0.000
- seam-band edge-slope retention: coordinated 0.658, naive 0.655, screened input 0.662
- seam-band acutance retention: coordinated 0.665, naive 0.663
- export rebuild from JSON matches the run: True (max delta 0.000000, heavy grid omitted: True)
- per-zone:
  - left: weight 0.84, detail 0.989, suppression 22.6 dB
  - right: weight 0.81, detail 0.984, suppression 18.5 dB

## very_narrow_region

a 14 px wide strip with its own reduced strength inside the left screen; no hard edge may appear

- coordinated hard boundaries: **0** of 7 (seam suspected: False)
- naive per-block hard boundaries: **5** (seam suspected: True)
- min transition width / feather: 0.38
- weight sum range: 1.000–1.000, base-only fraction 0.000
- seam-band edge-slope retention: coordinated 0.661, naive 0.660, screened input 0.666
- seam-band acutance retention: coordinated 0.667, naive 0.667
- export rebuild from JSON matches the run: True (max delta 0.000000, heavy grid omitted: True)
- per-zone:
  - left: weight 0.99, detail 0.984, suppression 39.2 dB
  - narrow: weight 0.47, detail 0.991, suppression 9.0 dB
  - right: weight 0.99, detail 0.980, suppression 57.1 dB

## source_size_mapping

- page 1200x900 → 600x450: max rect drift 0.000 px, feathers scaled True, weights still sum to 1.000–1.000
- ROI rebind correct: True

All scenarios: coordinated no hard seam = True; naive control flagged in 3/4.

Note: the naive control is *not* flagged on the direction-change
scenario because the input there already contains a genuine screen
phase splice at the band edge; the continuity metric exempts a step
that is present before processing. Detail retention is nearly equal
between the two implementations in the seam band — they differ in
boundary continuity, not local detail loss.
