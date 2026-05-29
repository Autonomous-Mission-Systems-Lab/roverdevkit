# Validation data

Single-wheel testbed data digitised from published papers. Used **only** to
validate the evaluator; never used for training the surrogate.

## Active reference grids

- `wong_layer3_reference.csv` — Layer-3 BW-vs-published reference grid
  exercised by `tests/test_terramechanics.py::test_layer3_published_reference_grid`.
  Each row is one (wheel, soil, vertical load, slip) operating point with
  per-quantity `[lo, hi]` tolerance bands. Three row kinds:

    1. **`characterisation`** — Wong (2008) §4.2-style worked-example
       fixture (JSC-1A canonical Bekker parameters; R=0.10 m, b=0.06 m;
       50 N at slip ∈ {0.05, 0.20, 0.50}). Bounds pinned at the BW
       kernel's verified outputs to within ±5 % so the test guards
       against unintended kernel drift while staying inside the
       ±15-30 % BW model-form band reported in the literature.
    2. **`published_rover_class`** — Apollo nominal regolith × a smooth
       Pragyan-class wheel (R=0.135 m, b=0.10 m, W=70.2 N) and a
       grousered Yutu-2-class wheel (R=0.165 m, b=0.150 m, h_g=0.012 m,
       N_g=14, W=36.4 N). Bounds sized at the published Bekker-Wong
       model-form error (Ishigami 2007; Ding et al. 2011).
    3. **`closed_form_limit`** — Iizuka & Kubota (2011) grouser-thrust
       limit cases (smooth wheel ⇒ lift factor ≡ 1).

  Appending rows is additive — the test reads the CSV with
  `csv.DictReader` and parametrises one case per row.

## Target sources for additional digitisation

- Ding et al. 2011 — Lunar rover single-wheel slip/sinkage experiments
  (drawbar-pull-vs-slip on FJS-1).
- Iizuka & Kubota 2011 — Grousered wheel experiments (shear-thrust gain
  vs grouser packing density).
- Wong — datasets from *Theory of Ground Vehicles* (4th ed.) ch. 4.

Keep raw digitised CSVs in `raw/` (git-ignored, re-downloadable from the
papers) and curated, checked-in reference rows in
`wong_layer3_reference.csv` at this level.
