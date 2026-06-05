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

- `single_wheel_experiments.csv` — **experiment-vs-model worksheet** for the
  experimental anchor of Layer 3. Each row is one measured single-wheel
  operating point: wheel geometry, vertical load, slip, the soil simulant
  name (Bekker parameters resolved from `../soil_simulants.csv`), and the
  Janosi-Hanamoto shear modulus `soil_shear_modulus_k_m`. The
  `meas_drawbar_pull_n` / `meas_sinkage_m` / `meas_torque_nm` columns hold
  point measurements traced from the source figures.

  Consumed by `roverdevkit.validation.terramechanics_experiment`
  (`compare_to_experiment`, `summarise`) and
  `tests/test_terramechanics_experiment.py`. The harness runs the
  analytical Bekker-Wong kernel at every operating point and reports
  residuals + percentage errors against the measured columns. Notebook
  `paper/paper_figures.ipynb` renders the terramechanics-experiment figure
  (`reports/figures/fig_terramechanics_experiment.png`) from it.

## Sources

- **Ding et al. 2011**, *J. Terramechanics* 48(1):27-45 — rigid single-wheel
  slip/sinkage/drawbar-pull experiments (R=135/157 mm, b=110/165 mm, lugs
  0-15 mm, loads 30/80/150 N, slip 0-0.6). Digitised from Fig. 8/9 (Wh3
  family: Wh34 smooth + Wh32 grousered at 80 N; soil Bekker params from
  Table 2 → `Ding2011_planetary_simulant` in `../soil_simulants.csv`). BW
  reproduces drawbar pull within the literature model-form band (~27 %
  median |error|).
- **Wang & Han 2016**, *J. Korean Geotech. Soc.* 32(11):97-108 (open access) —
  KLS-1 single-wheel testbed (R=85 mm, b=80 mm, 59 N), smooth vs grousered
  (h=10 mm), slip 0.1-0.5. Digitised from Fig. 14. The paper publishes only
  shear strength (Table 2: C=1.716 kPa, φ=40.6°), so the pressure-sinkage
  terms in `KLS-1` (`../soil_simulants.csv`) are proxied from PSD-matched
  JSC-1A. This is a deliberate **stress case**: BW over-predicts smooth-wheel
  DP and sinkage and cannot capture the measured DP collapse at s≈0.5, so its
  accuracy band in the test is intentionally loose (it is not a validation
  claim).
- **Hurrell et al. 2025**, *Space Sci. Rev.* 221 art. 37 (open access, CC-BY) —
  Rashid-1 micro-rover wheel (R=100 mm, b=80 mm, 14 grousers h=20 mm, 24.5 N)
  on FJS-1, slip 0.1-0.5. Digitised from Figs. 5/6:
  `meas_drawbar_pull_n` = traction coefficient F_x/F_z (Fig. 5) × 24.5 N;
  `meas_sinkage_m` from Fig. 6; torque not reported. Soil = `Hurrell2025_FJS1`
  (`../soil_simulants.csv`): cohesion 2.4 kPa (Ozaki et al. 2023) and AoR 38°,
  pressure-sinkage proxied from catalogue FJS-1. **Most application-relevant
  case** (in-scope micro-rover wheel + load); BW lands within band on both DP
  (~24 % median) and sinkage (~28 %).
- **Iizuka & Kubota 2011** — grousered wheel experiments (shear-thrust gain
  vs grouser packing density).
- **Wong** — datasets from *Theory of Ground Vehicles* (4th ed.) ch. 4.

Keep raw digitised traces in `raw/` (git-ignored, re-downloadable from the
papers); curated, checked-in reference rows live in
`wong_layer3_reference.csv` / `single_wheel_experiments.csv` at this level.
