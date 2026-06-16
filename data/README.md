# Data

Small, curated datasets and citations live here. Large generated datasets
(LHS samples) are git-ignored — see `.gitignore`.

## Rover data

Three consumers describe the same set of rovers for verification, each holding
*purpose-specific* values, all reconciled against one canonical facts file:

- `rovers.yaml` — **canonical published-facts reference** (single source of
  truth). Holds only published/citable facts (mass, wheels, grousers, landing
  latitude, traverse/peak-solar/thermal truth, ...) with **per-field
  provenance** (`value` + `provenance` ∈ {published, derived, imputed} +
  `source`). Loaded by `roverdevkit/validation/rover_facts.py`. It deliberately
  excludes modeling-derived quantities (chassis mass, torque anchor, panel
  efficiency, thermal architecture, scenario duty cycles) that legitimately
  differ per consumer. `tests/test_rover_facts.py` enforces that the consumers
  below agree with the `published`/`derived` facts here, so the sources cannot
  silently drift.
- `mass_validation_set.csv` — published-rover mass and subsystem inputs used
  by `roverdevkit/mass/validation.py` to check the bottom-up mass model.
  Source/provenance details live in each row's `citation` and `imputation_notes`.
- `published_traverse_data.csv` — flown-rover traverse, peak-solar, thermal,
  and mission-duration truth data used by `roverdevkit/validation/rover_comparison.py`.
  Source details live in each row's `citation` and `notes`.
- `roverdevkit/validation/rover_registry.py` (code, not data) — executable
  design vectors + scenarios + thermal/panel architecture consumed by the
  evaluator, rediscovery, surrogate sanity check, and webapp.

## Other files
- `soil_simulants.csv` — Bekker parameters (n, k_c, k_phi, cohesion,
  friction angle) for common lunar soil simulants: FJS-1, JSC-1A, GRC-1,
  plus Apollo regolith estimates.
- `validation/` — single-wheel testbed data digitized from published
  papers (Ding 2011, Iizuka & Kubota 2011, Wong's datasets). Used as
  held-out data to sanity-check the evaluator — never used for training.
- `analytical/` — generated LHS samples from the analytical evaluator.
  Git-ignored except for schema documentation.

## Citation discipline

Every curated data row must carry a citation or provenance note. Prefer the
canonical `rovers.yaml` per-field provenance for published rover facts; use the
dedicated `citation` column where present; otherwise document sources and
imputations in `notes` / `imputation_notes`. If you can't cite it, don't fit
on it.
