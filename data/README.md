# Data

Small, curated datasets and citations live here. Large generated datasets
(LHS samples) are git-ignored — see `.gitignore`.

## Files

- `published_rovers.csv` — specs for ~10 real lunar/planetary micro-rovers
  (Rashid, Pragyan, Yutu-1/2, CADRE, Sojourner, Lunokhod, etc.) with
  citations. Used to fit parametric mass-estimating relationships
  (`roverdevkit/mass/parametric_mers.py`) and for rover rediscovery
  validations.
- `soil_simulants.csv` — Bekker parameters (n, k_c, k_phi, cohesion,
  friction angle) for common lunar soil simulants: FJS-1, JSC-1A, GRC-1,
  plus Apollo regolith estimates.
- `validation/` — single-wheel testbed data digitized from published
  papers (Ding 2011, Iizuka & Kubota 2011, Wong's datasets). Used as
  held-out data to sanity-check the evaluator — never used for training.
- `analytical/` — generated LHS samples from the analytical evaluator.
  Git-ignored except for schema documentation.

## Citation discipline

Every row in `published_rovers.csv` and `soil_simulants.csv` must carry a
citation. If you can't cite it, don't fit on it.
