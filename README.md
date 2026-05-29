# RoverDevKit

**Design-space exploration for lunar micro-rovers.**

RoverDevKit is an open-source research toolkit for early-stage lunar
micro-rover design. It combines a physics-based mission evaluator, a
multi-fidelity wheel-soil correction layer, calibrated surrogate models, and an
interactive web app for exploring mobility, power, mass, and mission tradeoffs.

The tool is aimed at conceptual design questions such as:

- How do wheel geometry, solar area, battery capacity, drivetrain torque, and
  chassis mass trade against each other for a given lunar mission, given a fixed
  scientific-payload mass and power requirement?
- Which candidate designs are Pareto-efficient for range, slope capability,
  energy margin, and total mass?
- Why does the model prefer one design over another?
- How do candidate designs compare with published lunar micro-rovers such as
  Pragyan, Yutu-2, MoonRanger, and Rashid-1?

## What RoverDevKit Provides

- **Mission evaluator:** deterministic end-to-end evaluation of a rover design
  in a lunar mission scenario, returning range, energy margin, slope capability,
  total mass, thermal survival, and drivetrain stall diagnostics. Scientific
  payload (instrument mass and continuous power) is treated as a mission
  *requirement* set alongside the scenario, not as a design variable the
  engineer trades; the chassis-mass input is the structural chassis only.
- **Terramechanics model:** Bekker-Wong wheel-soil mechanics with an
  engaged-grouser shear-thrust term and an optional learned correction toward
  PyChrono SCM single-wheel simulations.
- **Surrogate + uncertainty:** quantile XGBoost models that provide
  calibrated 90% prediction intervals around the evaluator's
  deterministic output and power TreeSHAP feature attributions on the
  Explain Design tab. The corrected physics evaluator is the
  reference output and the objective evaluated inside NSGA-II; the
  surrogate sits on top of it as the uncertainty-quantification and
  explainability layer.
- **Interactive web app:** current-design analysis, parametric sweeps,
  multi-objective optimization, Pareto-front visualization, and SHAP-style
  explanations for the active design.
- **Validation data and registry:** published lunar micro-rover design points,
  mission scenarios, soil simulants, and validation harnesses.

## Web App

The browser app is the easiest way to use RoverDevKit.

Tabs in the app:

- **Current Design:** set the mission inputs (scenario, scientific-payload mass
  and power, operational duty cycle), edit a rover design, and view predicted
  performance with calibrated 90% intervals.
- **Parametric Sweep:** vary one or two design variables and visualize how an
  output changes across the design space.
- **Optimize Design:** run multi-objective NSGA-II searches and inspect the
  resulting tradeoff front.
- **Explain Design:** show SHAP-style feature attributions for the active
  design and selected output.
- **Rediscovery Validation:** every registry rover's published design
  point overlaid on the optimizer's leave-one-out Pareto front under
  its class-generic mission scenario, used to check that the optimizer
  recovers real flown / design-target rovers from public specs.

### Run with Docker (one command)

The fastest way to try the tool is the multi-stage container — it bakes
in the surrogate, the SCM correction artifact, the canonical Pareto
fronts, and the rediscovery validation data, then serves backend + SPA
from a single `uvicorn` process.

```bash
docker compose -f webapp/docker-compose.yml up --build
```

Open <http://localhost:8000>. See [`webapp/README.md`](webapp/README.md)
for the hosted-demo readiness checklist (Hugging Face Spaces / Fly.io /
Duke container) and the env-var surface.

### Run from a Python environment (live reload)

```bash
# 1. Create and activate the Python environment.
mamba env create -f environment.yml
conda activate roverdevkit
pip install -e ".[dev,webapp]"

# 2. Install frontend dependencies.
cd webapp/frontend
npm install
cd ../..

# 3. Start backend and frontend together.
make webapp-dev
```

Open <http://localhost:5173>.

The FastAPI backend runs on <http://localhost:8000>, with OpenAPI docs at
<http://localhost:8000/docs>.

## Python Quickstart

You can also use the evaluator directly from Python:

```python
from roverdevkit.mission.evaluator import evaluate
from roverdevkit.mission.scenarios import load_scenario
from roverdevkit.schema import DesignVector

design = DesignVector(
    wheel_radius_m=0.10,
    wheel_width_m=0.10,
    grouser_height_m=0.012,
    grouser_count=14,
    n_wheels=6,
    chassis_mass_kg=20.0,
    wheelbase_m=0.60,
    solar_area_m2=0.50,
    battery_capacity_wh=100.0,
    avionics_power_w=15.0,
    peak_wheel_torque_nm=1.50,
)

scenario = load_scenario("equatorial_mare_traverse")
metrics = evaluate(design, scenario)

print(metrics.range_km)
print(metrics.energy_margin_raw_pct)
print(metrics.slope_capability_deg)
print(metrics.total_mass_kg)
```

## Installation Notes

The recommended setup is **Miniforge/Mambaforge + conda** because PyChrono is
distributed through conda channels.

```bash
mamba env create -f environment.yml
conda activate roverdevkit
pip install -e ".[dev,webapp]"
pytest -q
```

The conda environment uses Python 3.12. If PyChrono is not available on your
platform, most analytical evaluator, surrogate, and web-app workflows can still
run without invoking SCM directly.

The web app expects the trained surrogate bundle to be present at the
configured path under `reports/` or supplied through the
`ROVERDEVKIT_QUANTILE_BUNDLES` environment variable. The evaluator routes
(current-design evaluation, parametric sweeps, NSGA-II in the Optimize
Design tab) work without it; the surrogate is needed only for the 90%
prediction intervals on the Current Design tab and the SHAP attributions
on the Explain Design tab.

## Repository Layout

```text
roverdevkit/
├── data/                 # Soil parameters, analytical data, published rover data
├── roverdevkit/          # Python package
│   ├── drivetrain/       # Motor and cruise-speed helpers
│   ├── mass/             # Parametric mass-estimating relationships
│   ├── mission/          # Scenarios, evaluator, traverse simulator
│   ├── power/            # Solar, battery, thermal models
│   ├── surrogate/        # Features, datasets, baselines, uncertainty
│   ├── terramechanics/   # Bekker-Wong, SCM wrapper, correction model
│   ├── tradespace/       # Sweeps, NSGA-II optimization, design explanations
│   └── validation/       # Rover registry and validation helpers
├── scripts/              # Dataset, tuning, validation, and report scripts
├── tests/                # Python test suite
└── webapp/               # FastAPI backend and React frontend
```

## Development

```bash
# Python tests
conda run -n roverdevkit pytest

# Backend web tests
conda run -n roverdevkit pytest webapp/backend/tests

# Frontend checks
cd webapp/frontend
npm run lint
npm run build
```

Useful app commands:

```bash
make webapp-dev       # backend on :8000, frontend on :5173
make webapp-backend   # backend only
make webapp-frontend  # frontend only
make webapp-test      # backend tests + frontend lint/build
```

## Validation snapshot

The toolkit ships with a layered validation chain. Summary numbers
(full reports under `reports/`):

- **Layer 1 — surrogate vs corrected evaluator.** R² ≥ 0.95 on every
  primary regression target on the canonical 10 % LHS test split
  (range 0.952, energy margin 0.983, slope 0.977, total mass 0.999);
  AUC 0.992 on the feasibility classifier; ≈89–96 % empirical coverage
  on the calibrated 90 % prediction intervals. The schema-v9 corpus
  adds scientific payload (mass and power) as two explicit
  mission-requirement inputs sampled over [0, 30], which adds a little
  variance to the range and slope heads relative to v8. See
  `reports/surrogate_v9/`.
- **Layer 2 — corrected evaluator vs PyChrono SCM.** On a held-out
  LHS comparison sample, the corrected evaluator (Bekker-Wong +
  wheel-level SCM correction) reaches ≥ 99 % feasibility-flip
  agreement with PyChrono SCM, ~10× lower continuous-metric error
  than uncorrected Bekker-Wong, and ~100× faster wall-clock than
  PyChrono SCM run end-to-end. PyChrono SCM is itself a semi-empirical
  contact model (Tasora et al. 2023) and is treated here as a
  high-fidelity comparator, not as ground truth.
- **Layer 3 — sub-model checks against published references.**
  Parametric tolerance grid in
  `data/validation/wong_layer3_reference.csv` exercising a Wong (2008)
  §4.2 worked-example fixture, Pragyan- and Yutu-2-class operating
  points on Apollo regolith, and Iizuka & Kubota (2011) grouser-thrust
  limit cases; assertions in
  `tests/test_terramechanics.py::test_layer3_published_reference_grid`.
- **Layer 5 — rover rediscovery, leave-one-out.** Six-rover
  leave-one-out NSGA-II sweep with a class-neutral δ_ops anchor,
  scenario-driven panel-tilt orientation, and each rover's published
  scientific payload injected as a mission requirement (schema v9), so
  the optimiser carries the same instrument mass and power the real
  rover flew. Median evaluator-backed design-space distance **0.473**
  (≈ 39 % of the 1.22 random-pair baseline in the 9-D unit cube). Per-
  rover distances: CADRE-unit 0.291, Tenacious 0.306, Rashid-1 0.363,
  MoonRanger 0.583, Pragyan 0.626, Yutu-2 1.146 — i.e. the optimiser
  places **CADRE-unit and Tenacious inside a 0.3 design-distance band**
  of their published designs and the remaining rovers in their broader
  neighbourhood. Promoting payload from a per-rover chassis-mass
  convention to an explicit requirement *tightened* the median distance
  (0.502 → 0.473) and the median per-rover error (54 % → 50 %): forcing
  candidates to carry the real payload mass removes the previously
  spurious ultra-light Pareto points. Five of the six published designs
  remain Pareto-dominated under the class-neutral δ_ops anchor: at
  matched mass budgets, lighter sun-tracking designs out-perform the
  published configurations once panel orientation is matched. Full
  results and per-rover JSON artefacts are in
  `reports/rediscovery_loo_evaluator/rediscovery_loo_report.md` (the
  four-configuration narrative in `reports/rediscovery_loo_comparison.md`
  carries the supporting panel-tilt and backend-comparison analysis).

## Reproducibility

Artifacts that back the webapp and the figures in the manuscript
live under `reports/` and are regenerated by these commands:

| Command | Output | Purpose |
| --- | --- | --- |
| `make pareto-fronts` | `reports/pareto_fronts/` | Evaluator-driven NSGA-II Pareto fronts for the four canonical scenarios. Every point is corrected-evaluator output (no surrogate involvement). Reference artifacts for the manuscript figures; the webapp runs its own live NSGA-II on demand. |
| `python scripts/run_rediscovery_loo.py --all` | `reports/rediscovery_loo_evaluator/` | Leave-one-out rediscovery sweep over the six-rover registry (the headline validation result). |
| `notebooks/paper_figures.ipynb` | `reports/figures/` | Single top-to-bottom notebook that renders every manuscript figure (Pareto fronts, rediscovery overlay, flown-rover validation) from the artifacts above. Runs in well under a minute. |

`make pareto-fronts` runs end-to-end in ~4 minutes on a laptop with the
`roverdevkit` conda environment activated; the rediscovery sweep takes
~10 minutes. With both in place, `notebooks/paper_figures.ipynb`
regenerates the figures into `reports/figures/`.

## Research Background

RoverDevKit is developed by the Autonomous Mission Systems Lab at Duke University.
The project focuses on open, reproducible design-space exploration for lunar
micro-rovers in the pre-Phase A / conceptual-design regime.

## Citation

A paper is in preparation. Citation information will be added on submission.

## License

MIT — see [`LICENSE`](LICENSE).
