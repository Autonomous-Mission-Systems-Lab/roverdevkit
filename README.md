# RoverDevKit

**Design-space exploration for lunar micro-rovers.**

RoverDevKit is an open-source research toolkit for early-stage lunar
micro-rover design. It combines a physics-based mission evaluator,
calibrated surrogate models, and an
interactive web app for exploring mobility, power, mass, and mission tradeoffs.

The tool is aimed at conceptual design questions such as:

- How do wheel geometry, solar area, battery capacity, drivetrain torque, and
  chassis mass trade against each other for a given lunar mission, given a fixed
  scientific payload mass and power requirement?
- Which candidate designs are Pareto-efficient for range, slope capability,
  energy margin, and total mass?
- For a given candidate design, what features contribute most to its predicted values
  for range, mass, and slope capability?
- How do candidate designs compare with historical and published lunar micro-rover designs?

## What RoverDevKit Provides

- **Mission evaluator:** deterministic end-to-end evaluation of a rover design
  in a lunar mission scenario, returning range, energy margin, slope capability,
  total mass, thermal survival, and drivetrain stall diagnostics. Scientific
  payload (instrument mass and continuous power) is treated as a mission
  *requirement* set alongside the scenario, not as a design variable the
  engineer trades; the chassis-mass input is the structural chassis only. It
  couples several physics sub-models:
  - **Terramechanics:** Bekker-Wong wheel-soil mechanics with an
    engaged-grouser shear-thrust term (Iizuka & Kubota 2011).
  - **Drivetrain:** motor torque, stall, and cruise-speed limits.
  - **Power:** solar generation, battery storage, and thermal survival.
  - **Mass:** parametric mass-estimating relationships for the rover subsystems.
- **Surrogate + uncertainty:** quantile XGBoost models that provide
  calibrated 90% prediction intervals around the evaluator's
  deterministic output and power TreeSHAP feature attributions on the
  Explain Design tab. The analytical physics evaluator is the
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

### Run with Docker (one command)

The fastest way to try the tool is the multi-stage container — it bakes
in the surrogate and the canonical Pareto fronts, then serves backend + SPA
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
print(metrics.obstacle_capability_m)
print(metrics.obstacle_margin_m)
```

The design vector trades `mobility_architecture` (`rigid_4wheel` vs.
`rocker_bogie_6wheel`) alongside wheel geometry. Missions may specify
`required_obstacle_height_m` on the scenario; rocker-bogie carries a
suspension mass penalty but negotiates taller obstacles via the
architecture proxy in `roverdevkit/architecture.py`. Run
`make architecture-crossover` to regenerate the manuscript crossover
figure from evaluator-backed NSGA-II sweeps.

## Installation Notes

The recommended setup is **Miniforge/Mambaforge + conda**.

```bash
mamba env create -f environment.yml
conda activate roverdevkit
pip install -e ".[dev,webapp]"
pytest -q
```

The conda environment uses Python 3.12.

The web app expects the trained surrogate bundle at
`models/surrogate_v9/quantile_bundles.joblib` or supplied through the
`ROVERDEVKIT_QUANTILE_BUNDLES` environment variable. The evaluator routes
(current-design evaluation, parametric sweeps, NSGA-II in the Optimize
Design tab) work without it; the surrogate is needed only for the 90%
prediction intervals on the Current Design tab and the SHAP attributions
on the Explain Design tab.

## Repository Layout

```text
roverdevkit/
├── models/               # Shipped trained surrogate bundles (runtime artifacts)
├── data/                 # Soil parameters, analytical data, published rover data
├── roverdevkit/          # Python package
│   ├── drivetrain/       # Motor and cruise-speed helpers
│   ├── mass/             # Parametric mass-estimating relationships
│   ├── mission/          # Scenarios, evaluator, traverse simulator
│   ├── power/            # Solar, battery, thermal models
│   ├── surrogate/        # Features, datasets, baselines, uncertainty
│   ├── terramechanics/   # Bekker-Wong wheel-soil mechanics
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

- **Layer 1 — surrogate vs analytical evaluator.** R² ≥ 0.92 on every
  primary regression target on the canonical 10 % LHS test split
  (range 0.922, energy margin 0.968, slope 0.979, total mass 0.999);
  AUC 0.993 on the feasibility classifier; ≈83–96 % empirical coverage
  on the calibrated 90 % prediction intervals. The training corpus
  includes scientific payload (mass and power) as two explicit
  mission-requirement inputs sampled over [0, 30], and six-wheel
  samples carry the updated rocker-bogie suspension mass in the
  evaluator labels. Training metrics live under `reports/surrogate_v9/`;
  the shipped runtime bundle is `models/surrogate_v9/quantile_bundles.joblib`
  (published automatically by `scripts/calibrate_intervals.py` on a full run).
- **Layer 3 — sub-model checks against published references.**
  Parametric tolerance grid in
  `data/validation/wong_layer3_reference.csv` exercising a Wong (2008)
  §4.2 worked-example fixture, Pragyan- and Yutu-2-class operating
  points on Apollo regolith, and Iizuka & Kubota (2011) grouser-thrust
  limit cases; assertions in
  `tests/test_terramechanics.py::test_layer3_published_reference_grid`.
- **Layer 5 — rover rediscovery.** Leakage-free NSGA-II rediscovery
  sweep over the registry (no model coefficient is fit to the
  registry, so this is a rediscovery comparison rather than
  leave-one-out cross-validation) with a class-neutral
  operational-duty-cycle anchor, scenario-driven panel-tilt
  orientation, and each rover's published scientific payload injected
  as a mission requirement (schema v9), so the optimiser carries the
  same instrument mass and power the real rover flew. Across the five
  in-scope (< 50 kg) micro-rovers the median evaluator-backed
  design-space distance is **0.49** (≈ 41 % of the ≈1.20 random-pair
  baseline in the 9-D unit cube — the mean pairwise separation between
  random designs). Per-rover distances: Tenacious 0.12, Rashid-1 0.34,
  CADRE-unit 0.49, Pragyan 0.52, MoonRanger 0.59 — all
  five between 10 % and 49 % of the random-pair baseline. Yutu-2
  (~135 kg) sits above the micro-rover scope and is reported separately
  at 1.01 (≈ the random baseline): the optimiser does not spuriously
  place an out-of-class rover near its front. Distances are reported
  relative to the random-pair baseline rather than any fixed pass/fail
  cutoff. (CADRE-unit at 2 kg sits below the ~5–50 kg mass-model
  calibration band and is flagged out-of-regime — the evaluator still
  rediscovers it; the surrogate backend cannot reach its mass.)
  Most published designs are Pareto-dominated
  by the optimiser front under the class-neutral anchor; this reflects
  design constraints the conceptual model does not carry (radiation,
  deployability, integration and redundancy margins), not a claim that
  flight designs are sub-optimal. Full results and per-rover JSON
  artefacts are in
  `reports/rediscovery_loo_evaluator/rediscovery_loo_report.md`.

## Reproducibility

Artifacts that back the webapp and the figures in the manuscript
live under `reports/` and are regenerated by these commands:

| Command | Output | Purpose |
| --- | --- | --- |
| `make pareto-fronts` | `reports/pareto_fronts/` | Evaluator-driven NSGA-II Pareto fronts for the four canonical scenarios. Every point is corrected-evaluator output (no surrogate involvement). Reference artifacts for the manuscript figures; the webapp runs its own live NSGA-II on demand. |
| `make optimizer-robustness` | `reports/optimizer_robustness/` | Multi-seed, multi-budget evaluator-backed NSGA-II sweep used to check Pareto-front repeatability and convergence for the manuscript. |
| `make architecture-crossover` | `reports/architecture_obstacle_crossover/` | Sweeps `required_obstacle_height_m` across the four canonical scenarios and records when rocker-bogie six-wheel layouts enter the Pareto set. |
| `python scripts/run_rediscovery_loo.py --all --n-seeds 5` | `reports/rediscovery_loo_evaluator/` | Rediscovery sweep over the six-rover registry (the headline validation result). The `--n-seeds 5` ensemble is the paper configuration; the script default (`--n-seeds 1`) runs a faster single-seed sweep whose per-rover distances are noisier. |
| `make figures` | `paper/figures/` | Renders every manuscript figure (Pareto fronts, rediscovery distance + overlay, flown-rover peak-solar, terramechanics validation, terramechanics sensitivity, architecture crossover) from the artifacts above via the `scripts/make_*_figure.py` regenerators. Runs in well under a minute. |

`make pareto-fronts` runs end-to-end in ~4 minutes on a laptop with the
`roverdevkit` conda environment activated; the rediscovery sweep takes
~10 minutes. With both in place, `make figures`
regenerates the figures into `paper/figures/`.

## Research Background

RoverDevKit is developed by the Autonomous Mission Systems Lab at Duke University.
The project focuses on open, reproducible design-space exploration for lunar
micro-rovers in the pre-Phase A / conceptual-design regime.

## Use of AI Tools

Portions of this repository's code and documentation were developed with the
assistance of an AI coding tool — Anthropic's Claude (Opus 4.8) via the Cursor
IDE — for code generation, refactoring, and documentation drafting. All
AI-assisted contributions were reviewed, tested, and validated by the authors,
who take full responsibility for the correctness of the physics models,
methods, results, and claims presented here. This disclosure is provided in the
interest of research transparency and reproducibility.

## Citation

A paper is in preparation. Citation information will be added on submission.

## License

MIT — see [`LICENSE`](LICENSE).
