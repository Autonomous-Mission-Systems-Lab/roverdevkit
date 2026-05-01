# RoverDevKit

**Design-space exploration for lunar micro-rovers.**

RoverDevKit is an open-source research toolkit for early-stage lunar
micro-rover design. It combines a physics-based mission evaluator, a
multi-fidelity wheel-soil correction layer, calibrated surrogate models, and an
interactive web app for exploring mobility, power, mass, and mission tradeoffs.

The tool is aimed at conceptual design questions such as:

- How do wheel geometry, solar area, battery capacity, drivetrain torque, and
  chassis mass trade against each other for a given lunar mission?
- Which candidate designs are Pareto-efficient for range, slope capability,
  energy margin, and total mass?
- Why does the model prefer one design over another?
- How do candidate designs compare with published lunar micro-rovers such as
  Pragyan, Yutu-2, MoonRanger, and Rashid-1?

## What RoverDevKit Provides

- **Mission evaluator:** deterministic end-to-end evaluation of a rover design
  in a lunar mission scenario, returning range, energy margin, slope capability,
  total mass, thermal survival, and drivetrain stall diagnostics.
- **Terramechanics model:** Bekker-Wong wheel-soil mechanics with an
  engaged-grouser shear-thrust term and an optional learned correction toward
  PyChrono SCM single-wheel simulations.
- **Surrogate + uncertainty:** quantile XGBoost models that provide calibrated
  90% prediction intervals and make large design searches fast.
- **Interactive web app:** current-design analysis, parametric sweeps,
  multi-objective optimization, Pareto-front visualization, and SHAP-style
  explanations for the active design.
- **Validation data and registry:** published lunar micro-rover design points,
  mission scenarios, soil simulants, and validation harnesses.

## Web App

The browser app is the easiest way to use RoverDevKit.

Tabs in the app:

- **Current Design:** edit a rover design, choose a mission scenario, and view
  predicted performance with calibrated 90% intervals.
- **Parametric Sweep:** vary one or two design variables and visualize how an
  output changes across the design space.
- **Optimize Design:** run multi-objective NSGA-II searches and inspect the
  resulting tradeoff front.
- **Explain Design:** show SHAP-style feature attributions for the active
  design and selected output.

Run the app locally:

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

The web app expects trained surrogate artifacts to be present at the configured
paths under `reports/` or supplied through environment variables such as
`ROVERDEVKIT_QUANTILE_BUNDLES`. If the artifact is missing, deterministic
evaluator routes still work, while surrogate-backed prediction intervals and
optimization may be unavailable until the artifact is generated or provided.

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

## Research Background

RoverDevKit is developed by the Space Systems Autonomy Lab at Duke University.
The project focuses on open, reproducible design-space exploration for lunar
micro-rovers in the pre-Phase A / conceptual-design regime.

For research motivation and model assumptions, see [`project_brief.md`](project_brief.md).

## Citation

A paper is in preparation. Citation information will be added on submission.

## License

MIT — see [`LICENSE`](LICENSE).
