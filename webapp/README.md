# RoverDevKit Web App

The web app provides an interactive interface for RoverDevKit's mission
evaluator, surrogate predictions, parametric sweeps, multi-objective design
optimization, and SHAP-style design explanations.

```
webapp/
├── backend/        FastAPI app over the roverdevkit Python package
└── frontend/       React + Vite + TypeScript single-page app
```

## Backend

The backend is a thin FastAPI layer over the Python evaluator and trained
surrogate artifacts. The API keeps the browser UI aligned with the same models
used by scripts and notebooks.

Common routes:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/healthz` | Liveness and artifact-presence probe. |
| GET | `/version` | Dataset, surrogate, and git version metadata. |
| GET | `/scenarios` | List bundled mission scenarios. |
| GET | `/scenarios/{name}` | Return one scenario and its nominal soil parameters. |
| GET | `/registry` | Return published rover registry entries. |
| POST | `/predict` | Surrogate median and 90% interval for one design. |
| POST | `/evaluate` | Physics evaluator output for one design. |
| POST | `/sweep` | One- or two-dimensional parametric sweep. |
| POST | `/optimize` | Start an NSGA-II multi-objective optimization job. |
| GET | `/optimize/{job_id}/stream` | Stream optimization progress with server-sent events. |
| GET | `/optimize/{job_id}/result` | Fetch a completed optimization result. |
| POST | `/shap/explain` | Explain one current design prediction. |

## Run Locally

From the repository root, with the `roverdevkit` conda environment activated:

```bash
conda activate roverdevkit
pip install -e ".[webapp]"
uvicorn webapp.backend.main:app --reload --port 8000
```

OpenAPI docs are available at <http://localhost:8000/docs>.

## Frontend

The frontend is a Vite + React + TypeScript app. Its typed fetch client lives in
`src/lib/api.ts`, and the dev server proxies backend routes to
`http://localhost:8000`.

```bash
cd webapp/frontend
npm install
npm run dev
npm run build
npm run lint
```

Run both servers together from the repository root:

```bash
make webapp-dev
```

Open <http://localhost:5173> after the frontend server starts.

## UI Sections

- **Current Design** evaluates the active rover design and mission scenario.
- **Parametric Sweep** explores one- and two-variable design sensitivities.
- **Optimize Design** runs NSGA-II searches and visualizes completed Pareto
  fronts.
- **Explain Design** shows SHAP-style feature attributions for the active
  design and selected target.

## Tests

```bash
pytest webapp/backend/tests -q
cd webapp/frontend && npm run lint && npm run build
```

The top-level helper runs the same checks:

```bash
make webapp-test
```
