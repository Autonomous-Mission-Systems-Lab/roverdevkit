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

- **Current Design** evaluates the active rover design under the mission inputs
  (scenario, scientific-payload mass and power, operational duty cycle). Payload
  is a mission requirement set at the top of the panel, not a design variable.
- **Parametric Sweep** explores one- and two-variable design sensitivities.
- **Optimize Design** runs NSGA-II searches and visualizes completed Pareto
  fronts.
- **Explain Design** shows SHAP-style feature attributions for the active
  design and selected target.
- **Rediscovery Validation** overlays each registry rover's published
  design point on the optimizer's leave-one-out NSGA-II Pareto front
  under its class-generic `*_micro` scenario. Backed by the
  precomputed artifacts under `reports/rediscovery_loo_evaluator/`
  (every Pareto point is corrected-evaluator output) and
  `reports/rediscovery_loo_surrogate_v9/` (faster surrogate-backed
  reference). This is the live, in-app rendering of the Layer-5
  rover-rediscovery validation summarised in the top-level
  `README.md`.

## Canonical Pareto Fronts

The Optimize Design tab always runs NSGA-II live against the corrected
physics evaluator. There is no in-app "load reference" affordance because
the live job completes in ~30–60 seconds at the default budget and produces
evaluator-truth points anyway.

The repo still ships a precomputed reference set under
`reports/pareto_fronts/` (one CSV + metadata JSON per canonical scenario,
plus a top-level `manifest.json`). These are reference artifacts for
documentation figures and notebook fixtures; the evaluator is
deterministic and each file is small (~14 kB), so they are committed
to the repo for reproducibility.

Regenerate them whenever scenario configs or the wheel-level correction
artifact change:

```bash
make pareto-fronts
```

Pass extra arguments through `SCRIPT_ARGS`:

```bash
make pareto-fronts SCRIPT_ARGS="--population-size 80 --generations 80"
```

The default settings (50-point population, 60 generations, BW + SCM
evaluator) complete all four canonical scenarios in about 4 minutes on a
laptop.

## Docker / Hosted Deploy

A multi-stage [`webapp/Dockerfile`](Dockerfile) builds the React frontend
with Node 20 LTS, then installs the Python package + the `[webapp]`
extras on top of `python:3.12-slim` and bakes in:

- the corrected mission evaluator and the wheel-level SCM correction
  (`data/scm/correction_v1.joblib`),
- the v9 quantile-XGB surrogate bundles
  (`reports/surrogate_v9/quantile_bundles.joblib`),
- the canonical Pareto fronts (`reports/pareto_fronts/`),
- the rediscovery LOO artifacts
  (`reports/rediscovery_loo_evaluator/`,
  `reports/rediscovery_loo_surrogate_v9/`),
- the built React frontend at `/app/static`.

The runtime image runs a single `uvicorn` process that serves the
FastAPI backend at `/healthz`, `/predict`, `/evaluate`, `/sweep`,
`/optimize`, `/shap`, `/registry`, `/scenarios`, and
`/validate/rediscovery`, and serves the React SPA off
`ROVERDEVKIT_STATIC_DIR=/app/static` with a history-mode catch-all so
deep links survive a hard refresh.

PyChrono is **not** installed in the image. The runtime evaluator
loads the wheel-level SCM correction from a precomputed `joblib`
blob; regenerating that blob requires the conda PyChrono build and is
done outside Docker via `scripts/run_scm_sweep.py`.

### Local boot via Docker Compose

```bash
docker compose -f webapp/docker-compose.yml up --build
```

Open <http://localhost:8000>. First boot takes 2-3 min for the
`pip install` layer; subsequent rebuilds reuse the wheel cache and
finish in ~30 s for a code-only edit.

### Direct `docker build` (e.g. for CI or a one-off image push)

```bash
# Build context must be the repo root so the Dockerfile can reach
# pyproject.toml, roverdevkit/, data/, reports/, and webapp/.
docker build -f webapp/Dockerfile -t roverdevkit/webapp:dev .
docker run --rm -p 8000:8000 roverdevkit/webapp:dev
```

### Hosted-demo readiness checklist

The image is intentionally hosting-platform agnostic. To stand it up
on Hugging Face Spaces, Fly.io, Railway, or a Duke container, walk
through this checklist:

- [ ] `docker build -f webapp/Dockerfile -t roverdevkit/webapp:dev .`
      succeeds locally; record the resulting image size (~700 MB
      compressed for the v9 + rediscovery + correction bundle).
- [ ] `docker run --rm -p 8000:8000 roverdevkit/webapp:dev` boots
      cleanly; `curl localhost:8000/healthz` returns
      `{"status":"ok","surrogate_loaded":true,...}`.
- [ ] Set `ROVERDEVKIT_CORS_ORIGINS` to the platform's hosted origin
      (e.g. `https://huggingface.co,https://<user>-<space>.hf.space`).
      Defaults to `http://localhost:5173` (Vite dev server) which
      will block browser calls in prod.
- [ ] If the platform fronts the container with TLS, leave the
      Dockerfile's `--proxy-headers --forwarded-allow-ips=*` flags
      in place so client IP / scheme propagate from the edge.
- [ ] Confirm the image runs as `roverdevkit` (UID 1000) — Fly.io,
      HF Spaces, and most K8s pod-security policies require non-root.
- [ ] If the deploy uses a persistent volume to mount alternate
      surrogate / correction artefacts, point the volume at
      `/app/reports/surrogate_v9/quantile_bundles.joblib` (or
      override via `ROVERDEVKIT_QUANTILE_BUNDLES`); see
      `webapp/backend/config.py` for the full env-var surface.
- [ ] (HF Spaces) drop a `Dockerfile` symlink or a one-line
      `Spaces config: docker` block at the repo root that points at
      `webapp/Dockerfile`.
- [ ] Smoke-test `GET /validate/rediscovery` and
      `GET /validate/rediscovery/pragyan` against the hosted URL —
      these touch the rediscovery JSON tree and confirm the
      committed artefacts shipped intact.

### Image size and build context

`.dockerignore` at the repo root excludes the LHS training corpora
(`data/analytical/`, `data/scm/runs_*.parquet`), the training-time
reports (`reports/baselines_*`, `reports/tuned_*`,
`reports/validation_*`, `reports/intervals_*`) and any superseded
surrogate versions (`reports/surrogate_v7_1/`, `reports/surrogate_v8/`),
and Node / Python build caches. Only the runtime artefacts the backend
actually loads — `reports/surrogate_v9/quantile_bundles.joblib`, the
Pareto fronts, and the rediscovery LOO trees — are baked into the
image.

## Tests

```bash
pytest webapp/backend/tests -q
cd webapp/frontend && npm run lint && npm run build
```

The top-level helper runs the same checks:

```bash
make webapp-test
```
