# RoverDevKit webapp — multi-stage Dockerfile
#
# Stage 1 (`frontend-build`) builds the Vite production bundle with
# Node 20 LTS. Stage 2 (`runtime`) installs the Python package + the
# `[webapp]` extras on top of `python:3.12-slim`, copies the package
# source / on-disk artifacts / built frontend bundle in, and runs
# uvicorn as a non-root user.
#
# Build context expectation: this file is invoked from the repo
# root so it can reach `pyproject.toml`, `roverdevkit/`, `data/`,
# `models/`, `reports/`, and `webapp/` in one COPY plane:
#
#     docker build -f webapp/Dockerfile -t roverdevkit/webapp:dev .
#
# The image bakes in:
#  - the analytical Bekker-Wong mission evaluator,
#  - the v9 quantile-XGB surrogate bundles
#    (`models/surrogate_v9/quantile_bundles.joblib`),
#  - the canonical Pareto fronts (`reports/pareto_fronts/`),
#  - the built React frontend (`/app/static/`).

# ---------------------------------------------------------------------------
# Stage 1: build the frontend bundle
# ---------------------------------------------------------------------------

FROM node:20-bookworm-slim AS frontend-build

WORKDIR /build

# Install dependencies first so the npm cache is reusable across edits
# of `webapp/frontend/src/`. Lockfile copy + `npm ci` gives a
# reproducible install.
COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY webapp/frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: runtime image
# ---------------------------------------------------------------------------

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    ROVERDEVKIT_STATIC_DIR=/app/static

# `libgomp1` is needed by xgboost on Linux for the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so a code-only edit doesn't bust
# the heavy ML wheel cache. Editable installs need the package
# source in the same layer; copy the build metadata here and the
# rest in the next layer.
COPY pyproject.toml README.md LICENSE ./
COPY roverdevkit/ ./roverdevkit/
RUN pip install --no-cache-dir ".[webapp]"

# Webapp backend + on-disk artifacts.
COPY webapp/backend/ ./webapp/backend/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/

# Built frontend bundle from stage 1 → mounted at /app/static via
# the ROVERDEVKIT_STATIC_DIR env var above.
COPY --from=frontend-build /build/dist/ ./static/

# Run as non-root to satisfy the standard hosting-platform contract
# (Fly.io, HF Spaces, K8s pod security policies, etc.). UID 1000 is
# arbitrary; pick whatever your hosting environment prefers.
RUN useradd --create-home --uid 1000 roverdevkit \
    && chown -R roverdevkit:roverdevkit /app
USER roverdevkit

EXPOSE 8000

# `--proxy-headers` lets the deployment reverse proxy (Fly's edge,
# HF Spaces' router, etc.) pass through the original client IP and
# scheme. `--forwarded-allow-ips='*'` is safe behind a trusted
# proxy and avoids 403s on the WebSocket / SSE probes some hosts
# use.
CMD ["uvicorn", "webapp.backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--forwarded-allow-ips=*"]
