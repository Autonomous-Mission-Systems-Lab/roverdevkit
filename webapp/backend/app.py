"""FastAPI application factory.

The ``create_app`` function is the single entry point used by the
production server (``main.py``) and the test suite. Building the app
inside a factory rather than a module-level ``app = FastAPI()`` makes
two things easier:

1. **Per-test isolation.** Each test can build its own app with a
   patched cache / config, avoiding cross-test bleed.
2. **Future config injection.** When deployment grows env-driven
   feature flags (e.g. enable SSE optimisation, mount alternate model
   paths), they all funnel through ``get_settings()`` and the factory.

Static frontend mount (Docker / hosted deploy)
----------------------------------------------
When the env var ``ROVERDEVKIT_STATIC_DIR`` is set and points at an
existing directory containing the production frontend bundle
(``index.html`` + Vite-emitted assets), the factory mounts that
directory at the application root with HTML history-mode fallback.
This lets a single uvicorn process serve both the API (``/api`` prefix
not used today) and the React SPA -- the deploy story for Docker /
HF Spaces / Fly.io. Local dev leaves the var unset; the Vite dev
server handles the SPA on :5173 and proxies API calls to :8000.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.backend.config import Settings, get_settings
from webapp.backend.routes import evaluate as evaluate_routes
from webapp.backend.routes import health as health_routes
from webapp.backend.routes import optimize as optimize_routes
from webapp.backend.routes import predict as predict_routes
from webapp.backend.routes import registry as registry_routes
from webapp.backend.routes import scenarios as scenarios_routes
from webapp.backend.routes import shap as shap_routes
from webapp.backend.routes import sweep as sweep_routes

logger = logging.getLogger(__name__)


API_TITLE = "roverdevkit tradespace API"
API_DESCRIPTION = (
    "Backend for the webapp interactive tradespace exploration tool. "
    "Wraps the corrected mission evaluator and the quantile-calibration "
    "quantile-XGBoost surrogate."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the FastAPI app.

    Parameters
    ----------
    settings
        Optional override; falls back to :func:`get_settings`. Tests
        pass a custom ``Settings`` to point at fixture artifacts; the
        server entry point uses the env-driven default.
    """
    cfg = settings or get_settings()
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health_routes.router)
    app.include_router(scenarios_routes.router)
    app.include_router(registry_routes.router)
    app.include_router(predict_routes.router)
    app.include_router(evaluate_routes.router)
    app.include_router(sweep_routes.router)
    app.include_router(optimize_routes.router)
    app.include_router(shap_routes.router)

    _maybe_mount_frontend(app)

    logger.info(
        "FastAPI app built (artifacts_present=%s, dataset_version=%s)",
        cfg.artifacts_present,
        cfg.dataset_version,
    )
    return app


def _maybe_mount_frontend(app: FastAPI) -> None:
    """Mount the production frontend bundle at ``/`` when configured.

    Activated by setting ``ROVERDEVKIT_STATIC_DIR`` to a directory
    that contains a built Vite bundle (``index.html`` + ``assets/``).
    Wires up:

    - ``/assets/*`` for the hashed JS / CSS / image bundles served
      directly by :class:`StaticFiles` with caching headers,
    - ``/`` for the SPA entry point, plus a catch-all rewrite that
      hands any non-API path back to ``index.html`` so the React
      router's history-mode URLs survive a hard refresh.

    API routes are registered first (above) so they always win over
    the catch-all. We only register fallthrough handlers when the
    static dir actually exists; tests and `make webapp-dev` leave
    the env var unset and skip this branch.
    """
    static_dir_env = os.environ.get("ROVERDEVKIT_STATIC_DIR")
    if not static_dir_env:
        return
    static_dir = Path(static_dir_env).expanduser().resolve()
    index_path = static_dir / "index.html"
    if not index_path.is_file():
        logger.warning(
            "ROVERDEVKIT_STATIC_DIR=%s but %s is missing; skipping frontend mount",
            static_dir,
            index_path,
        )
        return

    assets_dir = static_dir / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="frontend-assets",
        )

    @app.get("/", include_in_schema=False)
    def _serve_index_root() -> FileResponse:
        return FileResponse(index_path)

    @app.get("/{path:path}", include_in_schema=False)
    def _spa_fallback(path: str) -> FileResponse:
        candidate = (static_dir / path).resolve()
        # Reject path-traversal escapes; serve the file directly when
        # it exists (e.g. /favicon.ico, /robots.txt). Anything else
        # is treated as a SPA route and re-served as index.html so
        # the React router can pick it up on the client.
        try:
            candidate.relative_to(static_dir)
        except ValueError:
            return FileResponse(index_path)
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_path)

    logger.info("mounted frontend static bundle from %s", static_dir)
