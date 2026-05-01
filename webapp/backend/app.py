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
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webapp.backend.config import Settings, get_settings
from webapp.backend.routes import evaluate as evaluate_routes
from webapp.backend.routes import health as health_routes
from webapp.backend.routes import optimize as optimize_routes
from webapp.backend.routes import pareto as pareto_routes
from webapp.backend.routes import predict as predict_routes
from webapp.backend.routes import registry as registry_routes
from webapp.backend.routes import scenarios as scenarios_routes
from webapp.backend.routes import shap as shap_routes
from webapp.backend.routes import sweep as sweep_routes

logger = logging.getLogger(__name__)


API_TITLE = "roverdevkit tradespace API"
API_DESCRIPTION = (
    "Backend for the Phase-3 interactive tradespace exploration tool. "
    "Wraps the corrected mission evaluator and the W8 step-4 "
    "quantile-XGBoost surrogate. See project_plan.md §6 / Phase 3."
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
    app.include_router(pareto_routes.router)
    app.include_router(shap_routes.router)

    logger.info(
        "FastAPI app built (artifacts_present=%s, dataset_version=%s)",
        cfg.artifacts_present,
        cfg.dataset_version,
    )
    return app
