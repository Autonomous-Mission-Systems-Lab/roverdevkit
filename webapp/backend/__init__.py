"""FastAPI backend for the webapp tradespace exploration tool.

The package is intentionally thin: every route delegates to the
existing :mod:`roverdevkit` core (mission evaluator, surrogate,
validation registry) so the web app cannot drift from the methodology
paper's reported numbers. The backend is intentionally thin so scripts,
notebooks, and the browser use the same evaluator and surrogate artifacts.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app():  # type: ignore[no-untyped-def]
    """Re-export of :func:`webapp.backend.app.create_app` for convenience."""
    from webapp.backend.app import create_app as _factory

    return _factory()
