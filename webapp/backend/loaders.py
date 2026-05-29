"""Cached loaders for the immutable artifacts the API serves.

Everything here is built once per process and reused across requests.
The loaders are deliberately small wrappers around the existing
roverdevkit core so the cache invalidation story is "restart the
process" — there is no in-process model reloading endpoint by design
(simple, and matches the methodology paper's "frozen artifacts" story).

Rediscovery artifacts live as committed JSON under
``reports/rediscovery_loo_evaluator/`` and
``reports/rediscovery_loo_surrogate_v9/`` (one file per rover, plus
a manifest). The :func:`get_rediscovery_loo` loader reads them
lazily from disk so the server boots fast and degrades gracefully
when a fresh clone is missing the surrogate-backed sweep.

Cache strategy
--------------
Each loader uses :func:`functools.lru_cache(maxsize=1)`. That gives:

- Lazy initialisation (first request pays the cost),
- Single shared object across requests,
- Trivial unit-test reset via the ``cache_clear`` method on each
  loader.

Tests can also point the backend at alternate artifacts by setting the
``ROVERDEVKIT_QUANTILE_BUNDLES`` env var **before** the loader's first
call (or by calling :func:`reset_caches` after the env change).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib

from roverdevkit.mission.scenarios import list_scenarios, load_scenario
from roverdevkit.schema import MissionScenario, ScenarioName
from roverdevkit.surrogate.uncertainty import QuantileHeads
from roverdevkit.terramechanics.bekker_wong import SoilParameters
from roverdevkit.terramechanics.correction_model import (
    DEFAULT_CORRECTION_PATH,
    WheelLevelCorrection,
    load_correction_or_none,
)
from roverdevkit.terramechanics.soils import get_soil_parameters
from roverdevkit.validation.rover_registry import (
    RoverRegistryEntry,
    registry,
)
from webapp.backend.config import get_settings

logger = logging.getLogger(__name__)

RediscoveryBackend = Literal["evaluator", "surrogate"]
"""See :data:`webapp.backend.schemas.RediscoveryBackend` for details."""

_REDISCOVERY_SUBDIRS: dict[RediscoveryBackend, str] = {
    "evaluator": "rediscovery_loo_evaluator",
    "surrogate": "rediscovery_loo_surrogate_v9",
}
"""Subdirectory under ``reports/`` for each backend's per-rover JSONs.

Pinned to the v9 surrogate sweep: schema v9 promoted scientific
payload to two explicit mission-requirement inputs, so the
rediscovery harness now injects each rover's published payload onto
both the rover's re-evaluation and the NSGA-II candidates. Bumping
the surrogate dataset version requires a deliberate edit here so the
API never silently mixes calibration regimes."""

# Per-rover JSON files inside each backend directory have these stems.
# Hard-coded rather than glob-discovered so adding a new rover is an
# explicit registry change (and the route stays apples-to-apples
# across backends).
_REDISCOVERY_SLUGS: tuple[str, ...] = (
    "pragyan",
    "yutu_2",
    "moonranger",
    "rashid_1",
    "cadre_unit",
    "tenacious",
)


# ---------------------------------------------------------------------------
# Surrogate (quantile-calibration quantile-XGBoost bundles)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_quantile_bundles() -> dict[str, QuantileHeads]:
    """Return the ``{target -> QuantileHeads}`` dict from disk.

    Raises
    ------
    FileNotFoundError
        If the artifact does not exist at the configured path. The
        :http:get:`/healthz` route catches this and reports
        ``surrogate_loaded=False`` rather than crashing the process.
    TypeError
        If the artifact deserialises to something other than the
        expected ``dict[str, QuantileHeads]``.
    """
    settings = get_settings()
    path = settings.quantile_bundles_path
    if not path.exists():
        raise FileNotFoundError(
            f"quantile bundles artifact not found at {path}. "
            "Run scripts/calibrate_intervals.py to generate it."
        )
    obj: Any = joblib.load(path)
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict[str, QuantileHeads] at {path}; got {type(obj).__name__}")
    for target, head in obj.items():
        if not isinstance(head, QuantileHeads):
            raise TypeError(
                f"bundle entry {target!r} is not a QuantileHeads (got {type(head).__name__})."
            )
    logger.info("loaded quantile bundles for targets: %s", sorted(obj.keys()))
    return dict(obj)


# ---------------------------------------------------------------------------
# Scenarios (canonical four)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_canonical_scenarios() -> dict[ScenarioName, MissionScenario]:
    """Return the canonical scenarios keyed by name.

    Validation-only scenarios (Pragyan / Yutu-2 / Rashid-1 / etc.) are
    intentionally excluded -- those are exposed via ``/registry``.
    """
    return {name: load_scenario(name) for name in list_scenarios()}


@lru_cache(maxsize=1)
def get_soil_for_simulant(simulant_name: str) -> SoilParameters:
    """Return the nominal :class:`SoilParameters` for a named simulant.

    Wrapped in :func:`lru_cache` so the catalogue CSV is only re-parsed
    once per simulant per process. Shared with the predict path so the
    nominal soil block in the response matches what the surrogate sees
    for that scenario.
    """
    return get_soil_parameters(simulant_name)


# ---------------------------------------------------------------------------
# Registry (real-rover validation set)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_registry() -> tuple[RoverRegistryEntry, ...]:
    """Return the full real-rover registry (flown + design-target)."""
    return registry()


# ---------------------------------------------------------------------------
# Wheel-level SCM correction (optional; required for the corrected evaluator)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rediscovery LOO artifacts (Layer-5 paper figure)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def get_rediscovery_loo(backend: RediscoveryBackend) -> dict[str, dict[str, Any]]:
    """Return ``{slug -> rediscovery JSON dict}`` for the given backend.

    Reads the committed per-rover JSONs from
    ``reports/rediscovery_loo_<backend>/`` once per process per
    backend. Slugs not present on disk are silently skipped (a
    fresh clone may not yet have re-run the surrogate-backed sweep,
    and the route surfaces "no data" rather than 500-ing).

    Raises
    ------
    FileNotFoundError
        If the entire backend directory is missing -- in that case
        the route should return an empty list with the right backend
        echoed back, not a half-populated payload.
    """
    cfg = get_settings()
    sub = _REDISCOVERY_SUBDIRS[backend]
    base: Path = cfg.repo_root / "reports" / sub
    if not base.is_dir():
        raise FileNotFoundError(
            f"rediscovery LOO artifacts not found at {base}. "
            "Re-run scripts/run_rediscovery_loo.py for this backend."
        )
    out: dict[str, dict[str, Any]] = {}
    for slug in _REDISCOVERY_SLUGS:
        path = base / f"{slug}.json"
        if not path.exists():
            logger.warning(
                "rediscovery artifact %s missing for backend=%s; skipping",
                path,
                backend,
            )
            continue
        with path.open("r", encoding="utf-8") as fp:
            out[slug] = json.load(fp)
    if not out:
        raise FileNotFoundError(
            f"no per-rover rediscovery JSONs under {base}; "
            "expected one or more of "
            + ", ".join(f"{slug}.json" for slug in _REDISCOVERY_SLUGS)
        )
    logger.info(
        "loaded %d rediscovery LOO entries for backend=%s",
        len(out),
        backend,
    )
    return out


@lru_cache(maxsize=1)
def get_correction() -> WheelLevelCorrection | None:
    """Return the production wheel-level SCM correction artifact, if available.

    The artifact lives at
    :data:`roverdevkit.terramechanics.correction_model.DEFAULT_CORRECTION_PATH`
    and is shared by every ``/evaluate`` call so the joblib load only
    happens once per process. Returns ``None`` when the file is missing
    so the route can fall back to the BW-only path with an explicit
    error rather than silently degrading to a different physics model.
    """
    return load_correction_or_none(DEFAULT_CORRECTION_PATH, on_missing="warn")


# ---------------------------------------------------------------------------
# Test / dev helpers
# ---------------------------------------------------------------------------


def reset_caches() -> None:
    """Clear every backend-level cache. Used by tests and ``/healthz`` retries."""
    get_quantile_bundles.cache_clear()
    get_canonical_scenarios.cache_clear()
    get_soil_for_simulant.cache_clear()
    get_registry.cache_clear()
    get_correction.cache_clear()
    get_rediscovery_loo.cache_clear()
