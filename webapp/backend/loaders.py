"""Cached loaders for the immutable artifacts the API serves.

Everything here is built once per process and reused across requests.
The loaders are deliberately small wrappers around the existing
roverdevkit core so the cache invalidation story is "restart the
process" — there is no in-process model reloading endpoint by design
(simple, and matches the methodology paper's "frozen artifacts" story).

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

import logging
from functools import lru_cache
from typing import Any

import joblib

from roverdevkit.mission.scenarios import list_scenarios, load_scenario
from roverdevkit.schema import MissionScenario, ScenarioName
from roverdevkit.surrogate.uncertainty import QuantileHeads
from roverdevkit.terramechanics.bekker_wong import SoilParameters
from roverdevkit.terramechanics.soils import get_soil_parameters
from roverdevkit.validation.rover_registry import (
    RoverRegistryEntry,
    registry,
)
from webapp.backend.config import get_settings

logger = logging.getLogger(__name__)


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
# Test / dev helpers
# ---------------------------------------------------------------------------


def reset_caches() -> None:
    """Clear every backend-level cache. Used by tests and ``/healthz`` retries."""
    get_quantile_bundles.cache_clear()
    get_canonical_scenarios.cache_clear()
    get_soil_for_simulant.cache_clear()
    get_registry.cache_clear()
