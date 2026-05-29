"""Shared pytest fixtures.

Session-scoped fixtures cache expensive evaluator runs across modules so
the integration suites (cross-scenario sensitivity, real-rover
validation) don't re-run the same physics dozens of times. The
non-cached version of the suite was ~580 s; with these fixtures it
drops to under 60 s on a typical developer laptop.
"""

from __future__ import annotations

import pytest

from roverdevkit.schema import DesignVector, MissionScenario
from roverdevkit.validation.cross_scenario import (
    ArchetypeRanking,
    SensitivityEntry,
    one_at_a_time_sensitivity,
    rank_archetypes,
)
from roverdevkit.validation.rover_comparison import (
    ComparisonSummary,
    RoverComparisonResult,
    compare_all,
    compare_one,
)
from roverdevkit.validation.rover_registry import flown_registry, registry_by_name


@pytest.fixture
def rashid_like_design() -> DesignVector:
    """A Rashid-like design vector for tests and worked examples.

    Numbers chosen to match published Rashid specs where available
    (see data/published_rovers.csv) and reasonable defaults otherwise.
    """
    return DesignVector(
        wheel_radius_m=0.1,
        wheel_width_m=0.06,
        grouser_height_m=0.005,
        grouser_count=12,
        n_wheels=4,
        chassis_mass_kg=6.0,
        wheelbase_m=0.35,
        solar_area_m2=0.4,
        battery_capacity_wh=100.0,
        avionics_power_w=15.0,
        peak_wheel_torque_nm=1.5,
    )


@pytest.fixture
def equatorial_scenario() -> MissionScenario:
    return MissionScenario(
        name="equatorial_mare_traverse",
        latitude_deg=20.2,
        traverse_distance_m=5000.0,
        terrain_class="mare_nominal",
        soil_simulant="Apollo_regolith_nominal",
        mission_duration_earth_days=14.0,
        max_slope_deg=15.0,
        sun_geometry="diurnal",
        operational_duty_cycle=0.30,
    )


# ---------------------------------------------------------------------------
# Session-scoped evaluator caches
# ---------------------------------------------------------------------------
# These fixtures run the evaluator once per test session and let every
# downstream test consume the same precomputed results. They are pure
# (no test-induced state); reusing them across tests is safe because the
# evaluator is deterministic. If a test needs a *different* evaluator
# call, it should not depend on these fixtures and pay its own cost.


@pytest.fixture(scope="session")
def cross_scenario_rankings() -> dict[str, ArchetypeRanking]:
    """Cached :func:`rank_archetypes` output (12 evaluator runs)."""
    return rank_archetypes()


@pytest.fixture(scope="session")
def cross_scenario_sensitivity() -> list[SensitivityEntry]:
    """Cached :func:`one_at_a_time_sensitivity` output (~24 evaluator runs)."""
    return one_at_a_time_sensitivity()


@pytest.fixture(scope="session")
def cross_scenario_sensitivity_by_var(
    cross_scenario_sensitivity: list[SensitivityEntry],
) -> dict[str, SensitivityEntry]:
    """Sensitivity entries keyed by variable name."""
    return {e.variable: e for e in cross_scenario_sensitivity}


@pytest.fixture(scope="session")
def rover_compare_summary() -> ComparisonSummary:
    """Cached :func:`compare_all` output (one evaluator run per rover)."""
    return compare_all()


@pytest.fixture(scope="session")
def rover_compare_results(
    rover_compare_summary: ComparisonSummary,
) -> dict[str, RoverComparisonResult]:
    """Per-rover comparison results from the cached summary."""
    return {r.rover_name: r for r in rover_compare_summary.results}


@pytest.fixture(scope="session")
def registered_rover_names() -> list[str]:
    """Stable list of rover names (the parametrize ids)."""
    return [e.rover_name for e in flown_registry()]


# ``registry_by_name`` and ``compare_one`` are module-level helpers; we
# re-export them as symbols so tests can keep their existing call sites
# without changing imports during the refactor. They're intentionally
# *not* fixtures (they take arguments).
__all__ = ["compare_one", "registry_by_name"]
