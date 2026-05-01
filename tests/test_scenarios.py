"""Tests for the mission-scenario YAML loader."""

from __future__ import annotations

import pytest

from roverdevkit.mission.scenarios import list_scenarios, load_scenario
from roverdevkit.schema import MissionScenario
from roverdevkit.terramechanics.soils import list_soil_simulants

EXPECTED_SCENARIOS = {
    "equatorial_mare_traverse",
    "polar_prospecting",
    "highland_slope_capability",
    "crater_rim_survey",
}


def test_list_scenarios_returns_all_four_canonical_scenarios() -> None:
    names = set(list_scenarios())
    missing = EXPECTED_SCENARIOS - names
    assert not missing, f"missing scenarios: {missing}"


@pytest.mark.parametrize("name", sorted(EXPECTED_SCENARIOS))
def test_load_scenario_round_trips_to_pydantic_model(name: str) -> None:
    scenario = load_scenario(name)
    assert isinstance(scenario, MissionScenario)
    assert scenario.name == name


@pytest.mark.parametrize("name", sorted(EXPECTED_SCENARIOS))
def test_soil_simulant_in_every_scenario_is_in_the_catalogue(name: str) -> None:
    # The traverse sim resolves soil names via the catalogue; if a
    # scenario references an unknown simulant the evaluator will crash
    # later. Catch it at config-load time.
    scenario = load_scenario(name)
    assert scenario.soil_simulant in list_soil_simulants()


def test_unknown_scenario_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="scenario config"):
        load_scenario("nonexistent_scenario")


def test_equatorial_scenario_has_expected_fields() -> None:
    s = load_scenario("equatorial_mare_traverse")
    assert s.latitude_deg == pytest.approx(20.2)
    assert s.mission_duration_earth_days == pytest.approx(14.0)
    assert s.traverse_distance_m > 0


def test_polar_scenario_has_high_latitude() -> None:
    s = load_scenario("polar_prospecting")
    assert abs(s.latitude_deg) >= 70.0
    assert s.sun_geometry == "polar_intermittent"


# ---------------------------------------------------------------------------
# Schema v6 (W12 step B): operational_duty_cycle calibration on YAMLs
# ---------------------------------------------------------------------------
# Pin the four canonical scenarios to the calibration agreed in
# reports/week12_design/decision.md so an accidental YAML edit doesn't
# silently shift the surrogate's training distribution. Mare 0.30,
# crater 0.20, highland 0.15, polar 0.05 — see decision.md §"δ_ops
# calibration" for the published-rover-anchored derivation.

_EXPECTED_DOPS = {
    "equatorial_mare_traverse": 0.30,
    "crater_rim_survey": 0.20,
    "highland_slope_capability": 0.15,
    "polar_prospecting": 0.05,
}


@pytest.mark.parametrize("name,expected", sorted(_EXPECTED_DOPS.items()))
def test_canonical_scenario_operational_duty_cycle_matches_calibration(
    name: str, expected: float
) -> None:
    """``operational_duty_cycle`` on each canonical scenario YAML matches
    the W12 step B calibration. Catch silent YAML drift before it
    contaminates the LHS dataset."""
    s = load_scenario(name)
    assert s.operational_duty_cycle == pytest.approx(expected, abs=1e-9), (
        f"{name}: operational_duty_cycle drifted from the W12 calibration"
    )
